"""Hugging Face backend for the pipeline's `Generator` interface."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.bfloat16


class HFGenerator:
    def __init__(self, model, tokenizer):
        """Wrap an already-loaded model, e.g. from the steering loader in a notebook."""
        self.model, self.tokenizer = model.eval(), tokenizer
        self.tokenizer.padding_side = "left"  # decoder-only batches must end at the prompt
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def load(cls, model_id=MODEL_ID, load_4bit=False):
        """Load a checkpoint; `load_4bit` matches the steering setup (CUDA only)."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                              bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                         quantization_config=quantization, dtype=torch.bfloat16)
        else:
            device, dtype = pick_device()
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        return cls(model, tokenizer)

    def generate(self, conversations, *, temperature, max_new_tokens, seed):
        torch.manual_seed(seed)
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 for messages in conversations]
        # The chat template already adds the start-of-text token.
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {
            "do_sample": False, "temperature": None, "top_p": None}
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tokenizer.pad_token_id, **sampling)
        return self.tokenizer.batch_decode(tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
