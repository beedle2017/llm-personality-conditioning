"""Bi-directional preference optimization for activation steering."""

from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import DEFAULT_SEED, load_trait_data

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LAYER = 15
DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4


def load_model(
	model_id: str = MODEL_ID,
) -> tuple[Any, Any]:
	"""Load the Llama model and tokenizer used by the notebook.

	The returned model is placed on CUDA, MPS, or CPU using the same device
	and dtype selection as ``notebooks/load_llama3_1_8b_instruct.ipynb``.
	"""
	if torch.cuda.is_available():
		device, dtype = "cuda", torch.bfloat16
	elif torch.backends.mps.is_available():
		device, dtype = "mps", torch.float16
	else:
		device, dtype = "cpu", torch.bfloat16

	tokenizer = AutoTokenizer.from_pretrained(model_id)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
	model: Any = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
	model = model.to(device)
	return model, tokenizer


def _model_device(model: Any) -> torch.device:
	return next(model.parameters()).device


def _model_layer(model: Any, layer: int) -> nn.Module:
	"""Return a decoder layer for Llama-style causal language models."""
	try:
		return model.model.layers[layer]
	except AttributeError as error:
		raise ValueError("model must expose decoder layers at model.model.layers") from error


def _add_vector(output: Any, vector: Tensor) -> Any:
	"""Add ``vector`` to a decoder layer output without dropping metadata."""
	if isinstance(output, Tensor):
		return output + vector.to(device=output.device, dtype=output.dtype)
	if isinstance(output, tuple):
		hidden = output[0] + vector.to(device=output[0].device, dtype=output[0].dtype)
		return (hidden, *output[1:])
	raise TypeError(f"Unsupported decoder layer output type: {type(output).__name__}")


@contextmanager
def _steering_hook(layer: nn.Module, vector: Tensor, direction: float) -> Iterator[None]:
	def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
		return _add_vector(output, direction * vector)

	handle = layer.register_forward_hook(hook)
	try:
		yield
	finally:
		handle.remove()


def _response_logprob(model: nn.Module, input_ids: Tensor, response_start: int) -> Tensor:
	outputs = model(input_ids=input_ids)
	log_probs = outputs.logits.log_softmax(dim=-1)
	response_tokens = input_ids[:, response_start:]
	if response_tokens.shape[1] == 0:
		raise ValueError("Each preference response must contain at least one token")
	token_log_probs = log_probs[:, response_start - 1 : -1].gather(
		-1, response_tokens.unsqueeze(-1)
	)
	return token_log_probs.squeeze(-1).sum(dim=-1)


def _encoded_response(
	tokenizer: Any, prompt: str, response: str, device: torch.device, chat_template: bool
) -> tuple[Tensor, int]:
	if chat_template:
		prompt_ids = tokenizer.apply_chat_template(
			[{"role": "user", "content": prompt}],
			tokenize=True,
			add_generation_prompt=True,
			return_tensors="pt",
		)
		prompt_ids = prompt_ids["input_ids"] if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "keys") else prompt_ids
		input_ids = tokenizer.apply_chat_template(
			[
				{"role": "user", "content": prompt},
				{"role": "assistant", "content": response},
			],
			tokenize=True,
			add_generation_prompt=False,
			return_tensors="pt",
		)
		input_ids = input_ids["input_ids"] if isinstance(input_ids, dict) or hasattr(input_ids, "keys") else input_ids
		return input_ids.to(device), prompt_ids.shape[-1]
	prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
	response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
	input_ids = torch.cat((prompt_ids, response_ids), dim=-1).to(device)
	return input_ids, prompt_ids.shape[-1]


def _freeze_model(model: nn.Module) -> dict[nn.Parameter, bool]:
	previous = {parameter: parameter.requires_grad for parameter in model.parameters()}
	for parameter in model.parameters():
		parameter.requires_grad_(False)
	return previous


def _restore_model(model: nn.Module, previous: dict[nn.Parameter, bool]) -> None:
	for parameter, requires_grad in previous.items():
		parameter.requires_grad_(requires_grad)


def steering(
	model: Any,
	trait: str,
	tokenizer: Any | None = None,
	*,
	data_dir: str | Path = "data/pairs",
	layer: int = DEFAULT_LAYER,
	beta: float = DEFAULT_BETA,
	learning_rate: float = DEFAULT_LEARNING_RATE,
	weight_decay: float = DEFAULT_WEIGHT_DECAY,
	epochs: int = DEFAULT_EPOCHS,
	batch_size: int = DEFAULT_BATCH_SIZE,
	seed: int = DEFAULT_SEED,
	chat_template: bool = True,
) -> Tensor:
	"""Train and return a BiPO steering vector for ``trait``.

	Only ``TraitData.train`` is used for optimization. The returned tensor has
	shape ``(hidden_size,)`` and is suitable for broadcast addition to every
	token position at ``layer`` during generation.
	"""
	if epochs < 1 or batch_size < 1:
		raise ValueError("epochs and batch_size must be positive")
	if beta <= 0 or learning_rate <= 0:
		raise ValueError("beta and learning_rate must be positive")

	trait_data = load_trait_data(trait, data_dir)
	if not trait_data.train:
		raise ValueError(f"No training rows available for trait {trait!r}")

	tokenizer_obj: Any = tokenizer
	if tokenizer_obj is None:
		model_id = getattr(model, "name_or_path", MODEL_ID)
		tokenizer_obj = AutoTokenizer.from_pretrained(model_id)
	if tokenizer_obj.pad_token_id is None:
		tokenizer_obj.pad_token = tokenizer_obj.eos_token

	device = _model_device(model)
	hidden_size = int(model.config.hidden_size)
	vector = torch.zeros(hidden_size, device=device, dtype=torch.float32, requires_grad=True)
	optimizer = torch.optim.AdamW([vector], lr=learning_rate, weight_decay=weight_decay)
	decoder_layer = _model_layer(model, layer)
	previous_requires_grad = _freeze_model(model)
	model.eval()
	rng = random.Random(seed)

	try:
		for _ in range(epochs):
			rows = list(trait_data.train)
			rng.shuffle(rows)
			for start in range(0, len(rows), batch_size):
				# print(f"Processing row {start} with batch size {batch_size}")
				batch = rows[start : start + batch_size]
				optimizer.zero_grad(set_to_none=True)
				losses: list[Tensor] = []
				for row in batch:
					prompt = row.get("prompt")
					positive = row.get("pos")
					negative = row.get("neg")
					if not all(isinstance(value, str) and value for value in (prompt, positive, negative)):
						raise ValueError("Each training row must contain non-empty prompt, pos, and neg strings")
					prompt_text = cast(str, prompt)
					positive_text = cast(str, positive)
					negative_text = cast(str, negative)

					positive_ids, positive_start = _encoded_response(
						tokenizer_obj, prompt_text, positive_text, device, chat_template
					)
					negative_ids, negative_start = _encoded_response(
						tokenizer_obj, prompt_text, negative_text, device, chat_template
					)
					with torch.no_grad():
						positive_base = _response_logprob(model, positive_ids, positive_start)
						negative_base = _response_logprob(model, negative_ids, negative_start)

					direction = 1.0 if rng.randrange(2) == 1 else -1.0
					with _steering_hook(decoder_layer, vector, direction):
						positive_steered = _response_logprob(model, positive_ids, positive_start)
						negative_steered = _response_logprob(model, negative_ids, negative_start)
					preference_margin = (positive_steered - positive_base) - (
						negative_steered - negative_base
					)
					losses.append(-torch.nn.functional.logsigmoid(beta * direction * preference_margin))

				torch.stack(losses).mean().backward()
				optimizer.step()
	finally:
		_restore_model(model, previous_requires_grad)

	return vector.detach()


__all__ = ["MODEL_ID", "load_model", "steering"]
