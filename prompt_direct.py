import torch


'''
Inputs: the 5 numerical values of OCEAN scores of a desired personality
Output: string personality prompt for LLM
'''
def make_personality(O, C, E, A, N):
    str = (f"You are a person with the following personality type:"
           f"{O}/5 openness, {C}/5 conscientiousness, {E}/5 extraversion, "
           f"{A}/5 agreeableness, and {N}/5 neuroticism.")
    return str

'''
Inputs:
    model: LLM model
    tokenizer: tokenizer
    personality: prompt for the model's personality traits that are desired
    prompt: prompt from BFI test, asking what model would do in a situation
    max_new_tokens: maximum number of tokens allowed for the prompt
Output: string response of the model's answer
'''
def generate_response(model, tokenizer, personality, prompt, max_new_tokens=200):
    messages = [
        {"role": "system", "content": personality},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_tokens = outputs[0][inputs.shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    return response.strip()