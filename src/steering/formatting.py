"""Shared Llama chat formatting for CAA and BiPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FormattedPair:
    input_ids: Any
    attention_mask: Any
    response_mask: Any


def format_pair(tokenizer: Any, prompt: str, response: str) -> FormattedPair:
    """Format one user/assistant pair and mask assistant response tokens only."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    input_ids = full if full.ndim == 2 else full.unsqueeze(0)
    prefix_ids = prefix if prefix.ndim == 2 else prefix.unsqueeze(0)
    response_start = prefix_ids.shape[-1]
    response_mask = input_ids.new_zeros(input_ids.shape, dtype=__import__("torch").bool)
    response_mask[:, response_start:] = True
    return FormattedPair(
        input_ids=input_ids,
        attention_mask=__import__("torch").ones_like(input_ids),
        response_mask=response_mask,
    )
