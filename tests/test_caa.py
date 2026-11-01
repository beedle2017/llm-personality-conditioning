"""Contract tests for shared CAA/BiPO steering components."""

from __future__ import annotations

import torch

from steering.common_config import score_to_coeff
from steering.formatting import format_pair
from steering.hooks import steering_hook


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
        del tokenize, return_tensors
        length = 4 if add_generation_prompt else 7
        if messages[-1]["role"] == "assistant":
            length = 7
        return torch.arange(length).unsqueeze(0)


def test_response_mask_excludes_prompt_and_template_tokens() -> None:
    formatted = format_pair(FakeTokenizer(), "prompt", "response")
    assert formatted.response_mask.tolist() == [[False, False, False, False, True, True, True]]


def test_score_zero_and_t_three_are_unsteered() -> None:
    assert score_to_coeff(3, 5) == 0
    hidden = torch.zeros(2, 4, 3)
    hook = steering_hook(0, torch.ones(3))
    assert torch.equal(hook(None, None, hidden), hidden)


def test_score_to_coeff_is_antisymmetric() -> None:
    assert score_to_coeff(1, 2) == -score_to_coeff(5, 2)
    assert score_to_coeff(2, 2) == -score_to_coeff(4, 2)


def test_hook_adds_at_all_positions() -> None:
    hidden = torch.zeros(2, 4, 3)
    output = steering_hook(2, torch.tensor([1.0, 2.0, 3.0]))(None, None, hidden)
    expected = torch.tensor([2.0, 4.0, 6.0]).expand_as(hidden)
    assert torch.equal(output, expected)
