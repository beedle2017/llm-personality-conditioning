from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from eval.behaviorbench import evaluate_behaviorbench


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors, padding=False):
        del messages, tokenize, add_generation_prompt, return_tensors, padding
        return torch.tensor([[1, 2]])

    def decode(self, tokens, skip_special_tokens):
        del tokens, skip_special_tokens
        return "Yes"


class FakeModel:
    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids, **kwargs):
        del kwargs
        return torch.cat((input_ids, torch.tensor([[3]])), dim=1)


def test_evaluate_behaviorbench_writes_endorsement_rows(tmp_path):
    data_path = tmp_path / "test.json"
    output_path = tmp_path / "results.csv"
    data_path.write_text(json.dumps([{
        "id": "bb-O-001-01",
        "trait": "Openness",
        "response_type": "yes_no",
        "scenario": "Try a new activity?",
        "options": [
            {"key": "Yes", "text": "Yes", "endorsement": 1},
            {"key": "No", "text": "No", "endorsement": 0},
        ],
    }]))

    rows = evaluate_behaviorbench(
        FakeModel(), FakeTokenizer(), output_path,
        data_path=data_path, traits=("Openness",), target_scores=(3,),
    )

    assert rows[0]["endorsement_rate"] == 1.0
    assert rows[0]["invalid_rate"] == 0.0
    assert rows[0]["E_tau_t"] == ""
    assert output_path.exists()