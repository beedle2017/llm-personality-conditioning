from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "behaviorbench" / "adapted"))

from eval_behaviorbench_caa import counterbalance_item
from prepare_data import score_response


def test_counterbalance_is_deterministic() -> None:
    item = {
        "id": "item-1",
        "response_type": "forced_choice",
        "options": [
            {"key": "A", "text": "first", "endorsement": 1},
            {"key": "B", "text": "second", "endorsement": 0},
        ],
    }
    assert counterbalance_item(item, 20261026, 1) == counterbalance_item(item, 20261026, 1)


def test_invalid_response_is_not_scored_as_no() -> None:
    item = {
        "options": [
            {"key": "Yes", "text": "Yes", "endorsement": 1},
            {"key": "No", "text": "No", "endorsement": 0},
        ]
    }
    assert score_response(item, "maybe") is None
