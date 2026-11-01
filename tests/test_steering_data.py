"""Tests for the contrastive steering data loader."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from steering.data import load_trait_data


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(index: int, split: str | None, context: str) -> dict[str, str]:
    row = {"id": str(index), "context": context, "prompt": f"prompt {index}"}
    if split is not None:
        row["split"] = split
    return row


def test_preserves_complete_splits(tmp_path: Path) -> None:
    rows = [_row(index, "dev" if index < 50 else "train", "a") for index in range(60)]
    _write_rows(tmp_path / "conscientiousness.jsonl", rows)

    data = load_trait_data("conscientiousness", tmp_path)

    assert len(data.train) == 10
    assert len(data.dev) == 50
    assert {row["id"] for row in data.dev} == {str(index) for index in range(50)}


def test_fallback_is_deterministic_stratified_and_preserves_existing_dev(
    tmp_path: Path, caplog
) -> None:
    rows = [_row(index, "dev" if index < 2 else None, "a" if index % 2 else "b") for index in range(12)]
    _write_rows(tmp_path / "openness.jsonl", rows)

    with caplog.at_level(logging.INFO, logger="steering.data"):
        first = load_trait_data("openness", tmp_path, seed=7)
    second = load_trait_data("openness", tmp_path, seed=7)

    assert [(row["id"], row["split"]) for row in first.dev] == [
        (row["id"], row["split"]) for row in second.dev
    ]
    assert {"0", "1"}.issubset({row["id"] for row in first.dev})
    assert len(first.dev) == 4
    assert len(first.train) == 8
    assert "creating deterministic dev split" in caplog.text
