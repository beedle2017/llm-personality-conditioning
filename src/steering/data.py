"""Load contrastive personality data with deterministic train/dev handling."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
DEFAULT_SEED = 42
DEFAULT_DEV_FRACTION = 0.2
MIN_DEV_ROWS = 50


@dataclass(frozen=True)
class TraitData:
    """A trait's training and evaluation rows.

    Only ``train`` should be used to compute a steering vector. ``dev`` is
    reserved for selecting layers and alpha values.
    """

    trait: str
    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _split_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row.get("split") or "missing" for row in rows)


def _stratified_indices(
    rows: list[dict[str, Any]], n_dev: int, seed: int
) -> set[int]:
    """Choose exactly ``n_dev`` row indices, stratified by context."""
    if n_dev <= 0:
        return set()
    if n_dev >= len(rows):
        return set(range(len(rows)))

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("context", ""))].append(index)

    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    total = len(rows)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for context in sorted(groups):
        exact = len(groups[context]) * n_dev / total
        count = min(len(groups[context]), int(exact))
        allocations[context] = count
        allocated += count
        remainders.append((exact - count, context))

    for _, context in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : n_dev - allocated
    ]:
        allocations[context] += 1

    selected: set[int] = set()
    for context, indices in groups.items():
        selected.update(indices[: allocations[context]])
    return selected


def _make_fallback_split(
    rows: list[dict[str, Any]], seed: int, dev_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_dev = [row for row in rows if row.get("split") == "dev"]
    pool = [row for row in rows if row.get("split") in {"train", None, ""}]
    n_dev = round(len(pool) * dev_fraction)
    selected = _stratified_indices(pool, n_dev, seed)

    train = [dict(row, split="train") for index, row in enumerate(pool) if index not in selected]
    dev = [dict(row, split="dev") for row in existing_dev]
    dev.extend(dict(pool[index], split="dev") for index in sorted(selected))
    return train, dev


def load_trait_data(
    trait: str,
    data_dir: str | Path = "data/pairs",
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    min_dev_rows: int = MIN_DEV_ROWS,
) -> TraitData:
    """Load one trait, preserving usable splits or creating a deterministic dev set."""
    if trait not in TRAITS:
        raise ValueError(f"Unknown trait {trait!r}; expected one of {', '.join(TRAITS)}")

    path = Path(data_dir) / f"{trait}.jsonl"
    rows = _read_rows(path)
    counts = _split_counts(rows)
    print(
        f"{trait}: rows={len(rows)} "
        f"train={counts.get('train', 0)} dev={counts.get('dev', 0)} "
        f"test={counts.get('test', 0)} missing={counts.get('missing', 0)}"
    )

    has_complete_splits = (
        all(row.get("split") not in (None, "") for row in rows)
        and counts.get("train", 0) > 0
        and counts.get("dev", 0) > 0
        and counts.get("dev", 0) >= min_dev_rows
    )
    if has_complete_splits:
        train = [row for row in rows if row.get("split") == "train"]
        dev = [row for row in rows if row.get("split") == "dev"]
    else:
        LOGGER.info(
            "%s: creating deterministic dev split (seed=%d, fraction=%.2f); "
            "using existing dev rows and stratifying unsplit/train rows",
            trait,
            seed,
            dev_fraction,
        )
        train, dev = _make_fallback_split(rows, seed, dev_fraction)

    print(f"{trait}: final train={len(train)} dev={len(dev)}")
    return TraitData(trait=trait, train=train, dev=dev)


def load_all_traits(
    data_dir: str | Path = "data/pairs",
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    min_dev_rows: int = MIN_DEV_ROWS,
) -> dict[str, TraitData]:
    """Load all five personality traits using the same split rule."""
    return {
        trait: load_trait_data(
            trait,
            data_dir,
            seed=seed,
            dev_fraction=dev_fraction,
            min_dev_rows=min_dev_rows,
        )
        for trait in TRAITS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_all_traits(args.data_dir, seed=args.seed)


if __name__ == "__main__":
    main()
