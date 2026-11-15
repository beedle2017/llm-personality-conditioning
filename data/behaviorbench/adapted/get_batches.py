"""Utilities for creating batches from ./data/test.json (relative to this file).

The file is expected to contain a JSON array of objects, each with an "id" field.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

DATA_FILE = Path(__file__).resolve().parent / "data" / "test.json"
DEFAULT_SEED = 42  # fixed default => repeated runs give identical output


def load_jsons() -> List[Dict[str, Any]]:
    """Load the JSON array from ./data/test.json (relative to this file)."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected the JSON file to contain a top-level array.")
    return data


def get_all_ids() -> List[Any]:
    """Return all ids present in the JSON file (in file order)."""
    return [item["id"] for item in load_jsons()]


def batches_of_size(
    n: int,
    seed: int = DEFAULT_SEED,
) -> Iterator[List[Dict[str, Any]]]:
    """Infinite generator: each yield is a random batch (list) of `n` jsons.

    Every batch is drawn independently from the full set, so a json can appear
    in many batches, but never twice within the same batch. The sequence of
    batches is fully determined by `seed`.
    """
    items = load_jsons()
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if n > len(items):
        raise ValueError(f"n={n} is larger than the number of jsons ({len(items)}).")

    rng = random.Random(seed)
    while True:
        yield rng.sample(items, n)


def batches_ending_with_id(
    id: Any,
    seed: int = DEFAULT_SEED,
) -> Iterator[List[Dict[str, Any]]]:
    """Yield lists of increasing length (1, 2, 3, ...), each ending with the
    json whose id is `id`.

    The other jsons are shuffled using `seed`; the k-th yield contains the first
    (k-1) of them followed by the target json. So each list extends the previous
    one, and the last list contains every json in the file.
    """
    items = load_jsons()

    target_idx = next((i for i, item in enumerate(items) if item.get("id") == id), None)
    if target_idx is None:
        raise ValueError(f"No json with id {id!r} found in {DATA_FILE}.")

    target = items[target_idx]
    others = items[:target_idx] + items[target_idx + 1:]
    random.Random(seed).shuffle(others)

    for k in range(len(others) + 1):
        yield others[:k] + [target]