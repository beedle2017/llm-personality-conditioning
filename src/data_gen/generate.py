"""Resumable scenario and contrastive-pair generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import TRAITS, CONTEXTS
from .llm_client import LLMClient, QuotaExhausted, parse_json_response

LOGGER = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).parent / "prompts"


def _template(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def append_jsonl(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _valid_batch(items: Any, required: tuple[str, ...], expected: int) -> bool:
    return isinstance(items, list) and len(items) == expected and all(
        isinstance(item, dict) and all(isinstance(item.get(key), str) and item[key].strip() for key in required)
        for item in items
    )


def generate_trait(trait: str, target: int, client: LLMClient, output_dir: str | Path, seed: int, dry_run: bool = False) -> Path:
    if trait not in TRAITS:
        raise ValueError(f"Unknown trait: {trait}")
    path = Path(output_dir) / f"{trait}.jsonl"
    existing = read_jsonl(path)
    needed = max(0, target - len(existing))
    if dry_run:
        print(_template("scenario.txt").format(count=5, trait=trait, facet=TRAITS[trait][0], context=CONTEXTS[0], length_band="short"))
        return path
    offset = len(existing)
    while needed:
        facet = TRAITS[trait][offset % len(TRAITS[trait])]
        context = CONTEXTS[offset % len(CONTEXTS)]
        band = "short" if offset % 2 == 0 else "medium"
        count = min(client.config.batch_pairs_per_call, 10, needed)
        scenario_prompt = _template("scenario.txt").format(count=count, trait=trait, facet=facet, context=context, length_band=band)
        completed = False
        for attempt in range(client.config.max_retries + 1):
            try:
                scenarios = parse_json_response(client.complete(scenario_prompt, client.config.generator_model))
                if not _valid_batch(scenarios, ("prompt", "context", "facet", "length_band"), count):
                    raise ValueError(f"Expected {count} valid scenarios")
                pair_prompt = _template("pairs.txt").format(trait=trait, length_band=band, scenarios=json.dumps(scenarios))
                pairs = parse_json_response(client.complete(pair_prompt, client.config.generator_model))
                if not _valid_batch(pairs, ("prompt", "pos", "neg", "facet", "context", "length_band"), count):
                    raise ValueError(f"Expected {count} valid pairs")
                pairs = [dict(pair, trait=trait) for pair in pairs]
                append_jsonl(path, pairs)
                needed -= len(pairs)
                completed = True
                break
            except QuotaExhausted:
                LOGGER.warning("Daily quota exhausted; saved progress in %s", path)
                return path
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt >= client.config.max_retries:
                    raise
                LOGGER.warning("Invalid batch for %s (attempt %d): %s", trait, attempt + 1, exc)
        if not completed:
            break
        offset += 1
    return path
