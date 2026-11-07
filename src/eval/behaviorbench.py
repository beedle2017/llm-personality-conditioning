"""Fast, model-agnostic evaluation on adapted BehaviorBench."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "data" / "behaviorbench" / "adapted"
import sys

sys.path.insert(0, str(BENCHMARK_ROOT))
from prepare_data import score_response  # noqa: E402

TRAITS = ("Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism")
RESULT_FIELDS = (
    "method", "trait", "split", "target_score", "endorsement_rate",
    "invalid_rate", "n", "valid_n", "E_tau_t",
)
SteeringFactory = Callable[[Any, Any, str, int], ContextManager[Any] | None]
TraitScore = Callable[[str, int, float], float]


def _stable_seed(item_id: str, target: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{target}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _prompt(item: dict[str, Any]) -> str:
    choices = "Yes or No" if item["response_type"] == "yes_no" else "A or B"
    return (
        "Answer using exactly one requested choice. "
        f"Reply with only {choices}, with no explanation.\n\n{item['scenario']}"
    )


def _display_item(item: dict[str, Any], target: int, seed: int) -> dict[str, Any]:
    """Counterbalance forced-choice display while preserving source endorsements."""
    if item["response_type"] != "forced_choice":
        return item
    if random.Random(_stable_seed(item["id"], target, seed)).randrange(2) == 0:
        return item
    displayed = dict(item)
    displayed["options"] = [
        dict(option, key=key)
        for key, option in zip(("A", "B"), reversed(item["options"]))
    ]
    return displayed


def _generate(
    model: Any,
    tokenizer: Any,
    items: Iterable[dict[str, Any]],
    steering_factory: SteeringFactory | None,
    trait: str,
    target: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    device = next(model.parameters()).device
    context = steering_factory(model, tokenizer, trait, target) if steering_factory else None
    with context if context is not None else nullcontext():
        responses = []
        items = list(items)
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            encoded = tokenizer.apply_chat_template(
                [[{"role": "user", "content": _prompt(item)}] for item in batch],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
            )
            input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            input_ids = input_ids.to(device)
            attention_mask = getattr(encoded, "attention_mask", None)
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape)
            with __import__("torch").inference_mode():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask.to(device),
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[:, input_ids.shape[-1]:]
            if hasattr(tokenizer, "batch_decode"):
                responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            else:
                responses.extend(tokenizer.decode(row, skip_special_tokens=True) for row in generated)
    return [response.strip() for response in responses]


def evaluate_behaviorbench(
    model: Any,
    tokenizer: Any,
    output_csv: str | Path,
    *,
    data_path: str | Path = BENCHMARK_ROOT / "data" / "test.json",
    method: str = "model",
    split: str = "test",
    traits: tuple[str, ...] = TRAITS,
    target_scores: tuple[int, ...] = (1, 2, 3, 4, 5),
    steering_factory: SteeringFactory | None = None,
    trait_score: TraitScore | None = None,
    seed: int = 20261026,
    max_new_tokens: int = 8,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Evaluate any loaded model with optional steering on BehaviorBench.

    ``steering_factory`` returns an optional context manager for each
    trait/target pair. ``trait_score`` converts endorsement to a Big Five score
    when that mapping is explicitly defined; otherwise ``E_tau_t`` is blank.
    """
    items = json.loads(Path(data_path).read_text(encoding="utf-8"))
    items = [item for item in items if item.get("trait") in traits]
    rows: list[dict[str, Any]] = []
    for trait in traits:
        trait_items = [item for item in items if item["trait"] == trait]
        for target in target_scores:
            displayed = [_display_item(item, target, seed) for item in trait_items]
            responses = _generate(
                model, tokenizer, displayed, steering_factory, trait, target,
                max_new_tokens, batch_size,
            )
            scores = [score_response(item, response) for item, response in zip(displayed, responses)]
            valid = [float(score) for score in scores if score is not None]
            endorsement = float(np.mean(valid)) if valid else float("nan")
            score_error: float | str = ""
            if trait_score is not None and np.isfinite(endorsement):
                score_error = abs(float(trait_score(trait, target, endorsement)) - target)
            rows.append({
                "method": method,
                "trait": trait,
                "split": split,
                "target_score": target,
                "endorsement_rate": endorsement,
                "invalid_rate": 1.0 - len(valid) / len(scores) if scores else float("nan"),
                "n": len(scores),
                "valid_n": len(valid),
                "E_tau_t": score_error,
            })

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows