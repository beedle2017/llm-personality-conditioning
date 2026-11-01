"""Build and evaluate CAA steering vectors using the shared protocol."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.common_config import (  # noqa: E402
    DECODING,
    DTYPE,
    MODEL_NAME,
    QUANTIZATION,
    SEED,
    score_to_coeff,
    shared_config,
)
from steering.data import TRAITS, load_all_traits  # noqa: E402
from steering.formatting import format_pair  # noqa: E402
from steering.hooks import add_steering_hook  # noqa: E402

LOGGER = logging.getLogger("run_caa")
TARGETS = (1, 2, 3, 4, 5)
LAYERS = tuple(range(8, 25))
K_GRID = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
RESULT_FIELDS = (
    "method", "trait", "layer", "k", "coeff", "target_score", "context",
    "metric", "value", "ci_low", "ci_high", "n", "seed",
)


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _load_model() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _layer_modules(model: Any) -> list[Any]:
    candidates = getattr(getattr(model, "model", None), "layers", None)
    if candidates is None:
        raise RuntimeError("Could not locate decoder layers at model.model.layers")
    return list(candidates)


def _activation(model: Any, tokenizer: Any, row: dict[str, Any], layer: int) -> torch.Tensor:
    formatted = format_pair(tokenizer, row["prompt"], row["pos"])
    device = next(model.parameters()).device
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())
        return output

    handle = _layer_modules(model)[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(
                input_ids=formatted.input_ids.to(device),
                attention_mask=formatted.attention_mask.to(device),
                use_cache=False,
            )
    finally:
        handle.remove()
    mask = formatted.response_mask.to(captured[0].device)
    return captured[0][mask].float().mean(dim=0).cpu()


def build_vector(model: Any, tokenizer: Any, rows: list[dict[str, Any]], layer: int) -> torch.Tensor:
    differences = []
    for row in rows:
        pos = _activation(model, tokenizer, row, layer)
        neg_row = dict(row, pos=row["neg"])
        neg = _activation(model, tokenizer, neg_row, layer)
        differences.append(pos - neg)
    vector = torch.stack(differences).mean(dim=0)
    return vector / vector.norm().clamp_min(1e-8)


def _token_logprob(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor) -> float:
    log_probs = logits[:, :-1].log_softmax(dim=-1)
    token_ids = input_ids[:, 1:]
    token_mask = mask[:, 1:]
    values = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)[token_mask]
    return float(values.mean().item()) if values.numel() else float("nan")


def pair_margin(
    model: Any, tokenizer: Any, row: dict[str, Any], layer: int, vector: torch.Tensor, alpha: float
) -> float:
    device = next(model.parameters()).device
    formatted_pos = format_pair(tokenizer, row["prompt"], row["pos"])
    formatted_neg = format_pair(tokenizer, row["prompt"], row["neg"])
    module = _layer_modules(model)[layer]
    handle = add_steering_hook(module, alpha, vector)
    try:
        with torch.inference_mode():
            pos_logits = model(
                input_ids=formatted_pos.input_ids.to(device),
                attention_mask=formatted_pos.attention_mask.to(device),
                use_cache=False,
            ).logits
            neg_logits = model(
                input_ids=formatted_neg.input_ids.to(device),
                attention_mask=formatted_neg.attention_mask.to(device),
                use_cache=False,
            ).logits
    finally:
        handle.remove()
    return _token_logprob(pos_logits, formatted_pos.input_ids.to(device), formatted_pos.response_mask.to(device)) - _token_logprob(
        neg_logits, formatted_neg.input_ids.to(device), formatted_neg.response_mask.to(device)
    )


def _bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(1000)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_not_run(reason: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = output_dir.parent / "results.csv"
    if not results.exists():
        results.write_text(",".join(RESULT_FIELDS) + "\n", encoding="utf-8")
    config = shared_config()
    config.update({"status": "not run", "reason": reason, "method": "caa"})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "tuning_log.csv").write_text(
        "iteration,trait,change,status,reason\n0,all,initialization,not run,"
        + reason.replace(",", ";")
        + "\n",
        encoding="utf-8",
    )


def run(output_dir: Path) -> None:
    _seed_everything()
    data = load_all_traits(ROOT / "data" / "pairs", seed=SEED)
    for trait, trait_data in data.items():
        print(
            f"polarity check {trait}: "
            f"pos={trait_data.train[0]['pos'][:100]!r}; "
            f"neg={trait_data.train[0]['neg'][:100]!r}"
        )
    try:
        model, tokenizer = _load_model()
    except Exception as exc:
        reason = f"Model unavailable: {type(exc).__name__}: {exc}"
        LOGGER.error(reason)
        _write_not_run(reason, output_dir)
        return

    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    for trait in TRAITS:
        train, dev = data[trait].train, data[trait].dev
        layer = 16
        vector = build_vector(model, tokenizer, train, layer)
        path = vectors_dir / f"{trait}_L{layer}.pt"
        torch.save(
            {"vector": vector, "metadata": {"model": MODEL_NAME, "layer": layer,
             "pooling": "mean_response_tokens", "normalization": "unit",
             "n_pairs": len(train), "seed": SEED}},
            path,
        )
        k = 1.0
        for target in TARGETS:
            margins = [pair_margin(model, tokenizer, row, layer, vector, score_to_coeff(target, k)) for row in dev]
            low, high = _bootstrap(margins, SEED)
            result_rows.append({
                "method": "caa", "trait": trait, "layer": layer, "k": k,
                "coeff": score_to_coeff(target, k), "target_score": target,
                "context": "all", "metric": "pair_margin", "value": np.mean(margins),
                "ci_low": low, "ci_high": high, "n": len(margins), "seed": SEED,
            })
        selected[trait] = {"layer": layer, "k": k, "n_train": len(train), "n_dev": len(dev)}
    _write_rows(output_dir.parent / "results.csv", result_rows)
    (output_dir / "config.json").write_text(json.dumps({**shared_config(), "status": "complete", "traits": selected}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "steering" / "caa")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.output_dir)


if __name__ == "__main__":
    main()
