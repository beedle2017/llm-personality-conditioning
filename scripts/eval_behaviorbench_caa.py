"""Evaluate CAA steering vectors on the adapted BehaviorBench benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "behaviorbench" / "adapted"))

from eval.behaviorbench import _display_item, _generate  # noqa: E402
from prepare_data import score_response  # noqa: E402
from run_caa import _layer_modules, _load_model, build_vector, pair_margin  # noqa: E402
from steering.common_config import MODEL_NAME, SEED, score_to_coeff  # noqa: E402
from steering.data import TRAITS, load_all_traits  # noqa: E402
from steering.hooks import add_steering_hook  # noqa: E402

TARGETS = (1, 2, 3, 4, 5)
CALIBRATION_TARGETS = (1, 3, 5)
CAA_K_GRID = (1, 2, 4, 6, 8, 12, 16, 24)
BIPO_K_GRID = (0.25, 0.5, 1, 2, 4, 8)
LAYERS = {
    "openness": 16,
    "conscientiousness": 16,
    "extraversion": 16,
    "agreeableness": 14,
    "neuroticism": 14,
}
RESULT_FIELDS = (
    "method", "trait", "layer", "k", "coeff", "target_score", "context",
    "metric", "value", "ci_low", "ci_high", "n", "seed",
)
RAW_FIELDS = ("item_id", "trait", "target_score", "response", "score", "valid", "seed")
CALIBRATION_FIELDS = (
    "method", "trait", "layer", "k", "slope", "spearman",
    "invalid_rate", "n", "seed",
)
BASELINE_FIELDS = (
    "method", "trait", "split", "target_score", "endorsement_rate",
    "invalid_rate", "n", "valid_n", "E_tau_t",
)


def load_benchmark_items(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_item_seed(item_id: str, target: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{target}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def counterbalance_item(item: dict[str, Any], seed: int, target: int) -> dict[str, Any]:
    """Apply the baseline's deterministic forced-choice counterbalancing."""
    return _display_item(item, target, seed)


def bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (1000, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_evaluation_model() -> tuple[Any, Any, str]:
    try:
        model, tokenizer = _load_model()
        runtime = "4-bit NF4"
    except ImportError as error:
        if "bitsandbytes" not in str(error):
            raise
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        runtime = f"unquantized {dtype} fallback"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, runtime


def load_or_build_vectors(
    model: Any,
    tokenizer: Any,
    traits: tuple[str, ...],
    vector_dir: Path,
) -> dict[str, torch.Tensor]:
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[str, torch.Tensor] = {}
    missing = [trait for trait in traits if not (vector_dir / f"{trait}_L{LAYERS[trait]}.pt").exists()]
    training = load_all_traits(ROOT / "data" / "pairs", seed=SEED) if missing else {}
    hidden_size = int(model.config.hidden_size)
    for trait in traits:
        layer = LAYERS[trait]
        path = vector_dir / f"{trait}_L{layer}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            vector = payload["vector"] if isinstance(payload, dict) else payload
        else:
            vector = build_vector(model, tokenizer, training[trait].train, layer)
            torch.save({
                "vector": vector,
                "metadata": {
                    "model": MODEL_NAME, "layer": layer,
                    "pooling": "mean_response_tokens", "normalization": "unit",
                    "n_pairs": len(training[trait].train), "seed": SEED,
                },
            }, path)
        vector = vector.float().flatten()
        if vector.numel() != hidden_size:
            raise ValueError(f"{trait}: vector width {vector.numel()} != {hidden_size}")
        print(f"{trait}: layer={layer} vector_norm={vector.norm().item():.6f} width={vector.numel()}")
        vectors[trait] = vector
    return vectors


@contextmanager
def steering_intervention(
    model: Any,
    trait: str,
    coefficient: float,
    vectors: dict[str, torch.Tensor],
) -> Iterator[None]:
    layer = LAYERS[trait]
    # CAA vectors are unit norm; BiPO vectors may be raw, so the same k gives
    # a different residual-stream dose because the hook adds coefficient*vector.
    handle = add_steering_hook(_layer_modules(model)[layer], coefficient, vectors[trait])
    try:
        yield
    finally:
        handle.remove()


def generate_and_score(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    trait: str,
    target: int,
    k: float,
    vectors: dict[str, torch.Tensor],
    *,
    batch_size: int = 4,
    max_new_tokens: int = 8,
) -> tuple[list[str], list[float | None]]:
    displayed = [counterbalance_item(item, SEED, target) for item in items]
    with steering_intervention(model, trait, score_to_coeff(target, k), vectors):
        responses = _generate(
            model, tokenizer, displayed, None, trait, target,
            max_new_tokens, batch_size,
        )
    scores = [score_response(item, response) for item, response in zip(displayed, responses)]
    return responses, scores


def append_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _mean_and_invalid(scores: list[float | None]) -> tuple[float, float, list[float]]:
    valid = [float(score) for score in scores if score is not None]
    return (
        float(np.mean(valid)) if valid else float("nan"),
        1.0 - len(valid) / len(scores) if scores else float("nan"),
        valid,
    )


def _remove_existing_caa_rows(path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    fieldnames = tuple(rows[0].keys())
    kept = [
        row for row in rows
        if not (row.get("method") == "caa" and row.get("metric", "").startswith("behaviorbench_"))
        and not (
            row.get("method") == "caa"
            and row.get("metric") == "dose_slope"
            and row.get("context") == "test"
        )
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)


def _calibrate(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    trait: str,
    vectors: dict[str, torch.Tensor],
    grid: tuple[float, ...],
    limit: int,
) -> list[dict[str, Any]]:
    calibration_items = [item for item in items if item["trait"].lower() == trait][:limit]
    rows = []
    for k in grid:
        means: list[float] = []
        invalid_rates: list[float] = []
        for target in CALIBRATION_TARGETS:
            _, scores = generate_and_score(model, tokenizer, calibration_items, trait, target, k, vectors)
            mean, invalid, _ = _mean_and_invalid(scores)
            means.append(mean)
            invalid_rates.append(invalid)
        finite = all(np.isfinite(means))
        slope = float(np.polyfit(CALIBRATION_TARGETS, means, 1)[0]) if finite else float("-inf")
        rho = float(np.corrcoef(CALIBRATION_TARGETS, means)[0, 1]) if finite and len(set(means)) > 1 else float("nan")
        invalid = float(max(invalid_rates))
        rows.append({
            "method": "caa_k_sweep", "trait": trait, "layer": LAYERS[trait],
            "k": k, "slope": slope, "spearman": rho, "invalid_rate": invalid,
            "n": len(calibration_items), "seed": SEED,
        })
        print(f"{trait}: k={k:g} dev_slope={slope:.4f} rho={rho:.4f} invalid={invalid:.3f}")
    candidates = [(r["k"], r["slope"], r["spearman"], r["invalid_rate"]) for r in rows]
    eligible = [candidate for candidate in candidates if candidate[2] >= 0.5 and candidate[3] <= 0.05 and candidate[1] > 0]
    return rows, max(eligible or candidates, key=lambda candidate: (candidate[1], -candidate[0]))


def diagnose_pairs(
    model: Any,
    tokenizer: Any,
    vectors: dict[str, torch.Tensor],
    output: Path,
) -> None:
    """Write the BiPO-compatible pair-margin diagnostic for CAA vectors."""
    data = load_all_traits(ROOT / "data" / "pairs", seed=SEED)
    fields = (
        "method", "trait", "layer", "split", "coefficient", "coefficient_label",
        "value", "ci_low", "ci_high", "n", "seed",
    )
    rows = []
    for trait in TRAITS:
        for split, examples in (("train", data[trait].train), ("dev", data[trait].dev)):
            coefficients = [(-4, "fixed"), (-2, "fixed"), (0, "fixed"), (2, "fixed"), (4, "fixed")]
            selected_k = None
            config_path = ROOT / "artifacts" / "steering" / "caa" / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                selected_k = config.get("behaviorbench", {}).get("traits", {}).get(trait, {}).get("k")
            if selected_k is not None:
                coefficients.extend([
                    (score_to_coeff(target, float(selected_k)), f"caa_k_scaled_t{target}")
                    for target in TARGETS
                ])
            for coefficient, label in coefficients:
                values = [
                    pair_margin(model, tokenizer, row, LAYERS[trait], vectors[trait], coefficient)
                    for row in examples
                ]
                low, high = bootstrap(values, SEED)
                rows.append({
                    "method": "caa", "trait": trait, "layer": LAYERS[trait],
                    "split": split, "coefficient": coefficient,
                    "coefficient_label": label, "value": float(np.mean(values)),
                    "ci_low": low, "ci_high": high, "n": len(values), "seed": SEED,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    _seed_everything()
    traits = tuple(args.traits) if args.traits else TRAITS
    grid = tuple(args.k_grid) if args.k_grid else CAA_K_GRID
    development = load_benchmark_items(args.development)
    test = load_benchmark_items(args.test)
    model, tokenizer, runtime = load_evaluation_model()
    vectors = load_or_build_vectors(model, tokenizer, traits, args.vector_dir)
    if args.force:
        _remove_existing_caa_rows(args.results)
    existing = set()
    if args.results.exists():
        with args.results.open(newline="", encoding="utf-8") as handle:
            existing = {
                row["trait"] for row in csv.DictReader(handle)
                if row.get("method") == "caa" and row.get("metric") == "dose_slope"
            }
    calibration_path = args.output_dir / "behaviorbench_k_calibration.csv"
    raw_path = args.output_dir / "behaviorbench_items.csv"
    baseline_path = args.output_dir / "behaviorbench_baseline.csv"
    if args.force or not calibration_path.exists():
        calibration_path.unlink(missing_ok=True)
    if args.force or not raw_path.exists():
        raw_path.unlink(missing_ok=True)
    if args.force or not baseline_path.exists():
        baseline_path.unlink(missing_ok=True)
    summary = []
    for trait in traits:
        if trait in existing and not args.force:
            print(f"{trait}: already present; skipping (use --force to rerun)")
            continue
        if trait == "neuroticism":
            print("neuroticism: exploratory heuristic-choice proxy")
        dev = [item for item in development if item["trait"].lower() == trait]
        test_items = [item for item in test if item["trait"].lower() == trait]
        if args.dev_limit is not None:
            dev = dev[:args.dev_limit]
        if args.test_limit is not None:
            test_items = test_items[:args.test_limit]
        calibration, selected = _calibrate(model, tokenizer, dev, trait, vectors, grid, len(dev))
        append_rows(calibration_path, CALIBRATION_FIELDS, calibration)
        selected_k = float(selected[0])
        result_rows: list[dict[str, Any]] = []
        means = []
        invalids = []
        counts = []
        valid_counts = []
        for target in (args.targets or TARGETS):
            print(f"{trait}: target={target} k={selected_k:g} n={len(test_items)}")
            responses, scores = generate_and_score(model, tokenizer, test_items, trait, target, selected_k, vectors)
            mean, invalid, valid = _mean_and_invalid(scores)
            low, high = bootstrap(valid, SEED)
            means.append(mean)
            invalids.append(invalid)
            counts.append(len(scores))
            valid_counts.append(len(valid))
            result_rows.extend([
                {"method": "caa", "trait": trait, "layer": LAYERS[trait], "k": selected_k,
                 "coeff": score_to_coeff(target, selected_k), "target_score": target,
                 "context": "test", "metric": "behaviorbench_endorsement", "value": mean,
                 "ci_low": low, "ci_high": high, "n": len(valid), "seed": SEED},
                {"method": "caa", "trait": trait, "layer": LAYERS[trait], "k": selected_k,
                 "coeff": score_to_coeff(target, selected_k), "target_score": target,
                 "context": "test", "metric": "behaviorbench_invalid_rate", "value": invalid,
                 "ci_low": "", "ci_high": "", "n": len(test_items), "seed": SEED},
            ])
            displayed = [counterbalance_item(item, SEED, target) for item in test_items]
            append_rows(raw_path, RAW_FIELDS, [
                {"item_id": item["id"], "trait": trait, "target_score": target,
                 "response": response, "score": score, "valid": score is not None, "seed": SEED}
                for item, response, score in zip(displayed, responses, scores)
            ])
        slope = float(np.polyfit(args.targets or TARGETS, means, 1)[0]) if len(means) > 1 else float("nan")
        result_rows.append({
            "method": "caa", "trait": trait, "layer": LAYERS[trait], "k": selected_k,
            "coeff": 0, "target_score": 3, "context": "test", "metric": "dose_slope",
            "value": slope, "ci_low": "", "ci_high": "", "n": len(test_items), "seed": SEED,
        })
        append_rows(args.results, RESULT_FIELDS, result_rows)
        append_rows(
            baseline_path,
            BASELINE_FIELDS,
            [
                {
                    "method": "caa",
                    "trait": trait.title(),
                    "split": "test",
                    "target_score": target,
                    "endorsement_rate": mean,
                    "invalid_rate": invalid,
                    "n": count,
                    "valid_n": valid_count,
                    "E_tau_t": "",
                }
                for target, mean, invalid, count, valid_count in zip(
                    args.targets or TARGETS,
                    means,
                    invalids,
                    counts,
                    valid_counts,
                )
            ],
        )
        summary.append((trait, selected_k, means, slope, max(invalids, default=float("nan"))))
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    config["behaviorbench"] = {
        "runtime": runtime, "quantization": runtime, "k_grid": list(grid),
        "decoding": {"do_sample": False, "max_new_tokens": 8, "batch_size": 4},
        "prompt_version": "eval.behaviorbench baseline prompt",
        "traits": {trait: {"layer": LAYERS[trait], "k": k} for trait, k, *_ in summary},
        "seed": SEED,
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if args.pair_diagnostics:
        diagnose_pairs(
            model, tokenizer, vectors,
            args.output_dir / "pair_margin_diagnostics.csv",
        )
    print("trait\tlayer\tk\tE(t=1..5)\tdose_slope\tinvalid_rate")
    for trait, k, means, slope, invalid in summary:
        print(f"{trait}\t{LAYERS[trait]}\t{k:g}\t{means}\t{slope:.4f}\t{invalid:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits", nargs="+", choices=TRAITS)
    parser.add_argument("--k-grid", nargs="+", type=float, default=None)
    parser.add_argument("--targets", nargs="+", type=int, choices=TARGETS)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pair-diagnostics", action="store_true")
    parser.add_argument("--development", type=Path, default=ROOT / "data/behaviorbench/adapted/data/development.json")
    parser.add_argument("--test", type=Path, default=ROOT / "data/behaviorbench/adapted/data/test.json")
    parser.add_argument("--vector-dir", type=Path, default=ROOT / "artifacts/steering/caa/vectors")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/steering/caa")
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts/steering/results.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "artifacts/steering/caa/config.json")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
