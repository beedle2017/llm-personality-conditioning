"""Resumable fast CAA evaluation using one shared model process."""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_caa import _layer_modules, _load_model  # noqa: E402
from steering.common_config import MODEL_NAME, QUANTIZATION, SEED, score_to_coeff  # noqa: E402
from steering.data import TRAITS, load_all_traits  # noqa: E402
from steering.formatting import format_pair  # noqa: E402
from steering.hooks import add_steering_hook  # noqa: E402

LAYERS = (14, 16, 18)
TARGETS = (1, 2, 3, 4, 5)
STAGES = ("extract", "layer_select", "k_calibrate", "final_eval", "generations")
BASE_BATCH_SIZE = 2
MAX_LENGTH = 384
RESULT_FIELDS = (
    "method", "trait", "layer", "k", "coeff", "target_score", "context",
    "metric", "value", "ci_low", "ci_high", "n", "seed",
)
LOGGER = logging.getLogger("run_caa_fast")


def stamp(message: str) -> None:
    LOGGER.info("%s %s", time.strftime("%H:%M:%S"), message)


def write_status(trait: str, stage: str, completed: list[str], started: float) -> None:
    done = len(completed) * len(STAGES) + (
        STAGES.index(stage) if stage in STAGES else 0
    )
    total = len(TRAITS) * len(STAGES)
    elapsed = time.time() - started
    remaining = elapsed / max(done, 1) * max(total - done, 0) / 60
    path = ROOT / "artifacts" / "steering" / "caa" / "status.json"
    path.write_text(json.dumps({
        "current_trait": trait, "stage": stage, "completed_traits": completed,
        "overall_percent": round(done / total * 100, 2),
        "minutes_remaining_estimate": round(remaining, 2),
    }, indent=2) + "\n", encoding="utf-8")


def seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def pack(tokenizer: Any, rows: list[dict[str, Any]], key: str, batch: int):
    items = [format_pair(tokenizer, row["prompt"], row[key]) for row in rows]
    length = min(MAX_LENGTH, max(item.input_ids.shape[-1] for item in items))
    ids = torch.full((len(items), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    response = torch.zeros_like(ids, dtype=torch.bool)
    for index, item in enumerate(items):
        size = min(length, item.input_ids.shape[-1])
        ids[index, :size] = item.input_ids[0, :size]
        attention[index, :size] = 1
        response[index, :size] = item.response_mask[0, :size]
    return ids, attention, response


def retry_oom(fn):
    global BASE_BATCH_SIZE
    while True:
        try:
            return fn(BASE_BATCH_SIZE)
        except torch.cuda.OutOfMemoryError:
            if BASE_BATCH_SIZE == 1:
                raise
            BASE_BATCH_SIZE = max(1, BASE_BATCH_SIZE // 2)
            torch.cuda.empty_cache()
            stamp(f"OOM; retrying with batch size {BASE_BATCH_SIZE}")


def extract(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    layers = _layer_modules(model)

    def run(batch_size: int):
        values = {layer: [] for layer in LAYERS}
        for start in tqdm(range(0, len(rows), batch_size), desc="extract", leave=False):
            batch = rows[start:start + batch_size]
            ids, attention, response = pack(tokenizer, batch, "pos", batch_size)
            with torch.inference_mode():
                output = model(
                    input_ids=ids.to(device), attention_mask=attention.to(device),
                    use_cache=False, output_hidden_states=True,
                )
            for layer in LAYERS:
                hidden = output.hidden_states[layer + 1].float().cpu()
                values[layer].append(torch.stack([
                    hidden[index][response[index]].mean(0)
                    for index in range(len(batch))
                ]))
            del output
        return {layer: torch.cat(chunks) for layer, chunks in values.items()}

    return retry_oom(run)


def margin_batch(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], layer: int,
    vector: torch.Tensor, coefficient: float,
) -> list[float]:
    device = next(model.parameters()).device
    handle = add_steering_hook(_layer_modules(model)[layer], coefficient, vector)

    def run(batch_size: int):
        result = []
        for start in range(0, len(rows), batch_size):
            source = rows[start:start + batch_size]
            expanded = []
            for row in source:
                expanded.extend((row, dict(row, pos=row["neg"])))
            ids, attention, response = pack(tokenizer, expanded, "pos", batch_size)
            with torch.inference_mode():
                logits = model(
                    input_ids=ids.to(device), attention_mask=attention.to(device),
                    use_cache=False,
                ).logits
            probs = logits[:, :-1].log_softmax(-1)
            token_ids = ids[:, 1:].to(device)
            masks = response[:, 1:].to(device)
            scores = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            means = [float(scores[i][masks[i]].mean()) for i in range(len(expanded))]
            result.extend(means[i] - means[i + 1] for i in range(0, len(means), 2))
        return result

    try:
        return retry_oom(run)
    finally:
        handle.remove()
        torch.cuda.empty_cache()


def generate(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], layer: int,
    vector: torch.Tensor, coefficient: float,
) -> list[str]:
    device = next(model.parameters()).device
    handle = add_steering_hook(_layer_modules(model)[layer], coefficient, vector)
    result = []
    try:
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            if hasattr(prompt, "input_ids"):
                prompt = prompt.input_ids
            prompt = prompt.to(device)
            with torch.inference_mode():
                output = model.generate(
                    prompt, do_sample=False, max_new_tokens=48, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            result.append(tokenizer.decode(
                output[0][prompt.shape[-1]:], skip_special_tokens=True
            ).strip())
        return result
    finally:
        handle.remove()
        torch.cuda.empty_cache()


def invalid_rate(texts: list[str]) -> float:
    bad = 0
    for text in texts:
        tokens = text.split()
        repeated = len(tokens) >= 8 and len(set(tokens[-8:])) <= 2
        if not text or len(tokens) < 2 or repeated:
            bad += 1
    return bad / max(len(texts), 1)


def bootstrap(values: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    sample = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(sample, (1000, len(sample)), replace=True), axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_trait(model: Any, tokenizer: Any, trait: str, data: Any, completed: list[str], started: float) -> None:
    root = ROOT / "artifacts" / "steering" / "caa"
    cache = root / "cache"
    generations = root / "generations"
    vectors_dir = root / "vectors"
    cache.mkdir(parents=True, exist_ok=True)
    generations.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    write_status(trait, "extract", completed, started)
    pos_path, neg_path = cache / f"{trait}_pos.pt", cache / f"{trait}_neg.pt"
    if pos_path.exists() and neg_path.exists():
        pos = torch.load(pos_path, weights_only=True)
        neg = torch.load(neg_path, weights_only=True)
        stamp(f"{trait}: loaded existing activation cache")
    else:
        pos = extract(model, tokenizer, data.train)
        neg = extract(model, tokenizer, [dict(row, pos=row["neg"]) for row in data.train])
        torch.save(pos, pos_path)
        torch.save(neg, neg_path)
    vectors = {}
    norms = {}
    for layer in LAYERS:
        delta = (pos[layer] - neg[layer]).mean(0)
        vectors[layer] = delta / delta.norm().clamp_min(1e-8)
        norms[layer] = float(torch.cat((pos[layer], neg[layer])).norm(dim=1).mean())

    write_status(trait, "layer_select", completed, started)
    subset = data.dev[: min(20, len(data.dev))]
    baseline = margin_batch(model, tokenizer, subset, 16, vectors[16], 0.0)
    layer_scores = []
    for layer in LAYERS:
        k = 0.5 * norms[layer]
        low = np.mean(margin_batch(model, tokenizer, subset, layer, vectors[layer], score_to_coeff(1, k)))
        high = np.mean(margin_batch(model, tokenizer, subset, layer, vectors[layer], score_to_coeff(5, k)))
        layer_scores.append((layer, float(high - low)))
        stamp(f"{trait}: layer {layer} slope proxy={high-low:.6f}")
    layer = max(layer_scores, key=lambda value: value[1])[0]

    write_status(trait, "k_calibrate", completed, started)
    calibration = []
    for multiplier in (0.25, 0.5, 1.0):
        texts = generate(
            model, tokenizer, subset[:5], layer, vectors[layer],
            score_to_coeff(5, multiplier * norms[layer]),
        )
        rate = invalid_rate(texts)
        calibration.append((multiplier, rate))
        stamp(f"{trait}: kx={multiplier} invalid={rate:.3f}")
    passing = [multiplier for multiplier, rate in calibration if rate <= 0.05]
    multiplier = max(passing) if passing else min(calibration, key=lambda item: item[1])[0]
    k = multiplier * norms[layer]

    write_status(trait, "final_eval", completed, started)
    margins = {}
    result_rows = []
    for target in tqdm(TARGETS, desc=f"{trait} final"):
        values = margin_batch(
            model, tokenizer, data.dev, layer, vectors[layer],
            score_to_coeff(target, k),
        )
        margins[target] = values
        low, high = bootstrap(values)
        result_rows.append({
            "method": "caa", "trait": trait, "layer": layer, "k": k,
            "coeff": score_to_coeff(target, k), "target_score": target,
            "context": "all", "metric": "pair_margin", "value": np.mean(values),
            "ci_low": low, "ci_high": high, "n": len(values), "seed": SEED,
        })
    means = [float(np.mean(margins[target])) for target in TARGETS]
    rho = float(spearmanr(TARGETS, means).statistic)
    slope = float(np.polyfit(TARGETS, means, 1)[0])
    for context in sorted({str(row.get("context", "")) for row in data.dev}):
        indices = [i for i, row in enumerate(data.dev) if str(row.get("context", "")) == context]
        context_means = [np.mean([margins[target][i] for i in indices]) for target in TARGETS]
        result_rows.append({
            "method": "caa", "trait": trait, "layer": layer, "k": k, "coeff": 0,
            "target_score": 3, "context": context, "metric": "dose_slope",
            "value": np.polyfit(TARGETS, context_means, 1)[0],
            "ci_low": "", "ci_high": "", "n": len(indices), "seed": SEED,
        })

    write_status(trait, "generations", completed, started)
    samples = {}
    generation_rows = data.dev[:10]
    for target in (1, 3, 5):
        texts = generate(
            model, tokenizer, generation_rows, layer, vectors[layer],
            score_to_coeff(target, k),
        )
        samples[str(target)] = texts
        result_rows.append({
            "method": "caa", "trait": trait, "layer": layer, "k": k,
            "coeff": score_to_coeff(target, k), "target_score": target,
            "context": "all", "metric": "invalid_rate", "value": invalid_rate(texts),
            "ci_low": "", "ci_high": "", "n": len(texts), "seed": SEED,
        })
    append(ROOT / "artifacts" / "steering" / "results.csv", result_rows)
    (generations / f"{trait}.json").write_text(json.dumps({
        "trait": trait, "layer": layer, "k": k, "samples": samples,
    }, indent=2), encoding="utf-8")
    torch.save({"vector": vectors[layer], "metadata": {
        "model": MODEL_NAME, "layer": layer, "pooling": "mean_response_tokens",
        "normalization": "unit", "n_pairs": len(data.train), "seed": SEED,
    }}, vectors_dir / f"{trait}_L{layer}.pt")
    completed.append(trait)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config.update({"model": MODEL_NAME, "quantization": QUANTIZATION, "seed": SEED, "status": "fast_complete"})
    config.setdefault("traits", {})[trait] = {
        "layer": layer, "k": k, "k_multiplier": multiplier,
        "spearman": rho, "slope": slope, "invalid_rate": invalid_rate(samples["3"]),
        "runtime_seconds": time.time() - started,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_status(trait, "complete", completed, started)
    stamp(f"RESULT trait={trait} layer={layer} k={k:.6f} spearman={rho:.4f} slope={slope:.6f} invalid={invalid_rate(samples['3']):.3f} minutes={(time.time()-started)/60:.2f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed()
    data = load_all_traits(ROOT / "data" / "pairs", seed=SEED)
    result_path = ROOT / "artifacts" / "steering" / "results.csv"
    existing = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
    completed = [trait for trait in TRAITS if f",{trait}," in existing]
    started = time.time()
    model, tokenizer = _load_model()
    tokenizer.pad_token = tokenizer.eos_token
    progress = tqdm(total=len(TRAITS) * len(STAGES), desc="CAA fast overall", unit="stage")
    try:
        for trait in TRAITS:
            if trait in completed:
                progress.update(len(STAGES))
                continue
            run_trait(model, tokenizer, trait, data[trait], completed, started)
            progress.update(len(STAGES))
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        progress.close()


if __name__ == "__main__":
    main()
