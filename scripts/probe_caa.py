"""Print an end-to-end CAA proof on the extraversion split."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_caa import _layer_modules, _load_model  # noqa: E402
from steering.common_config import SEED, score_to_coeff  # noqa: E402
from steering.data import load_trait_data  # noqa: E402
from steering.formatting import format_pair  # noqa: E402
from steering.hooks import add_steering_hook  # noqa: E402


def _batches(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _capture(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    response_key: str,
    batch_size: int = 8,
    layer_indices: tuple[int, ...] = (16,),
) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    captures: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
    handles = []
    for index in layer_indices:
        def capture(_module: Any, _inputs: Any, output: Any, index=index) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            captures[index].append(hidden.detach().float().cpu())
            return output
        handles.append(_layer_modules(model)[index].register_forward_hook(capture))
    try:
      for batch in _batches(rows, batch_size):
        formatted = [format_pair(tokenizer, row["prompt"], row[response_key]) for row in batch]
        max_len = max(item.input_ids.shape[-1] for item in formatted)
        input_ids = torch.full(
            (len(formatted), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        masks = torch.zeros_like(input_ids, dtype=torch.bool)
        attention = torch.zeros_like(input_ids)
        for index, item in enumerate(formatted):
            length = item.input_ids.shape[-1]
            input_ids[index, :length] = item.input_ids[0]
            masks[index, :length] = item.response_mask[0]
            attention[index, :length] = 1
        with torch.inference_mode():
            model(input_ids=input_ids.to(device), attention_mask=attention.to(device), use_cache=False)
        for index in captures:
            hidden = captures[index][-1]
            captures[index][-1] = torch.stack(
                [hidden[row_index][masks[row_index]].mean(0)
                 for row_index in range(len(batch))]
            )
    finally:
        for handle in handles:
            handle.remove()
    return {index: torch.cat(values) for index, values in captures.items()}


def _forward_logits(model: Any, tokenizer: Any, rows: list[dict[str, Any]], layer: int, vector: torch.Tensor, alpha: float) -> list[float]:
    device = next(model.parameters()).device
    module = _layer_modules(model)[layer]
    handle = add_steering_hook(module, alpha, vector)
    values = []
    try:
        for row in rows:
            formatted = format_pair(tokenizer, row["prompt"], row["pos"])
            with torch.inference_mode():
                logits = model(
                    input_ids=formatted.input_ids.to(device),
                    attention_mask=formatted.attention_mask.to(device),
                    use_cache=False,
                ).logits
            log_probs = logits[:, :-1].log_softmax(-1)
            ids = formatted.input_ids[:, 1:].to(device)
            mask = formatted.response_mask[:, 1:].to(device)
            values.append(float(log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)[mask].mean()))
    finally:
        handle.remove()
    return values


def _margin(model: Any, tokenizer: Any, row: dict[str, Any], layer: int, vector: torch.Tensor, alpha: float) -> float:
    pos = _forward_logits(model, tokenizer, [row], layer, vector, alpha)[0]
    neg_row = dict(row, pos=row["neg"])
    neg = _forward_logits(model, tokenizer, [neg_row], layer, vector, alpha)[0]
    return pos - neg


def _margins(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    batch_size: int = 8,
) -> list[float]:
    device = next(model.parameters()).device
    handle = add_steering_hook(_layer_modules(model)[layer], alpha, vector)
    values: list[float] = []
    try:
        for batch in _batches(rows, batch_size):
            formatted = []
            for row in batch:
                formatted.extend((
                    format_pair(tokenizer, row["prompt"], row["pos"]),
                    format_pair(tokenizer, row["prompt"], row["neg"]),
                ))
            max_len = max(item.input_ids.shape[-1] for item in formatted)
            input_ids = torch.full((len(formatted), max_len), tokenizer.pad_token_id, dtype=torch.long)
            masks = torch.zeros_like(input_ids, dtype=torch.bool)
            attention = torch.zeros_like(input_ids)
            for index, item in enumerate(formatted):
                length = item.input_ids.shape[-1]
                input_ids[index, :length] = item.input_ids[0]
                masks[index, :length] = item.response_mask[0]
                attention[index, :length] = 1
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    use_cache=False,
                ).logits
            log_probs = logits[:, :-1].log_softmax(-1)
            token_ids = input_ids[:, 1:].to(device)
            token_masks = masks[:, 1:].to(device)
            token_scores = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            means = [
                float(token_scores[index][token_masks[index]].mean())
                for index in range(len(formatted))
            ]
            values.extend(means[0::2][index] - means[1::2][index] for index in range(len(batch)))
    finally:
        handle.remove()
    return values


def main() -> None:
    torch.manual_seed(SEED)
    data = load_trait_data("extraversion", ROOT / "data" / "pairs", seed=SEED)
    model, tokenizer = _load_model()
    tokenizer.pad_token = tokenizer.eos_token
    print("PROBE trait=extraversion train=%d dev=%d" % (len(data.train), len(data.dev)))

    pos = _capture(model, tokenizer, data.train, "pos")
    neg = _capture(model, tokenizer, data.train, "neg")
    difference = (pos[16] - neg[16]).mean(0)
    vector = difference / difference.norm().clamp_min(1e-8)
    print(
        "LAYER %d vector_norm=%.6f mean_activation_norm=%.6f "
        "pos_neg_delta=%.6f finite=%s"
        % (16, difference.norm(), torch.cat((pos[16], neg[16])).norm(dim=1).mean(),
           difference.norm(), bool(torch.isfinite(vector).all()))
    )
    labels = np.r_[np.ones(len(data.train)), np.zeros(len(data.train))]
    small_pos = _capture(model, tokenizer, data.train[:8], "pos", layer_indices=tuple(range(8, 25)))
    small_neg = _capture(model, tokenizer, data.train[:8], "neg", layer_indices=tuple(range(8, 25)))
    for layer in range(8, 25):
        small_difference = (small_pos[layer] - small_neg[layer]).mean(0)
        small_vector = small_difference / small_difference.norm().clamp_min(1e-8)
        projections = torch.cat((small_pos[layer], small_neg[layer])).matmul(small_vector).numpy()
        print("LAYER %d vector_norm=%.6f mean_activation_norm=%.6f finite=%s" % (
            layer, small_difference.norm(), torch.cat((small_pos[layer], small_neg[layer])).norm(dim=1).mean(),
            bool(torch.isfinite(small_vector).all())))
        print("SEPARATION layer=%d auc=%.4f accuracy=%.4f" % (
            layer,
            roc_auc_score(np.r_[np.ones(8), np.zeros(8)], projections),
            np.mean((projections > np.median(projections)) == np.r_[np.ones(8), np.zeros(8)]),
        ))

    layer = 16
    formatted = format_pair(tokenizer, data.dev[0]["prompt"], data.dev[0]["pos"])
    device = next(model.parameters()).device
    with torch.inference_mode():
        baseline = model(input_ids=formatted.input_ids.to(device), attention_mask=formatted.attention_mask.to(device), use_cache=False).logits
    handle = add_steering_hook(_layer_modules(model)[layer], 0.0, vector)
    try:
        with torch.inference_mode():
            hooked = model(input_ids=formatted.input_ids.to(device), attention_mask=formatted.attention_mask.to(device), use_cache=False).logits
    finally:
        handle.remove()
    print("ZERO_COEFF max_abs_diff=%.9f pass=%s" % ((baseline - hooked).abs().max(), bool((baseline - hooked).abs().max() < 1e-3)))

    mean_norm = float(torch.cat((pos[layer], neg[layer])).norm(dim=1).mean())
    for multiplier in (1, 2, 4):
        k = multiplier * mean_norm
        margins = []
        for target in range(1, 6):
            values = _margins(model, tokenizer, data.dev, layer, vector, score_to_coeff(target, k))
            margins.append(float(np.mean(values)))
        print("DOSE k=%.4f margins=%s spearman=%.4f" % (k, [round(x, 5) for x in margins], spearmanr(range(1, 6), margins).statistic))

    prompt = data.dev[0]["prompt"]
    for target in (1, 3, 5):
        coefficient = score_to_coeff( target, mean_norm)
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        handle = add_steering_hook(_layer_modules(model)[layer], coefficient, vector)
        try:
            with torch.inference_mode():
                generated = model.generate(formatted_prompt, do_sample=False, max_new_tokens=64)
        finally:
            handle.remove()
        print("GENERATION t=%d: %s" % (target, tokenizer.decode(generated[0][formatted_prompt.shape[-1]:], skip_special_tokens=True).strip()))


if __name__ == "__main__":
    main()
