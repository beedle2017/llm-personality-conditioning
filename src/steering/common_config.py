"""Configuration shared by CAA and BiPO steering experiments."""

from __future__ import annotations

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DTYPE = "bfloat16"
QUANTIZATION = "4-bit NF4 (bitsandbytes)"
SEED = 20261026
DECODING = {"do_sample": False, "max_new_tokens": 128}


def score_to_coeff(target_score: float, k: float) -> float:
    """Map a target score in [1, 5] to a signed steering coefficient."""
    if not 1 <= target_score <= 5:
        raise ValueError(f"target_score must be in [1, 5], got {target_score}")
    return k * (target_score - 3) / 2


def shared_config() -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "quantization": QUANTIZATION,
        "seed": SEED,
        "decoding": DECODING,
    }
