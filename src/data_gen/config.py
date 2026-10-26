"""Configuration and controlled dimensions for synthetic personality data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

TRAITS = {
    "openness": ("curiosity", "imagination", "preference for novelty", "intellectual interest", "aesthetic appreciation"),
    "conscientiousness": ("orderliness", "self-discipline", "dutifulness", "achievement striving"),
    "extraversion": ("sociability", "assertiveness", "energy", "enthusiasm"),
    "agreeableness": ("trust", "cooperation", "empathy", "modesty"),
    "neuroticism": ("anxiety", "irritability", "vulnerability", "mood volatility"),
}
CONTEXTS = ("work", "friendship", "family", "online interaction", "planning", "decision-making", "problem-solving", "giving advice", "casual conversation")
LENGTH_BANDS = {"short": (1, 2), "medium": (3, 5)}


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "gemini"
    generator_model: str = ""
    judge_model: str = ""
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env: str = "GEMINI_API_KEY"
    requests_per_minute: int = 10
    daily_request_quota: int = 1500
    max_retries: int = 4
    batch_pairs_per_call: int = 5
    n_workers: int = 2
    seed: int = 20261026
    prompt_similarity_threshold: float = 0.8
    behaviorbench_similarity_threshold: float = 0.8


def load_config(path: str | Path) -> GenerationConfig:
    """Load YAML values, retaining safe defaults for omitted settings."""
    values: dict[str, Any] = {}
    config_path = Path(path)
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a mapping: {config_path}")
        values = loaded
    fields = {field.name for field in GenerationConfig.__dataclass_fields__.values()}
    config = GenerationConfig(**{key: value for key, value in values.items() if key in fields})
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", config.api_key_env):
        raise ValueError(
            "api_key_env must be an environment-variable name such as GEMINI_API_KEY; "
            "do not put an API key value in configs/data_gen.yaml"
        )
    return config
