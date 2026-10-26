"""Validation, deduplication, benchmark separation, judging, and splitting."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from typing import Any, Iterable

BANNED = {
    "openness": ("openness", "open-minded", "curious person"),
    "conscientiousness": ("conscientious", "dutiful", "organized person"),
    "extraversion": ("extraversion", "extravert", "extrovert", "introvert"),
    "agreeableness": ("agreeableness", "agreeable person"),
    "neuroticism": ("neuroticism", "neurotic"),
}
REFUSALS = ("i can't", "i cannot", "i’m unable", "i am unable", "as an ai")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def jaccard_ngrams(left: str, right: str, n: int = 3) -> float:
    a, b = words(left.lower()), words(right.lower())
    left_set = set(zip(*(a[i:] for i in range(n)))) if len(a) >= n else set()
    right_set = set(zip(*(b[i:] for i in range(n)))) if len(b) >= n else set()
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def validate_pair(pair: dict[str, Any], trait: str) -> str | None:
    required = ("prompt", "pos", "neg", "facet", "context", "length_band")
    if not all(isinstance(pair.get(key), str) and pair[key].strip() for key in required):
        return "malformed"
    pos, neg = words(pair["pos"]), words(pair["neg"])
    if len(pos) < 4 or len(neg) < 4:
        return "too_short"
    text = normalize(pair["pos"] + " " + pair["neg"])
    if any(label in text for label in BANNED.get(trait, ())):
        return "trait_label"
    if any(marker in text for marker in REFUSALS):
        return "refusal"
    ratio = len(pos) / len(neg)
    if not 0.7 <= ratio <= 1.3:
        return "length_ratio"
    return None


def filter_pairs(pairs: Iterable[dict[str, Any]], trait: str, benchmark_prompts: Iterable[str] = (), similarity_threshold: float = 0.8) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted, drops, seen = [], Counter(), []
    benchmarks = list(benchmark_prompts)
    for pair in pairs:
        reason = validate_pair(pair, trait)
        if reason:
            drops[reason] += 1
            continue
        prompt = normalize(pair["prompt"])
        if prompt in seen or any(jaccard_ngrams(prompt, old) >= similarity_threshold for old in seen):
            drops["duplicate"] += 1
            continue
        if any(jaccard_ngrams(prompt, normalize(old)) >= similarity_threshold for old in benchmarks):
            drops["behaviorbench_overlap"] += 1
            continue
        seen.append(prompt)
        accepted.append(pair)
    return accepted, drops


def assign_split(pairs: list[dict[str, Any]], n_dev: int, seed: int) -> list[dict[str, Any]]:
    """Assign whole prompts deterministically, never splitting duplicate prompts."""
    unique = {normalize(pair["prompt"]) for pair in pairs}
    rng = random.Random(seed)
    dev_prompts = set(rng.sample(sorted(unique), min(n_dev, len(unique))))
    return [dict(pair, split="dev" if normalize(pair["prompt"]) in dev_prompts else "train") for pair in pairs]


def judge_pairs(pairs: list[dict[str, Any]], trait: str, client: Any, model: str, batch_size: int = 10, seed: int = 0) -> tuple[list[dict[str, Any]], float]:
    rng = random.Random(seed)
    kept, agreements = [], 0
    for start in range(0, len(pairs), batch_size):
        batch, payload = pairs[start:start + batch_size], []
        for pair in batch:
            swapped = rng.choice((False, True))
            payload.append({"a": pair["neg" if swapped else "pos"], "b": pair["pos" if swapped else "neg"]})
        prompt = (
            f'For trait "{trait}", choose which response expresses the high pole more strongly. '
            "Both responses answer the same prompt and should be judged behaviorally. "
            "Return JSON only as an array with one object per item, each containing "
            '"choice" ("A" or "B") and "valid" (true only when both responses are helpful, '
            "non-refusal, and comparable). Items:\n" + json.dumps(payload)
        )
        raw = client.complete(prompt, model)
        from .llm_client import parse_json_response
        decisions = parse_json_response(raw)
        for index, (pair, item) in enumerate(zip(batch, decisions)):
            if not item.get("valid", False):
                continue
            expected = "B" if pair["pos"] == payload[index]["b"] else "A"
            if item.get("choice") == expected:
                agreements += 1
                kept.append(pair)
    return kept, agreements / max(1, len(pairs))


def stable_id(pair: dict[str, Any]) -> str:
    return hashlib.sha256(normalize(pair["prompt"]).encode()).hexdigest()[:16]
