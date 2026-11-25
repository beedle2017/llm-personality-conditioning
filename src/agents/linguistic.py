"""Linguistic style features and accommodation analysis."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

VADER = SentimentIntensityAnalyzer()

HEDGE_LEXICON = {
    "maybe",
    "perhaps",
    "i think",
    "might",
    "sort of",
    "kind of",
    "possibly",
    "seem",
    "seems",
    "seemed",
    "likely",
    "probably",
    "could be",
    "tend to",
    "tends to",
    "i guess",
    "i suppose",
    "somewhat",
    "apparent",
    "apparently",
}

FIRST_PERSON_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

CONTRACTION_PATTERN = re.compile(
    r"\b[a-zA-Z]+(?:'[a-zA-Z]+|\u2019[a-zA-Z]+)\b"
)

WORD_PATTERN = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

FEATURE_NAMES = (
    "word_count",
    "hedges_per_100",
    "contraction_rate",
    "mean_word_length",
    "pronoun_1p_rate",
    "sentiment",
    "bow_cosine",
)


def extract_words(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return [w.lower() for w in WORD_PATTERN.findall(text)]


def count_hedges(text: str) -> int:
    """Count hedge occurrences using the fixed multi-word and single-word lexicon."""
    lower_text = text.lower()
    count = 0
    # Check multi-word hedges first
    multi_word_hedges = [h for h in HEDGE_LEXICON if " " in h]
    for hedge in multi_word_hedges:
        pattern = r"\b" + re.escape(hedge) + r"\b"
        count += len(re.findall(pattern, lower_text))

    # Single word hedges
    words = extract_words(text)
    single_word_hedges = {h for h in HEDGE_LEXICON if " " not in h}
    for word in words:
        if word in single_word_hedges:
            count += 1
    return count


def cosine_similarity(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    """Compute cosine similarity between two frequency Counters."""
    all_keys = set(counter_a.keys()) | set(counter_b.keys())
    if not all_keys:
        return 0.0
    dot = sum(counter_a.get(k, 0) * counter_b.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_turn_features(text: str, partner_prev_text: str = "") -> dict[str, float]:
    """Extract all style features from an individual dialogue turn."""
    words = extract_words(text)
    n_words = len(words)

    if n_words == 0:
        return {
            "word_count": 0.0,
            "hedges_per_100": 0.0,
            "contraction_rate": 0.0,
            "mean_word_length": 0.0,
            "pronoun_1p_rate": 0.0,
            "sentiment": 0.0,
            "bow_cosine": 0.0,
        }

    # Hedges per 100 words
    n_hedges = count_hedges(text)
    hedges_per_100 = (n_hedges / n_words) * 100.0

    # Contractions per 100 words
    contractions = CONTRACTION_PATTERN.findall(text)
    contraction_rate = (len(contractions) / n_words) * 100.0

    # Mean word length
    mean_word_length = sum(len(w) for w in words) / n_words

    # 1st person pronouns per 100 words
    n_1p = sum(1 for w in words if w in FIRST_PERSON_PRONOUNS)
    pronoun_1p_rate = (n_1p / n_words) * 100.0

    # Sentiment compound score
    sentiment_score = float(VADER.polarity_scores(text)["compound"])

    # Bag-of-words cosine similarity (stopwords removed)
    content_words_self = [w for w in words if w not in STOPWORDS]
    if partner_prev_text:
        partner_words = [w for w in extract_words(partner_prev_text) if w not in STOPWORDS]
        bow_sim = cosine_similarity(Counter(content_words_self), Counter(partner_words))
    else:
        bow_sim = 0.0

    return {
        "word_count": float(n_words),
        "hedges_per_100": round(hedges_per_100, 4),
        "contraction_rate": round(contraction_rate, 4),
        "mean_word_length": round(mean_word_length, 4),
        "pronoun_1p_rate": round(pronoun_1p_rate, 4),
        "sentiment": round(sentiment_score, 4),
        "bow_cosine": round(bow_sim, 4),
    }


def compute_style_distances_for_dialogue(
    turns_a: list[dict[str, Any]],
    turns_b: list[dict[str, Any]],
    feature_stds: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute direction-free style distance change between two partners in a dialogue.

    For each feature:
        early_dist = |f_A,early - f_B,early| / sigma_f
        late_dist = |f_A,late - f_B,late| / sigma_f
        distance_reduction = early_dist - late_dist

    Positive values denote accommodation / convergence (late distance < early distance).
    Near-zero values denote no change in style distance.
    Negative values denote style divergence.
    """
    if feature_stds is None:
        feature_stds = {}

    feats_ae = [extract_turn_features(t.get("text", "")) for t in turns_a[:2]]
    feats_al = [extract_turn_features(t.get("text", "")) for t in turns_a[-2:]]
    feats_be = [extract_turn_features(t.get("text", "")) for t in turns_b[:2]]
    feats_bl = [extract_turn_features(t.get("text", "")) for t in turns_b[-2:]]

    results: dict[str, float] = {}
    for feat in FEATURE_NAMES:
        std_f = feature_stds.get(feat, 1.0)
        if std_f < 1e-6:
            std_f = 1.0

        if feat == "bow_cosine":
            w_ae = [w for t in turns_a[:2] for w in extract_words(t.get("text", "")) if w not in STOPWORDS]
            w_al = [w for t in turns_a[-2:] for w in extract_words(t.get("text", "")) if w not in STOPWORDS]
            w_be = [w for t in turns_b[:2] for w in extract_words(t.get("text", "")) if w not in STOPWORDS]
            w_bl = [w for t in turns_b[-2:] for w in extract_words(t.get("text", "")) if w not in STOPWORDS]
            sim_e = cosine_similarity(Counter(w_ae), Counter(w_be))
            sim_l = cosine_similarity(Counter(w_al), Counter(w_bl))
            # Distance reduction: dist_early - dist_late = (1 - sim_e) - (1 - sim_l) = sim_l - sim_e
            results[feat] = (sim_l - sim_e) / std_f
        else:
            ae = sum(x[feat] for x in feats_ae) / max(1, len(feats_ae)) if feats_ae else 0.0
            al = sum(x[feat] for x in feats_al) / max(1, len(feats_al)) if feats_al else 0.0
            be = sum(x[feat] for x in feats_be) / max(1, len(feats_be)) if feats_be else 0.0
            bl = sum(x[feat] for x in feats_bl) / max(1, len(feats_bl)) if feats_bl else 0.0
            de = abs(ae - be) / std_f
            dl = abs(al - bl) / std_f
            results[feat] = de - dl

    return results


def compute_shifts_for_dialogue(
    transcripts: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute linguistic shift toward partner for Agent A and Agent B in one dialogue.

    Early turns: turns 1-2 for that agent.
    Late turns: last 2 turns for that agent.
    Shift: (f_late - f_early) * sign(f_partner_early - f_self_early)
    """
    by_speaker: dict[str, list[dict[str, float]]] = {"A": [], "B": []}
    prev_partner_text = ""

    # Sort transcripts by round, then half_turn
    sorted_turns = sorted(transcripts, key=lambda t: (t.get("round", 0), t.get("half_turn", 0)))
    for t in sorted_turns:
        speaker = t["speaker"]
        feats = extract_turn_features(t["text"], prev_partner_text)
        by_speaker[speaker].append(feats)
        prev_partner_text = t["text"]

    results: dict[str, dict[str, float]] = {"A": {}, "B": {}}
    for spk, other in [("A", "B"), ("B", "A")]:
        turns_self = by_speaker[spk]
        turns_other = by_speaker[other]

        if len(turns_self) < 2:
            continue

        early_self = turns_self[:2]
        late_self = turns_self[-2:]

        for feat in FEATURE_NAMES:
            val_early_self = sum(t[feat] for t in early_self) / len(early_self)
            val_late_self = sum(t[feat] for t in late_self) / len(late_self)

            if turns_other:
                early_other = turns_other[:min(2, len(turns_other))]
                val_early_other = sum(t[feat] for t in early_other) / len(early_other)
                diff = val_early_other - val_early_self
                direction = 1.0 if diff > 1e-6 else (-1.0 if diff < -1e-6 else 0.0)
            else:
                direction = 0.0

            shift = (val_late_self - val_early_self) * direction
            results[spk][feat] = shift

    return results
