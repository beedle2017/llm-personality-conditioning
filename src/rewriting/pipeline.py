"""Single-turn rewriting pipeline: draft -> rewrite -> blind extraction -> score.

Conditions, each extracted and scored the same way:
    draft      unconditioned text; its extracted decision is the baseline
    free       rewriter may change the decision
    locked     rewriter may change tone only
    paraphrase no target; measures flips caused by rewording alone
    no_draft   free-mode request without a draft
"""
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import prompts
from .traits import LEVELS

ROOT = Path(__file__).resolve().parents[2]
ADAPTED_DIR = ROOT / "data" / "behaviorbench" / "adapted"
sys.path.insert(0, str(ADAPTED_DIR))
from prepare_data import score_response  # noqa: E402

TARGET_CONDITIONS = ("free", "locked", "no_draft")
CONDITIONS = ("draft", "paraphrase") + TARGET_CONDITIONS
BFI_STEM = "I see myself as someone who {item}."

# Narrow on purpose: "I can't help you move" is a valid low-Agreeableness answer.
REFUSAL = re.compile(r"\bas an ai\b|\blanguage model\b|\bI (cannot|can't|won't) "
                     r"(fulfill|rewrite|comply|create content)\b", re.IGNORECASE)
PREAMBLE = re.compile(r"^\s*(here(?:'s| is)\b[^\n]*:|rewritten statement:|statement:)\s*", re.IGNORECASE)
WORD = re.compile(r"[a-z']+")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class Generator(Protocol):
    def generate(self, conversations, *, temperature, max_new_tokens, seed): ...


@dataclass(frozen=True)
class ProbeContext:
    """One question as displayed, plus how to score the extracted answer."""
    item_id: str
    trait_code: str
    kind: str  # yes_no, forced_choice, or bfi
    prompt: str
    item: dict | None = None
    displayed_options: tuple = ()
    reverse: bool = False


def behavior_context(item, seed):
    """Build the displayed question; forced-choice order is shuffled per item and seed."""
    if item["response_type"] == "forced_choice":
        options = [option["text"] for option in item["options"]]
        random.Random(f"{seed}:{item['id']}").shuffle(options)
        prompt = f"{item['scenario']} {options[0]} or {options[1]}?"
        return ProbeContext(item["id"], item["trait_code"], "forced_choice", prompt, item, tuple(options))
    prompt = f"Situation: {item['scenario']}\n\nWould you do the following: '{item['behavior']}'?"
    return ProbeContext(item["id"], item["trait_code"], "yes_no", prompt, item)


def bfi_context(index, text, trait_code, reverse):
    item = text[0].lower() + text[1:]
    prompt = f"Consider this description: {BFI_STEM.format(item=item)}"
    return ProbeContext(f"bfi-{index:02d}", trait_code, "bfi", prompt, reverse=reverse)


def clean_statement(text):
    """Remove wrappers models add around a statement; never edit the content."""
    text = PREAMBLE.sub("", text.strip()).strip()
    if len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


def score_extraction(context, answer):
    """Return the score for an extracted answer, or None if it is invalid."""
    if context.kind == "bfi":
        match = re.fullmatch(r"\s*([1-5])\W*", answer)
        if not match:
            return None
        rating = int(match.group(1))
        return 6 - rating if context.reverse else rating
    if context.kind == "forced_choice":
        # Stored A/B keys do not describe a shuffled display, so accept text only.
        if answer.strip().rstrip(".!?").strip().casefold() in {"a", "b"}:
            return None
    return score_response(context.item, answer)


def preservation(draft, rewrite):
    """Cheap draft-to-rewrite checks; a high overlap means the draft was really used."""
    left, right = set(WORD.findall(draft.lower())), set(WORD.findall(rewrite.lower()))
    numbers = set(NUMBER.findall(draft))
    return {"length_ratio": round(len(rewrite.split()) / max(1, len(draft.split())), 3),
            "word_jaccard": round(len(left & right) / max(1, len(left | right)), 3),
            "numbers_kept": round(len(numbers & set(NUMBER.findall(rewrite))) / len(numbers), 3) if numbers else None}


def plan_records(context, seed, target_traits, levels=LEVELS, conditions=CONDITIONS):
    """List the runs for one question; each target trait gets its own level sweep."""
    records = []
    for condition in conditions:
        if condition in TARGET_CONDITIONS:
            records += [{"condition": condition, "target_trait": trait, "level": level}
                        for trait in target_traits for level in levels]
        else:
            records.append({"condition": condition, "target_trait": None, "level": None})
    for record in records:
        record.update(item_id=context.item_id, item_trait=context.trait_code,
                      kind=context.kind, seed=seed, displayed_options=list(context.displayed_options))
    return records


def _batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _generate(generator, jobs, build, batch_size, **kwargs):
    """Run `build(job)` conversations in batches and return outputs in job order."""
    outputs = []
    for batch in _batches(jobs, batch_size):
        outputs += generator.generate([build(job) for job in batch], **kwargs)
    return outputs


def run(generator, contexts_with_targets, *, seed, levels=LEVELS, conditions=CONDITIONS,
        temperature=0.0, max_new_tokens=128, batch_size=8):
    """Run all conditions for `(context, target_traits)` pairs and return records.

    Drafts do not depend on the target, so each is made once and shared by every
    level, which makes comparisons between levels paired by item.
    """
    sampling = {"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed}
    contexts = [context for context, _ in contexts_with_targets]
    drafts = [clean_statement(text) for text in
              _generate(generator, contexts, prompts.draft_messages, batch_size, **sampling)]

    jobs = []
    for (context, traits), draft in zip(contexts_with_targets, drafts):
        jobs += [(context, draft, record)
                 for record in plan_records(context, seed, traits, levels, conditions)]

    def build(job):
        context, draft, record = job
        if record["condition"] == "paraphrase":
            return prompts.paraphrase_messages(context, draft)
        if record["condition"] == "no_draft":
            return prompts.no_draft_messages(context, record["target_trait"], record["level"])
        return prompts.rewrite_messages(context, draft, record["target_trait"],
                                        record["level"], record["condition"])

    pending = [job for job in jobs if job[2]["condition"] != "draft"]
    for job, text in zip(pending, _generate(generator, pending, build, batch_size, **sampling)):
        job[2]["text"] = clean_statement(text)
    for context, draft, record in jobs:
        record["draft"] = draft
        if record["condition"] == "draft":
            record["text"] = draft
        else:
            record["preservation"] = preservation(draft, record["text"])
        record["refusal"] = bool(REFUSAL.search(record["text"]))

    readable = [job for job in jobs if job[2]["text"] and not job[2]["refusal"]]
    answers = _generate(generator, readable, lambda job: prompts.extract_messages(job[0], job[2]["text"]),
                        batch_size, temperature=0.0, max_new_tokens=16, seed=seed)
    for job, answer in zip(readable, answers):
        job[2]["extracted"] = answer.strip()
    records = []
    for context, _, record in jobs:
        record["extracted"] = record.get("extracted")
        record["score"] = None if record["extracted"] is None else score_extraction(context, record["extracted"])
        records.append(record)
    return records
