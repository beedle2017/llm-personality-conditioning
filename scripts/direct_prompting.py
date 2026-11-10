"""Run direct persona prompting on BehaviorBench or the BFI and write JSONL records.

Conditions:
    draft    unconditioned answer (baseline; identical to the rewriting pipeline's draft)
    persona  answer generated with the target personality in the system prompt

Pilot: python3 scripts/direct_prompting.py --limit 10 --out outputs/persona/pilot.jsonl

In a notebook, load the model once and call `run_experiment` for each run:
    generator = HFGenerator.load(load_4bit=True)
    run_experiment(generator, "outputs/persona/pilot.jsonl", limit=10, levels=[1, 5])
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rewriting import prompts  # noqa: E402
from rewriting.pipeline import (ADAPTED_DIR, REFUSAL, _generate, behavior_context,  bfi_context, clean_statement, score_extraction) # noqa: E402
from rewriting.traits import LEVELS, TRAITS  # noqa: E402

DEFAULT_SEED = 20261026
CONDITIONS = ("draft", "persona")

# Tunable on the development split only
PERSONA_SYSTEM = (
    "Your personality: {target}\n"
    "Let it shape both what you decide to do and how you express it. "
    "Never mention the score or the scale."
)


def persona_messages(context, trait_code, level):
    """The baseline draft conversation with the target personality in the system message.

    The user turn is exactly the baseline draft's (question + DRAFT_TASK / BFI_DRAFT_TASK), and
    the system turn is prompts.SPEAKER followed by the persona. The trait text is (describe_target + "other traits neutral").
    """
    persona = PERSONA_SYSTEM.format(target=prompts._target_block(trait_code, level))
    messages = [dict(message) for message in prompts.draft_messages(context)]
    messages[0]["content"] = messages[0]["content"] + "\n\n" + persona
    return messages


def load_contexts(source, split, traits, limit, seed):
    if source == "bfi":
        from bfi import BFI44
        return [(bfi_context(index, text, trait, reverse), list(traits))
                for index, (text, trait, reverse) in enumerate(BFI44, 1)]
    items = json.loads((ADAPTED_DIR / "data" / f"{split}.json").read_text(encoding="utf-8"))
    selected = []
    for trait in traits:
        subset = [item for item in items if item["trait_code"] == trait]
        if limit and limit < len(subset):
            subset = [subset[i * len(subset) // limit] for i in range(limit)]
        selected += [(behavior_context(item, seed), [trait]) for item in subset]
    return selected


def plan_records(context, seed, target_traits, levels, conditions):
    """List the runs for one question"""
    records = []
    if "draft" in conditions:
        records.append({"condition": "draft", "target_trait": None, "level": None})
    if "persona" in conditions:
        records += [{"condition": "persona", "target_trait": trait, "level": level}
                    for trait in target_traits for level in levels]
    for record in records:
        record.update(item_id=context.item_id, item_trait=context.trait_code,
                      kind=context.kind, seed=seed, displayed_options=list(context.displayed_options))
    return records


def run(generator, contexts_with_targets, *, seed, levels=LEVELS, conditions=CONDITIONS,
        temperature=0.0, max_new_tokens=128, batch_size=8):
    """Generate with and without the persona prompt, extract blindly, score, return records."""
    sampling = {"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed}
    jobs = []
    for context, traits in contexts_with_targets:
        jobs += [(context, record) for record in plan_records(context, seed, traits, levels, conditions)]

    def build(job):
        context, record = job
        if record["condition"] == "draft":
            return prompts.draft_messages(context)
        return persona_messages(context, record["target_trait"], record["level"])

    for job, text in zip(jobs, _generate(generator, jobs, build, batch_size, **sampling)):
        job[1]["text"] = clean_statement(text)

    baseline = {job[0].item_id: job[1]["text"] for job in jobs if job[1]["condition"] == "draft"}
    for context, record in jobs:
        record["draft"] = baseline.get(context.item_id)
        record["refusal"] = bool(REFUSAL.search(record["text"]))

    readable = [job for job in jobs if job[1]["text"] and not job[1]["refusal"]]
    answers = _generate(generator, readable, lambda job: prompts.extract_messages(job[0], job[1]["text"]),
                        batch_size, temperature=0.0, max_new_tokens=16, seed=seed)
    for job, answer in zip(readable, answers):
        job[1]["extracted"] = answer.strip()

    records = []
    for context, record in jobs:
        record["extracted"] = record.get("extracted")
        record["score"] = None if record["extracted"] is None else score_extraction(context, record["extracted"])
        records.append(record)
    return records


def run_experiment(generator, out, *, source="behaviorbench", split="development",
                   confirm_frozen=False, traits=tuple(TRAITS), levels=LEVELS,
                   conditions=CONDITIONS, limit=None, seed=DEFAULT_SEED, seeds=1,
                   batch_size=8, temperature=0.0):
    """Run every seed with an already-loaded generator and write records to `out`."""
    if split == "test" and source != "bfi" and not confirm_frozen:
        raise ValueError("split='test' requires confirm_frozen=True; tune on development only")
    if "draft" not in conditions:
        raise ValueError("the draft condition is the baseline and cannot be removed")
    unknown = set(traits) - set(TRAITS)
    if unknown:
        raise ValueError(f"unknown traits: {sorted(unknown)}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as handle:
        for run_seed in range(seed, seed + seeds):
            records = run(generator, load_contexts(source, split, traits, limit, run_seed),
                          seed=run_seed, levels=levels, conditions=conditions,
                          temperature=temperature, batch_size=batch_size)
            for record in records:
                record.update(source=source, split=split if source != "bfi" else None)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            total += len(records)
            print(f"seed {run_seed}: {len(records)} records")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["behaviorbench", "bfi"], default="behaviorbench")
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--confirm-frozen", action="store_true",
                        help="required for --split test: prompts and thresholds are final")
    parser.add_argument("--traits", nargs="+", choices=list(TRAITS), default=list(TRAITS))
    parser.add_argument("--levels", nargs="+", type=int, choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, help="items per trait, for pilots")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="first seed; runs use seed, seed+1, ...")
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of runs; with greedy decoding extra runs only reshuffle choice order")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy, as in the shared protocol")
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-4bit", action="store_true", help="4-bit NF4 as in the steering runs; CUDA only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_frozen:
        parser.error("--split test requires --confirm-frozen; tune on development only")
    if "draft" not in args.conditions:
        parser.error("the draft condition is the baseline and cannot be removed")

    from rewriting.llm import MODEL_ID, HFGenerator
    generator = HFGenerator.load(args.model or MODEL_ID, load_4bit=args.load_4bit)
    run_experiment(generator, args.out, source=args.source, split=args.split,
                   confirm_frozen=args.confirm_frozen, traits=args.traits, levels=args.levels,
                   conditions=args.conditions, limit=args.limit, seed=args.seed, seeds=args.seeds,
                   batch_size=args.batch_size, temperature=args.temperature)


if __name__ == "__main__":
    main()