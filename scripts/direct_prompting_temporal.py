
"""Multi-turn direct persona prompting

Every question is asked inside one growing conversation. After each question,
the model's answer is appended to the history (as an assistant turn, following
the question as a user turn), so question k is answered with the previous
k-1 question/answer pairs as context.

Only persona-conditioned conversations are run. The target personality is
specified in the system prompt once, at the start of each conversation.

One conversation is built per (target trait, level).
Within a trait, every conversation sees the SAME item order (shuffled by seed),
so level comparisons are paired by item AND by position.

"""

import argparse
import json
import random
from pathlib import Path

from direct_prompting import DEFAULT_SEED, load_contexts, persona_messages
from rewriting.pipeline import REFUSAL, clean_statement, score_extraction
from rewriting import prompts
from rewriting.traits import LEVELS, TRAITS


def build_conversations(contexts_with_targets, seed, levels, max_turns):
    """Build one persona-conditioned conversation per (trait, level)."""
    groups = {}

    for context, trait_list in contexts_with_targets:
        for trait in trait_list:
            groups.setdefault(trait, []).append(context)

    convs = []

    for trait, contexts in groups.items():
        ordered = list(contexts)
        random.Random(seed).shuffle(ordered)

        if max_turns:
            ordered = ordered[:max_turns]

        for level in levels:
            convs.append({
                "condition": "persona",
                "target_trait": trait,
                "level": level,
                "contexts": ordered,
                "history": [],
            })

    for conv_id, conv in enumerate(convs):
        conv["conv_id"] = conv_id

    return convs


def run(generator, contexts_with_targets, *, seed, levels=LEVELS,
        temperature=0.0, max_new_tokens=128, batch_size=4,
        max_turns=None):

    convs = build_conversations(
        contexts_with_targets, seed, levels, max_turns)

    sampling = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,}

    rows = []

    for turn in range(max(len(c["contexts"]) for c in convs)):
        active = [c for c in convs if turn < len(c["contexts"])]

        for start in range(0, len(active), batch_size):
            batch = active[start:start + batch_size]

            turns = [
                persona_messages(
                    c["contexts"][turn],
                    c["target_trait"],
                    c["level"],
                )
                for c in batch]

            conversations = [[t[0]] + c["history"] + [t[1]] for c, t in zip(batch, turns)]
            outputs = generator.generate(conversations, **sampling)

            for conv, t, out in zip(batch, turns, outputs):
                text = clean_statement(out)

                conv["history"] += [t[1], {"role": "assistant", "content": text},]
                context = conv["contexts"][turn]

                rows.append(({
                    "condition": conv["condition"],
                    "target_trait": conv["target_trait"],
                    "level": conv["level"],
                    "item_id": context.item_id,
                    "item_trait": context.trait_code,
                    "kind": context.kind,
                    "seed": seed,
                    "displayed_options": list(context.displayed_options),
                    "conv_id": conv["conv_id"],
                    "turn": turn,
                    "n_turns": len(conv["contexts"]),
                    "text": text,
                    "refusal": bool(REFUSAL.search(text)),
                }, context))

        print(f"  turn {turn + 1}: {len(active)} conversations")

    # Blind extraction, independent per turn (no history, no target).
    readable = [(r, c) for r, c in rows
        if r["text"] and not r["refusal"]]

    step = batch_size * 2

    for start in range(0, len(readable), step):
        chunk = readable[start:start + step]

        answers = generator.generate(
            [prompts.extract_messages(c, r["text"]) for r, c in chunk],
            temperature=0.0,
            max_new_tokens=16,
            seed=seed,)

        for (record, _), answer in zip(chunk, answers):
            record["extracted"] = answer.strip()
    records = []

    for record, context in rows:
        record["extracted"] = record.get("extracted")
        record["score"] = (
            None if record["extracted"] is None
            else score_extraction(context, record["extracted"]))
        records.append(record)

    return sorted(records,key=lambda r: (r["conv_id"], r["turn"]))


def run_experiment(
    generator,
    out,
    *,
    source="behaviorbench",
    split="development",
    confirm_frozen=False,
    traits=tuple(TRAITS),
    levels=LEVELS,
    limit=None,
    seed=DEFAULT_SEED,
    seeds=1,
    batch_size=4,
    temperature=0.0,
    max_turns=None,
):
    """Run persona-only multi-turn experiments and write records to out."""

    if split == "test" and source != "bfi" and not confirm_frozen:
        raise ValueError(
            "split='test' requires confirm_frozen=True; "
            "tune on development only"
        )

    unknown = set(traits) - set(TRAITS)

    if unknown:
        raise ValueError(f"unknown traits: {sorted(unknown)}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with out.open("w", encoding="utf-8") as handle:
        for run_seed in range(seed, seed + seeds):

            records = run(
                generator,
                load_contexts(
                    source, split, traits, limit, run_seed
                ),
                seed=run_seed,
                levels=levels,
                temperature=temperature,
                batch_size=batch_size,
                max_turns=max_turns,
            )

            for record in records:
                record.update(source=source, split=split if source != "bfi" else None,)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            handle.flush()

            total += len(records)

            print(f"seed {run_seed}: {len(records)} records")

    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source",
        choices=["behaviorbench", "bfi"],
        default="behaviorbench",
    )

    parser.add_argument(
        "--split",
        choices=["development", "test"],
        default="development",
    )

    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="required for --split test: prompts and thresholds are final",
    )

    parser.add_argument(
        "--traits",
        nargs="+",
        choices=list(TRAITS),
        default=list(TRAITS),
    )

    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=LEVELS,
        default=list(LEVELS),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="items per trait to load, for pilots",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        help="turns per conversation (default: all items of the trait)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="first seed; runs use seed, seed+1, ...",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of runs; each reshuffles item order and choice order",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="conversations per batch; lower than single-turn because histories grow",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.split == "test" and not args.confirm_frozen:
        parser.error("--split test requires --confirm-frozen; tune on development only")

    from rewriting.llm import MODEL_ID, HFGenerator

    generator = HFGenerator.load(
        args.model or MODEL_ID,
        load_4bit=args.load_4bit,
    )

    run_experiment(
        generator,
        args.out,
        source=args.source,
        split=args.split,
        confirm_frozen=args.confirm_frozen,
        traits=args.traits,
        levels=args.levels,
        limit=args.limit,
        seed=args.seed,
        seeds=args.seeds,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
