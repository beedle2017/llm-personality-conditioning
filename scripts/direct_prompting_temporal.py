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
import itertools
import json
import random
import sys
from pathlib import Path

from direct_prompting import DEFAULT_SEED, persona_messages
from rewriting.pipeline import (ADAPTED_DIR, REFUSAL, behavior_context, clean_statement, score_extraction)
from rewriting import prompts
from rewriting.traits import LEVELS, TRAITS
sys.path.insert(0, str(ADAPTED_DIR))
from get_batches import batches_ending_with_id, load_jsons


def sample_ids(traits, ids_per_trait, ids, seed):
    """Target ids: an explicit list, or `ids_per_trait` random ids per trait (fixed seed)."""
    items = load_jsons()
    by_id = {item["id"]: item for item in items}

    if ids:
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"ids not found in the helper's data file: {missing}")
        return list(ids), by_id

    rng = random.Random(seed)
    sampled = []

    for trait in traits:
        pool = [item["id"] for item in items if item["trait_code"] == trait]
        sampled += rng.sample(pool, min(ids_per_trait, len(pool)))

    return sampled, by_id


def build_conversations(target_ids, by_id, seed, levels, max_turns):
    """Build one persona-conditioned conversation per (target item, level)."""
    convs = []

    for target_id in target_ids:
        steps = list(itertools.islice(
            batches_ending_with_id(target_id, seed=seed), max_turns))

        order = steps[-1][:-1]  # history items, in the order they are asked

        for k, step in enumerate(steps):  # each list must extend the previous one
            if step[:-1] != order[:k]:
                raise ValueError("batches_ending_with_id lists no longer extend each other")

        trait = by_id[target_id]["trait_code"]
        hist = [behavior_context(item, seed) for item in order]
        target = behavior_context(by_id[target_id], seed)

        for level in levels:
            convs.append({
                "condition": "persona",
                "target_trait": trait,
                "level": level,
                "hist": hist,
                "target": target,
                "n_steps": len(steps),
                "history": [],
            })

    for conv_id, conv in enumerate(convs):
        conv["conv_id"] = conv_id

    return convs


def ask(generator, convs, contexts, batch_size, sampling):
    """Ask each conversation its own question; returns (user_message, cleaned_text) per conversation."""
    results = []

    for start in range(0, len(convs), batch_size):
        chunk = convs[start:start + batch_size]
        ctxs = contexts[start:start + batch_size]

        turns = [persona_messages(ctx, c["target_trait"], c["level"])
                 for c, ctx in zip(chunk, ctxs)]

        conversations = [[t[0]] + c["history"] + [t[1]] for c, t in zip(chunk, turns)]
        outputs = generator.generate(conversations, **sampling)

        results += [(t[1], clean_statement(out)) for t, out in zip(turns, outputs)]

    return results


def run(generator, target_ids, by_id, *, seed, levels=LEVELS,
        temperature=0.0, max_new_tokens=128, batch_size=4,
        max_turns=None, probe_turns=None):

    convs = build_conversations(target_ids, by_id, seed, levels, max_turns)

    sampling = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,}

    rows = []

    for k in range(max(c["n_steps"] for c in convs)):
        active = [c for c in convs if k < c["n_steps"]]

        if k > 0:  # add the next "other" item and the model's full answer to the history
            history_items = [c["hist"][k - 1] for c in active]

            for conv, (user_message, text) in zip(
                    active, ask(generator, active, history_items, batch_size, sampling)):
                conv["history"] += [user_message, {"role": "assistant", "content": text}]

        if probe_turns is None or (k + 1) in probe_turns:
            # probe: the target item asked last; recorded, not added to the history
            targets = [c["target"] for c in active]

            for conv, (_, text) in zip(
                    active, ask(generator, active, targets, batch_size, sampling)):
                context = conv["target"]

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
                    "turn": k,
                    "n_turns": conv["n_steps"],
                    "text": text,
                    "refusal": bool(REFUSAL.search(text)),
                }, context))

        print(f"  list length {k + 1}: {len(active)} conversations")

    # Blind extraction, independent per probe (no history, no target).
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

    return sorted(records, key=lambda r: (r["conv_id"], r["turn"]))


def run_experiment(
    generator,
    out,
    *,
    confirm_frozen=False,
    traits=tuple(TRAITS),
    levels=LEVELS,
    ids_per_trait=10,
    ids=None,
    seed=DEFAULT_SEED,
    seeds=1,
    batch_size=4,
    temperature=0.0,
    max_turns=None,
    probe_turns=None,
):
    """Run persona-only multi-turn experiments and write records to out.

    The target ids are sampled once with `seed`, so every run uses the same ids; run k then
    reshuffles the history order (and choice order) with seed + k.
    `max_turns` is the number of lists per target (list lengths 1..max_turns).
    `probe_turns` (1-based list lengths, e.g. [5, 10]) skips the target probe at other lengths.
    """

    if not confirm_frozen:
        raise ValueError(
            "this runs on the helper's test.json; "
            "pass confirm_frozen=True once prompts are final"
        )

    unknown = set(traits) - set(TRAITS)

    if unknown:
        raise ValueError(f"unknown traits: {sorted(unknown)}")

    if probe_turns:
        if max_turns is None:
            max_turns = max(probe_turns)
        if max(probe_turns) > max_turns:
            raise ValueError("probe_turns cannot exceed max_turns")

    target_ids, by_id = sample_ids(traits, ids_per_trait, ids, seed)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with out.open("w", encoding="utf-8") as handle:
        for run_seed in range(seed, seed + seeds):

            records = run(
                generator,
                target_ids,
                by_id,
                seed=run_seed,
                levels=levels,
                temperature=temperature,
                batch_size=batch_size,
                max_turns=max_turns,
                probe_turns=probe_turns,
            )

            for record in records:
                record.update(source="behaviorbench", split="test")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            handle.flush()

            total += len(records)

            print(f"seed {run_seed}: {len(records)} records")

    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="required: the helper's data file is the test split",
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
        "--ids-per-trait",
        type=int,
        default=10,
        help="target items sampled per trait (fixed seed)",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        help="explicit target item ids (overrides --ids-per-trait)",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        help="lists per target, i.e. list lengths 1..max-turns (default: every other item; very expensive)",
    )

    parser.add_argument(
        "--probe-turns",
        nargs="+",
        type=int,
        help="only probe the target at these list lengths, e.g. 5 10 (default: every length)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed for sampling ids and for the first run; runs use seed, seed+1, ...",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of runs; each reshuffles the history order and choice order",
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

    if not args.confirm_frozen:
        parser.error("--confirm-frozen is required: the helper's data file is the test split")

    from rewriting.llm import MODEL_ID, HFGenerator

    generator = HFGenerator.load(
        args.model or MODEL_ID,
        load_4bit=args.load_4bit,
    )

    run_experiment(
        generator,
        args.out,
        confirm_frozen=True,
        traits=args.traits,
        levels=args.levels,
        ids_per_trait=args.ids_per_trait,
        ids=args.ids,
        seed=args.seed,
        seeds=args.seeds,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_turns=args.max_turns,
        probe_turns=args.probe_turns,
    )


if __name__ == "__main__":
    main()