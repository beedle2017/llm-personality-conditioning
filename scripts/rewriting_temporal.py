"""Multi-turn post-generation rewriting, matching scripts/direct_prompting_temporal.py.

Same targets (`sample_ids`), the same history order (`batches_ending_with_id`) and the same
probe lengths as the direct prompting run, so the two methods can be compared item by item.

There is no persona system prompt. At every turn the base model drafts an answer with the
conversation so far, and the rewriter (which sees the same history) rewrites it for the
target. The full rewritten answer is appended to the history. At a probe length the target
item is asked on a copy of the conversation and the rewrite is recorded, not appended.
Extraction is blind (scenario and statement only), as in the single-turn run.

Pilot:  python3 scripts/rewriting_temporal.py --ids-per-trait 2 --levels 1 5 \
            --max-turns 10 --probe-turns 5 10 --out outputs/rewriting/temporal_pilot.jsonl

In a notebook:
    from rewriting_temporal import run_experiment
    run_experiment(generator, f"{OUT}/temporal.jsonl", max_turns=10, probe_turns=[5, 10], batch_size=16)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from direct_prompting import DEFAULT_SEED  # noqa: E402
from direct_prompting_temporal import build_conversations, sample_ids  # noqa: E402
from rewriting import prompts  # noqa: E402
from rewriting.pipeline import REFUSAL, clean_statement, preservation, score_extraction  # noqa: E402
from rewriting.traits import LEVELS, TRAITS  # noqa: E402

MODE = "free"


def with_history(messages, history):
    """Insert earlier turns between the system message and the current request."""
    return [messages[0], *history, *messages[1:]]


def draft_conversation(conv, context):
    return with_history(prompts.draft_messages(context), conv["history"])


def rewrite_conversation(conv, context, draft):
    messages = prompts.rewrite_messages(context, draft, conv["target_trait"], conv["level"], MODE)
    return with_history(messages, conv["history"])


def _generate(generator, conversations, batch_size, sampling):
    outputs = []
    for start in range(0, len(conversations), batch_size):
        outputs += generator.generate(conversations[start:start + batch_size], **sampling)
    return outputs


def answer(generator, convs, contexts, batch_size, sampling):
    """Draft then rewrite each conversation's current question.

    Returns (user_message, draft, rewrite) per conversation; the user message is the plain
    draft request (scenario + task), which is what enters the history.
    """
    drafts = [clean_statement(text) for text in _generate(
        generator, [draft_conversation(c, x) for c, x in zip(convs, contexts)], batch_size, sampling)]
    rewrites = [clean_statement(text) for text in _generate(
        generator, [rewrite_conversation(c, x, d) for c, x, d in zip(convs, contexts, drafts)],
        batch_size, sampling)]
    users = [prompts.draft_messages(context)[1] for context in contexts]
    return list(zip(users, drafts, rewrites))


def run(generator, target_ids, by_id, *, seed, levels=LEVELS, temperature=0.0, max_new_tokens=128,
        batch_size=4, max_turns=None, probe_turns=None):
    convs = build_conversations(target_ids, by_id, seed, levels, max_turns)
    for conv in convs:
        conv["condition"] = MODE
    sampling = {"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed}
    rows = []

    for k in range(max(c["n_steps"] for c in convs)):
        active = [c for c in convs if k < c["n_steps"]]

        if k > 0:  # the next history item, answered and appended in full
            items = [c["hist"][k - 1] for c in active]
            for conv, (user, draft, rewrite) in zip(active, answer(generator, active, items, batch_size, sampling)):
                # An empty rewrite would leave a blank turn, so the history keeps the draft instead.
                conv["history"] += [user, {"role": "assistant", "content": rewrite or draft}]

        if probe_turns is None or (k + 1) in probe_turns:
            targets = [c["target"] for c in active]
            for conv, (_, draft, rewrite) in zip(active, answer(generator, active, targets, batch_size, sampling)):
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
                    "draft": draft,
                    "text": rewrite,
                    "refusal": bool(REFUSAL.search(rewrite)),
                    "preservation": preservation(draft, rewrite),
                }, context))

        print(f"  list length {k + 1}: {len(active)} conversations")

    # Blind extraction, independent per probe (no history, no target).
    readable = [(r, c) for r, c in rows if r["text"] and not r["refusal"]]
    extract_messages = [prompts.extract_messages(c, r["text"]) for r, c in readable]
    answers = _generate(generator, extract_messages, batch_size * 2,
                        {"temperature": 0.0, "max_new_tokens": 16, "seed": seed})
    for (record, _), extracted in zip(readable, answers):
        record["extracted"] = extracted.strip()

    records = []
    for record, context in rows:
        record["extracted"] = record.get("extracted")
        record["score"] = None if record["extracted"] is None else score_extraction(context, record["extracted"])
        records.append(record)
    return sorted(records, key=lambda r: (r["conv_id"], r["turn"]))


def run_experiment(generator, out, *, traits=tuple(TRAITS), levels=LEVELS, ids_per_trait=10, ids=None,
                   seed=DEFAULT_SEED, seeds=1, batch_size=4, temperature=0.0, max_turns=None, probe_turns=None):
    """Same arguments and target sampling as direct_prompting_temporal.run_experiment."""
    unknown = set(traits) - set(TRAITS)
    if unknown:
        raise ValueError(f"unknown traits: {sorted(unknown)}")
    if probe_turns:
        max_turns = max_turns or max(probe_turns)
        if max(probe_turns) > max_turns:
            raise ValueError("probe_turns cannot exceed max_turns")

    target_ids, by_id = sample_ids(traits, ids_per_trait, ids, seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as handle:
        for run_seed in range(seed, seed + seeds):
            records = run(generator, target_ids, by_id, seed=run_seed, levels=levels, temperature=temperature,
                          batch_size=batch_size, max_turns=max_turns, probe_turns=probe_turns)
            for record in records:
                record.update(source="behaviorbench", split="test", method="rewriting")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            total += len(records)
            print(f"seed {run_seed}: {len(records)} records")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traits", nargs="+", choices=list(TRAITS), default=list(TRAITS))
    parser.add_argument("--levels", nargs="+", type=int, choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--ids-per-trait", type=int, default=10)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--probe-turns", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from rewriting.llm import MODEL_ID, HFGenerator
    generator = HFGenerator.load(args.model or MODEL_ID, load_4bit=args.load_4bit)
    run_experiment(generator, args.out, traits=args.traits, levels=args.levels, ids_per_trait=args.ids_per_trait,
                   ids=args.ids, seed=args.seed, seeds=args.seeds, batch_size=args.batch_size,
                   temperature=args.temperature, max_turns=args.max_turns, probe_turns=args.probe_turns)


if __name__ == "__main__":
    main()
