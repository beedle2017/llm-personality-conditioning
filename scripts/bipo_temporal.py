"""Multi-turn BIPO steering, matching scripts/direct_prompting_temporal.py and scripts/rewriting_temporal.py.

Same targets (`sample_ids`), the same history order (`batches_ending_with_id`) and the same
probe lengths as the direct prompting and rewriting runs, so the three methods can be compared
item by item.

There is no persona system prompt and no rewriter. Each question gets the unconditioned draft
prompt (`prompts.draft_messages`, the same prompt as the rewriting draft) after the conversation so far.
The trait's BIPO vector is added at layer 15 with coefficient `score_to_coeff(level, k)`
for every generated turn of the conversation. The vectors, layer and k = 4 are the ones used
in the single-turn run in notebooks/evaluate_bipo_behaviorbench.ipynb.
The full steered answer is appended to the history. At a probe length the target item is asked
on a copy of the conversation and the answer is recorded, not appended. Extraction is blind
(scenario and statement only) and unsteered, as in the other two methods.

Steering adds one coefficient to every row of a batch, so batches are grouped by (trait, level).

Pilot:  python3 scripts/bipo_temporal.py --ids-per-trait 2 --levels 1 5 \
            --max-turns 10 --probe-turns 5 10 --out outputs/bipo/temporal_pilot.jsonl

In a notebook:
    from bipo_temporal import run_experiment
    run_experiment(generator, f"{OUT}/temporal.jsonl", max_turns=10, probe_turns=[5, 10], batch_size=16)
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from direct_prompting import DEFAULT_SEED  # noqa: E402
from direct_prompting_temporal import build_conversations, sample_ids  # noqa: E402
from rewriting import prompts  # noqa: E402
from rewriting.pipeline import REFUSAL, clean_statement, score_extraction  # noqa: E402
from rewriting.traits import LEVELS, TRAITS  # noqa: E402
from steering.common_config import score_to_coeff  # noqa: E402
from steering.hooks import add_steering_hook  # noqa: E402

MODE = "bipo"
LAYER = 15
K = 4.0
VECTOR_DIR = ROOT / "artifacts" / "steering_vectors" / "bipo"
TRAIT_NAMES = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
               "A": "agreeableness", "N": "neuroticism"}


def load_vectors(vector_dir=VECTOR_DIR, traits=tuple(TRAITS)):
    """One 1-D float vector per trait code, as in evaluate_bipo_behaviorbench.load_bipo_vectors."""
    import torch
    vectors = {}
    for trait in traits:
        vector = torch.load(Path(vector_dir) / f"{TRAIT_NAMES[trait]}.pt", map_location="cpu",
                            weights_only=True).float()
        if vector.ndim != 1:
            raise ValueError(f"{trait}: expected a one-dimensional vector, got {tuple(vector.shape)}")
        vectors[trait] = vector
    return vectors


@contextmanager
def steer(generator, vector, coeff, layer):
    """Add coeff * vector to the residual stream after decoder layer `layer` (the shared hook)."""
    layers = getattr(getattr(generator.model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers at model.model.layers")
    handle = add_steering_hook(layers[layer], coeff, vector)
    try:
        yield
    finally:
        handle.remove()


def with_history(messages, history):
    """Insert earlier turns between the system message and the current request."""
    return [messages[0], *history, *messages[1:]]


def _generate(generator, conversations, batch_size, sampling):
    outputs = []
    for start in range(0, len(conversations), batch_size):
        outputs += generator.generate(conversations[start:start + batch_size], **sampling)
    return outputs


def answer(generator, convs, contexts, batch_size, sampling, vectors, *, k, layer, steer_fn):
    """Answer each conversation's current question with its own steering.

    Returns (user_message, text) per conversation, in the order of `convs`.
    """
    groups = {}
    for index, conv in enumerate(convs):
        groups.setdefault((conv["target_trait"], conv["level"]), []).append(index)

    texts = [None] * len(convs)
    for (trait, level), indices in groups.items():
        conversations = [with_history(prompts.draft_messages(contexts[i]), convs[i]["history"])
                         for i in indices]
        with steer_fn(generator, vectors[trait], score_to_coeff(level, k), layer):
            outputs = _generate(generator, conversations, batch_size, sampling)
        for i, text in zip(indices, outputs):
            texts[i] = clean_statement(text)

    users = [prompts.draft_messages(context)[1] for context in contexts]
    return list(zip(users, texts))


def run(generator, target_ids, by_id, *, seed, vectors, levels=LEVELS, k=K, layer=LAYER,
        temperature=0.0, max_new_tokens=128, batch_size=4, max_turns=None, probe_turns=None,
        steer_fn=steer):
    convs = build_conversations(target_ids, by_id, seed, levels, max_turns)
    for conv in convs:
        conv["condition"] = MODE
    sampling = {"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed}
    options = {"k": k, "layer": layer, "steer_fn": steer_fn}
    rows = []

    for step in range(max(c["n_steps"] for c in convs)):
        active = [c for c in convs if step < c["n_steps"]]

        if step > 0:  # the next history item, answered with steering and appended in full
            items = [c["hist"][step - 1] for c in active]
            for conv, (user, text) in zip(active, answer(generator, active, items, batch_size, sampling,
                                                         vectors, **options)):
                conv["history"] += [user, {"role": "assistant", "content": text}]

        if probe_turns is None or (step + 1) in probe_turns:
            targets = [c["target"] for c in active]
            for conv, (_, text) in zip(active, answer(generator, active, targets, batch_size, sampling,
                                                      vectors, **options)):
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
                    "turn": step,
                    "n_turns": conv["n_steps"],
                    "layer": layer,
                    "k": k,
                    "coeff": score_to_coeff(conv["level"], k),
                    "text": text,
                    "refusal": bool(REFUSAL.search(text)),
                }, context))

        print(f"  list length {step + 1}: {len(active)} conversations")

    # Blind, unsteered extraction, independent per probe (no history, no target).
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
                   seed=DEFAULT_SEED, seeds=1, batch_size=4, temperature=0.0, max_turns=None, probe_turns=None,
                   k=K, layer=LAYER, vectors=None, steer_fn=steer):
    """Same arguments and target sampling as direct_prompting_temporal.run_experiment.

    `vectors` maps trait code to a BIPO vector; by default they are loaded from VECTOR_DIR.
    """
    unknown = set(traits) - set(TRAITS)
    if unknown:
        raise ValueError(f"unknown traits: {sorted(unknown)}")
    if probe_turns:
        max_turns = max_turns or max(probe_turns)
        if max(probe_turns) > max_turns:
            raise ValueError("probe_turns cannot exceed max_turns")

    vectors = vectors if vectors is not None else load_vectors(traits=traits)
    target_ids, by_id = sample_ids(traits, ids_per_trait, ids, seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as handle:
        for run_seed in range(seed, seed + seeds):
            records = run(generator, target_ids, by_id, seed=run_seed, vectors=vectors, levels=levels, k=k,
                          layer=layer, temperature=temperature, batch_size=batch_size, max_turns=max_turns,
                          probe_turns=probe_turns, steer_fn=steer_fn)
            for record in records:
                record.update(source="behaviorbench", split="test", method="bipo")
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
    parser.add_argument("--k", type=float, default=K)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from rewriting.llm import MODEL_ID, HFGenerator
    generator = HFGenerator.load(args.model or MODEL_ID, load_4bit=args.load_4bit)
    run_experiment(generator, args.out, traits=args.traits, levels=args.levels, ids_per_trait=args.ids_per_trait,
                   ids=args.ids, seed=args.seed, seeds=args.seeds, batch_size=args.batch_size,
                   temperature=args.temperature, max_turns=args.max_turns, probe_turns=args.probe_turns,
                   k=args.k, layer=args.layer)


if __name__ == "__main__":
    main()
