# Multi-turn (temporal) test for post-generation rewriting: plan

Mirrors PR #15 (`direct-time`, `scripts/direct_prompting_temporal.py`) so prompting and
rewriting can be compared item by item and turn by turn. No extra ablations: rewriting
gets the same test as the other methods.

## 1. What to copy exactly from the prompting PR

| Setting | Prompting PR #15 | Rewriting |
|---|---|---|
| Targets | `sample_ids`: 10 per trait, all 5 traits, seed 20261026 (50 items) | same function, same ids |
| History order | `batches_ending_with_id(target, seed=20261026)` | same, so the histories are identical |
| Levels | 1–5 | 1–5 |
| Length | `max_turns=10`, probes at list lengths 5 and 10 | same |
| Probe | asked on a copy, never added to the history | same |
| Extraction | blind: scenario and answer only, after the run | same |
| Batch size | 16 | 16 |
| Records | `conv_id`, `turn`, `n_turns`, `level`, `target_trait`, `text`, `extracted`, `score` | same fields, plus the `draft` text |

To guarantee the same targets and orders, import `sample_ids` and the conversation
construction from `direct_prompting_temporal.py` instead of copying them. Until PR #15
is merged, branch from `origin/direct-time`.

## 2. How a rewriting conversation runs

There is no persona system prompt. The persona enters through the rewriter, at every turn.

For each of the 250 conversations (one per target and level), every turn (history and probe) is:
1. **Draft**: `SPEAKER` system prompt, then the history so far, then the scenario plus `DRAFT_TASK`.
2. **Rewrite (free)**: the same history, then the usual `rewrite_messages` request
   (scenario, draft, target, free rule).

In history turns the full rewritten answer is appended to the history. At the probes
(lengths 5 and 10) it is recorded and extracted instead. The draft text is saved in the
record for reading, but it is not scored.

## 3. Cost

- History: 250 conversations × 9 turns × 2 calls = 4,500.
- Probes: 250 × 2 lengths × (draft + rewrite + extraction) = 1,500.
- Total: about 6,000 short generations (at most 10 turns). Expect under an hour on the L4.
- Pilot: `ids_per_trait=2, levels=[1, 5]`, as in the PR's pilot cell.

## 4. Code

- `src/rewriting/temporal.py`: advance all conversations one turn per step (batched); draft
  and rewrite messages with history; probes on a copy; blind extraction at the end. Reuse
  `prompts`, `behavior_context`, `clean_statement`, `score_extraction` and `REFUSAL`.
- `scripts/run_rewriting_temporal.py`: `run_experiment(...)`, with the same signature as the PR.
- Tests with the fake model: the history order equals `batches_ending_with_id`; probes never
  enter the history; the extractor never sees the history or the target.
- Notebook: a "Temporal" section in `post_generation_rewriting.ipynb` (pilot, then full run),
  printing the same level × trait tables at 5 and 10 turns as the prompting notebook.

## 5. Analysis

- The 25-cell level × trait matrix at 5 and 10 turns, as agreed.
- Spearman ρ(level, score) per trait, as in the prompting notebook.
- Compare with the single-turn results already in `artifacts/rewriting/results.csv`.
- One caveat for the write-up: the rewriter restates the target at every turn, so rewriting
  may hold up over a conversation by design. Say so when comparing it with prompting.

## 6. Points to raise on PR #15 (not blocking)

- **Seed for the history order.** Arka's sketch calls `batches_ending_with_id(id)` with the
  default seed 42, but PR #15 passes `seed=20261026`. The two give different histories. Agree
  on one for all methods; this plan follows PR #15, since that is what has been run.
- The docstring says every conversation in a trait sees the same order. `batches_ending_with_id`
  shuffles the other items per target, so only the levels of one target share an order.
  The pairing by item and position across levels still holds.
- With 10 targets per trait, each cell steps in 0.1, so differences under about 0.3 are noise.
  Consider `seeds=2` or 20 targets per trait for the final run (all methods together).
