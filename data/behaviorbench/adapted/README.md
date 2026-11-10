# Adapted BehaviorBench: development/test preparation

## Result

Prepared on 2026-10-27 from the five local `*_v2.json` files. All 510 source
items are retained, with original wording, choice order, labels, and source
locations preserved. No model calls, provider setup, or changes to the original
benchmark runner are required. This is data preparation for our evaluation,
not reproduction of the authors' experiment.

| Trait / proxy | Development items | Test items | Development/test groups |
|---|---:|---:|---:|
| Openness / risk-taking | 70 | 30 | 7 / 3 |
| Conscientiousness / self-control | 70 | 30 | 7 / 3 |
| Extraversion / social-media engagement | 80 | 30 | 8 / 3 |
| Agreeableness / altruism | 70 | 30 | 7 / 3 |
| Neuroticism / heuristic choice (exploratory) | 50 | 50 | 1 / 1 |
| **Total** | **340** | **170** | **31 / 12** |

The first four traits have an approximately 70/30 group split. The complete
dataset has a 66.7/33.3 split because the heuristic family holdout is 50/50.
No smaller experimental subset has been selected; that choice remains separate
from preparation of this reusable pool.

## Files and use

- `data/development.json`: prompt selection, parser development, steering-layer
  and coefficient calibration, and rewriting calibration.
- `data/test.json`: fixed held-out evaluation pool. Do not select configurations
  or items based on these responses.
- `data/all_items.json`: complete adapted inventory, including split and review flags.
- `data/split_manifest.json`: seed, group assignments, source SHA-256 fingerprints,
  and counts. Retain with experimental outputs.
- `data/validation_report.json`: machine-readable validation results.
- `VALIDATION.md`: checks, interpretation, and known limitations.

Run from the BehaviorBench directory:

```sh
python3 adapted/prepare_data.py
python3 -m unittest discover -s adapted -p 'test_*.py' -v
```

Both commands use only Python's standard library. Default seed: `20261027`.
Regeneration is byte-stable given unchanged sources, script, and seed. Changing
the seed or grouping after examining evaluation results invalidates the original
holdout. A different output location is supported through `--output-dir`.

## Adaptation and splitting procedure

Each record includes a stable source-position ID, trait, behavioral proxy,
behavior text (where supplied), scenario, options with binary endorsement
values, scoring polarity, source indices, split group, and review status.
IDs are stable for this source version; use source fingerprints to identify
changes to upstream data. Context labels are deliberately empty and explicitly
marked unannotated rather than guessed from keywords.

For the four behavior/scenario datasets, all ten scenarios under each source
behavior stay together. Development/test therefore measures generalization to
held-out behavior manifestations within each proxy, rather than merely to new
wordings of calibrated behaviors. Self-control is stratified by direct/reverse
polarity: development has 30 direct and 40 reverse items; test has 20 direct
and 10 reverse items. Report polarity-stratified results as well as pooled ones.

Inspection of the heuristic dataset identifies two broad question templates:
records 1–50 contrast salient anecdotes with statistics (availability), and
51–100 contrast descriptive evidence with population base rates
(representativeness). The source does not explicitly name these families;
these are project annotations. All availability questions are assigned to
development and all representativeness questions to test. No family is shared.
This conservative boundary avoids calibrating on examples of the same broad
template, but does **not** give a representative Neuroticism test. Analyze it
as exploratory cross-family transfer, separately from the other traits. For a
primary Neuroticism result, obtain independently reviewed additional families
or specify and justify a finer grouping before any model-based tuning.

## Scoring contract

Use `score_response(item, response)` from `prepare_data.py` or an equivalent
implementation. It returns 1 for source-keyed proxy endorsement, 0 for the
other valid option, and `None` for invalid output. It accepts exact Yes/No,
exact choice text, or A/B for heuristic choices (case-insensitive, with optional
terminal punctuation). Explanations and mixed answers are invalid, not No.

For self-control, the source label `R` reverses Yes/No; an empty label does not.
The polarity is carried by the record, never inferred from its flattened index.
For heuristic questions, Y/N means source-keyed heuristic endorsement, not
objectively correct/incorrect reasoning. These labels are not independently
validated ground truth.

Compute endorsement as `sum(valid_scores) / number_of_valid_scores`; report
invalid counts/rates separately. With zero valid answers, return an undefined
rate, not zero. Do not convert endorsement percentages directly to BFI scores.
Compare target-level trends within the same split and report each proxy
separately. Check baseline levels per split; development and test cover different
behavior groups and need not have the same baseline rate.

For heuristic probes, counterbalance displayed option order independently of
endorsement and record the displayed key-to-original-choice mapping. Score using
that mapping; never apply the stored A/B key to shuffled options unchanged.

## Evaluation safeguards and remaining review

Steering examples must come from development groups or a separately sourced
construction set screened against test scenarios and behavior groups. No steering
construction dataset is included here. Keep development calibration data and
steering-construction data distinct where feasible; never use test items for either.

All methods must use identical test IDs. At conversation checkpoints, probe
copies of history and discard those probe branches. The tests remain public,
not secret or demonstrably absent from model training.

Structural validation is complete. Independent semantic review and context
annotation are not complete. Each adapted item carries that status. Review
ambiguity, behavior/scenario compatibility, cultural assumptions, and proxy
validity before claiming a fully validated primary evaluation. Record exclusion
reasons before looking at model outcomes. Do not move individual scenarios
between splits to repair balance; revise and version the group policy instead.

Suggested review fields: item ID, reviewer, accept/revise/exclude, context labels,
ambiguity note, polarity confirmation, and rationale. A second reviewer should
resolve disagreements. Contexts may overlap (work, family, friendship, online,
education, etc.); report counts after annotation instead of promising balanced
context coverage in advance.

Source: Pu et al., *BehaviorBench*, IJCAI 2026,
https://www.ijcai.org/proceedings/2026/658 ; repository:
https://github.com/butra1n/BehaviorBench . Retain the upstream `LICENSE` and cite
the source in project artifacts. This adaptation adds no new psychological
validation to the original proxy mappings.
