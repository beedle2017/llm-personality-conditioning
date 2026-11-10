# Validation report — 2026-10-27

## Completed automated checks

- Loaded all five JSON files and validated required fields and nonempty text.
- Confirmed 510 unique item IDs: 410 behavior scenarios and 100 forced choices.
- Preserved source wording, labels, and indices; recorded source SHA-256 hashes.
- Validated heuristic choices are distinct and contain exactly one Y and one N.
- Validated all self-control source polarity labels are either empty or `R`.
- Found zero normalized exact scenario duplicate clusters.
- Confirmed disjoint item partitions covering all 510 items exactly once.
- Confirmed all 43 groups remain entirely within one split (41 source behaviors
  and two annotated heuristic families).
- Screened same-trait cross-split scenarios using normalized character similarity
  at a 0.90 threshold: zero candidate pairs. This is a lexical screen, not a
  semantic similarity guarantee.
- Verified scoring for every option, reverse scoring, invalid outputs, and
  deliberate group-leakage detection.
- Verified repeated builds produce identical data and manifests.
- Five automated tests pass.

## Findings requiring interpretation

1. **Heuristic answer position is confounded with family.** All 50 availability
   questions mark original choice 1 as Y; all 50 representativeness questions
   mark original choice 2 as Y. Display order must be counterbalanced in the
   eventual evaluation harness and mapped back to source labels.
2. **Heuristic family split is exploratory.** With only two broad templates, a
   group-disjoint split cannot also represent both families in both partitions.
   The 50/50 family holdout deliberately prioritizes template separation and
   does not support a broad Neuroticism robustness claim.
3. **Some source questions have semantic problems.** Heuristic source records
   62 and 74 explicitly reveal an attribute subsequently queried (electric
   vehicle ownership and scientific-journal reading). Record 68 keys Finance
   as heuristic endorsement despite a mindfulness description whose intended
   relation to the alternatives is unclear. These are flagged, not silently
   corrected or excluded. They illustrate why source labels must not be called
   objective reasoning ground truth. This is not an exhaustive item-quality audit.
4. **Proxy breadth is limited.** Risk-taking is not equivalent to Openness,
   social-media engagement is not equivalent to Extraversion, and heuristic
   endorsement is not equivalent to Neuroticism. Risk-taking includes health
   and financial choices that may reflect safety preferences or knowledge.
5. **Self-control polarity is preserved, not exactly balanced.** Development
   has 30 direct / 40 reverse items; test has 20 direct / 10 reverse. Report
   separate polarity results and account for this when comparing partitions.
6. **Context coverage remains unverified.** Context labels are empty/pending.
   No claims of balanced work/family/social coverage are justified yet.

## Status and exclusions

No items excluded; all source information retained. Structurally validated and
reproducibly split, with independent semantic review pending. No model outputs
were used to choose the seed, groups, or split. No model experiments were run.

The preparation milestone is complete for schema, scoring, provenance, and
partitioning. Full content/construct validation is not established by these
checks; the review flags must remain visible in downstream reporting.

## Suggested checkpoint wording

We standardized all 510 BehaviorBench items and established fixed development
(340 items) and test (170 items) partitions, holding out complete behavioral
groups and preserving self-control polarity. Automated checks verified source
schema, scoring, duplicate screening, reproducibility, and group separation.
The heuristic subset is treated as exploratory because its two question
families confound answer position and cannot both be represented in a
family-disjoint development/test split. Semantic review and context annotation
remain separate quality-control steps before primary experimental claims.
