# BehaviorBench source data and project adaptation

Source: https://github.com/butra1n/BehaviorBench

Paper: Pu et al., *BehaviorBench: A Psychologically Grounded Benchmark for
Evaluating Personality in Large Language Models Through Realistic Behaviors*,
IJCAI 2026. https://www.ijcai.org/proceedings/2026/658

The five `*_v2.json` files are unchanged copies from the local BehaviorBench
checkout. SHA-256 fingerprints are recorded in
`adapted/data/split_manifest.json`. The upstream Apache-2.0 license is retained
in this directory; these source materials are not relicensed under the parent
project's MIT license.

The `adapted/` directory contains the project's standardized records, fixed
development/test partitions, preparation script, tests, and validation notes.
Those records add IDs, provenance, explicit scoring, grouping, and review flags;
the original scenario and option wording is retained. Preparation dates from
2026-10-27; copied into this project on 2026-10-30.

See [the adaptation guide](adapted/README.md) for limitations and usage. No
original experiment scripts, profiles, credentials, or cache files are included.
