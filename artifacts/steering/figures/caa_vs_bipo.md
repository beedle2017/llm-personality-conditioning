# CAA vs BiPO

| trait | method | E(1) | E(5) | gap | default E(1) | default E(5) | dose slope | invalid rate | k | layer | t5-t1 CIs separate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| openness | caa | 0.867 | 1.000 | 0.133 | 0.933 | 0.933 | 0.040 | 0.0 | 4.0 | 16 | yes |
| openness | bipo | 0.711 | 0.889 | 0.178 | 0.933 | 0.933 | 0.048 | 0.0 | 4.0 | 15 | no |
| conscientiousness | caa | 0.700 | 0.667 | -0.033 | 0.767 | 0.767 | -0.010 | 0.0 | 4.0 | 16 | no |
| conscientiousness | bipo | 0.700 | 0.711 | 0.011 | 0.767 | 0.767 | 0.001 | 0.0 | 4.0 | 15 | no |
| extraversion | caa | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | -0.000 | 0.0 | 1.0 | 16 | no |
| extraversion | bipo | 0.950 | 0.983 | 0.033 | 1.000 | 1.000 | 0.008 | 0.0 | 4.0 | 15 | no |
| agreeableness | caa | 0.867 | 0.933 | 0.067 | 0.900 | 0.900 | 0.020 | 0.0 | 1.0 | 14 | no |
| agreeableness | bipo | 0.400 | 1.000 | 0.600 | 0.900 | 0.900 | 0.153 | 0.0 | 4.0 | 15 | yes |
| neuroticism | caa | 0.580 | 0.520 | -0.060 | 0.460 | 0.540 | -0.010 | 0.0 | 4.0 | 14 | no |
| neuroticism | bipo | 0.490 | 0.540 | 0.050 | 0.460 | 0.540 | 0.014 | 0.0 | 4.0 | 15 | no |

## Caveats

- CAA uses per-trait layers 14/16; BiPO uses fixed layer 15.
- CAA uses 4-bit NF4; BiPO used an unquantized fp16 fallback.
- CAA vectors are unit norm; BiPO vectors may be raw, so k scales are not directly equal.
- CAA uses k grid (1, 2, 4, 6, 8, 12, 16, 24); BiPO uses its notebook grid.
- Neuroticism is an exploratory 50/50 heuristic-choice proxy.
- Bootstrap-CI separation is descriptive and is not a winner claim.
