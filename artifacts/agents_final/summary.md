# Multi-Agent Social Robustness Summary

## 2x2 Accommodation Classification and Behavioral Shift

| Method | Trait | Pair | Task Type | r_A(0) | r_B(0) | Manip Check | Headroom | Net Shift [95% CI] | Ling Shift [95% CI] | BFI Err | PIS | Classification | Invalid Rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompting | C | 1-5 | collaborative | 0.2 | 1.0 | PASS | PASS | 0.018 [-0.225, 0.300] | -0.061 [-0.142, 0.020] | 0.500 | +0.375 | **neither** | 0.0% |
| prompting | A | 1-5 | collaborative | 0.6 | 0.8 | WARN (uninformative) | PASS | 0.001 [-0.100, 0.100] | 0.573 [0.436, 0.692] | 0.900 | +0.275 | **linguistic-only** | 0.0% |
| prompting | E | 1-5 | collaborative | 0.2 | 1.0 | PASS | PASS | 0.000 [0.000, 0.000] | 0.239 [-0.046, 0.591] | 0.400 | +0.400 | **neither** | 0.0% |
| prompting | C | 3-3 | collaborative | 0.8 | 0.8 | N/A | PASS | 0.047 [-0.176, 0.250] | -0.080 [-0.143, -0.027] | 0.800 | N/A | **neither** | 0.0% |
| steering | A | 1-5 | collaborative | 0.6 | 0.8 | WARN (uninformative) | WARN (ceiling-saturated) | 0.000 [ceiling-saturated] | 0.597 [0.345, 0.849] | 2.000 | +0.000 | **linguistic-only** | 0.0% |
| rewrite | C | 1-5 | collaborative | 0.4 | 0.8 | WARN (uninformative) | PASS | 0.073 [0.000, 0.225] | 0.265 [0.088, 0.492] | 0.800 | +0.300 | **linguistic-only** | 0.0% |
| prompting | C | 1-5 | disagreement | 0.2 | 1.0 | PASS | PASS | 0.095 [-0.100, 0.300] | 0.336 [0.171, 0.568] | 0.800 | +0.300 | **linguistic-only** | 0.0% |
| prompting | A | 1-5 | disagreement | 0.6 | 0.8 | WARN (uninformative) | PASS | -0.051 [-0.150, 0.000] | 0.370 [0.149, 0.660] | 0.700 | +0.325 | **linguistic-only** | 0.0% |

## Per-Feature Linguistic Accommodation Breakdown (Style Distance Reduction)

| Method | Trait | Pair | Task Type | Word Count | Hedges/100 | Contractions | Word Length | 1p Pronouns | Sentiment | BoW Cosine | Net Composite |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompting | C | 1-5 | collaborative | -0.116 | +0.239 | -0.771 | +0.003 | +0.737 | -0.286 | -0.231 | -0.061 |
| prompting | A | 1-5 | collaborative | +1.206 | +1.262 | +0.012 | +1.258 | -0.074 | -0.555 | +0.903 | +0.573 |
| prompting | E | 1-5 | collaborative | +1.002 | +0.455 | -0.663 | +0.629 | +0.101 | -0.125 | +0.276 | +0.239 |
| prompting | C | 3-3 | collaborative | -0.276 | -0.147 | +0.186 | -0.201 | -0.518 | +0.276 | +0.121 | -0.080 |
| steering | A | 1-5 | collaborative | +1.042 | -0.081 | +0.649 | +0.590 | +0.036 | +1.605 | +0.339 | +0.597 |
| rewrite | C | 1-5 | collaborative | +0.009 | -0.385 | +0.826 | +0.052 | +0.122 | +1.075 | +0.159 | +0.265 |
| prompting | C | 1-5 | disagreement | +0.021 | +0.776 | -0.534 | +0.610 | +0.523 | +1.045 | -0.089 | +0.336 |
| prompting | A | 1-5 | disagreement | -0.181 | +0.606 | +0.014 | +1.337 | +0.624 | -0.452 | +0.645 | +0.370 |

### Interpretation Guide
- **neither**: No significant shift in either behavioral decisions or linguistic style.
- **linguistic-only**: Surface style accommodation without behavioral compromise.
- **behavioral-only**: Changes in underlying trait-linked decisions without conversational style change.
- **both**: Deeper social drift affecting both linguistic style and behavioral decisions.

### Validity Status
- **prompting - C (1-5) [collaborative]**: Manipulation Check: `PASS`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
- **prompting - A (1-5) [collaborative]**: Manipulation Check: `WARN`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
- **prompting - E (1-5) [collaborative]**: Manipulation Check: `PASS`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
- **prompting - C (3-3) [collaborative]**: Manipulation Check: `N/A`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
- **steering - A (1-5) [collaborative]**: Manipulation Check: `WARN`, Headroom: `WARN (ceiling-saturated)`, Invalid Rate: `PASS` (0.0%)
- **rewrite - C (1-5) [collaborative]**: Manipulation Check: `WARN`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
- **prompting - C (1-5) [disagreement]**: Manipulation Check: `PASS`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
- **prompting - A (1-5) [disagreement]**: Manipulation Check: `WARN`, Headroom: `PASS`, Invalid Rate: `PASS` (0.0%)
