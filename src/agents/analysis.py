"""Statistical analysis, bootstrap confidence intervals, and 2x2 classification."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 20261026,
) -> BootstrapResult:
    """Compute deterministic two-sided bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"))
    if len(arr) == 1 or np.all(arr == arr[0]):
        val = float(arr[0])
        return BootstrapResult(val, val, val)

    rng = np.random.default_rng(seed)
    n = len(arr)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = np.mean(arr[indices], axis=1)

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    ci_low = float(np.percentile(resampled_means, low_pct))
    ci_high = float(np.percentile(resampled_means, high_pct))
    mean_val = float(np.mean(arr))

    return BootstrapResult(mean=mean_val, ci_low=ci_low, ci_high=ci_high)


def compute_agent_endorsements(
    probe_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, int, int], dict[str, Any]]:
    """Compute endorsement rates r_{i,k} per (method, trait, pair, agent, level, checkpoint).

    Returns a dict mapping key -> {mean_endorsement, invalid_rate, valid_scores, n_total}.
    """
    grouped: dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]] = {}
    for r in probe_records:
        key = (
            r["method"],
            r["trait"],
            r["pair"],
            r["agent"],
            r["level"],
            r["checkpoint"],
        )
        grouped.setdefault(key, []).append(r)

    results = {}
    for key, items in grouped.items():
        valid_items = [it for it in items if not it["invalid"] and it["score"] is not None]
        scores = [float(it["score"]) for it in valid_items]
        invalid_rate = (len(items) - len(valid_items)) / max(1, len(items))
        mean_score = float(np.mean(scores)) if scores else float("nan")
        results[key] = {
            "mean": mean_score,
            "invalid_rate": invalid_rate,
            "scores": scores,
            "n_total": len(items),
            "n_valid": len(valid_items),
        }
    return results


def analyze_pair_condition(
    interactive_probes: list[dict[str, Any]],
    solo_probes: list[dict[str, Any]] | None = None,
    interactive_transcripts: list[dict[str, Any]] | None = None,
    solo_transcripts: list[dict[str, Any]] | None = None,
    n_resamples: int = 1000,
    seed: int = 20261026,
) -> dict[str, Any]:
    """Perform full metrics computation for one (method, trait, pair) condition.

    Computes:
    - Manipulation check at k=0
    - Headroom bounds check
    - Behavioral drift r_{i,k} - r_{i,0}
    - Social shift S_{i,k} = d_i * (r_{i,k} - r_{i,0})
    - Net social shift (subtracting solo control)
    - Convergence gap |r_{A,k} - r_{B,k}|
    - Linguistic shifts and net composite score
    - 2x2 classification
    """
    if not interactive_probes:
        return {}

    first = interactive_probes[0]
    method = first["method"]
    trait = first["trait"]
    pair_str = first["pair"]
    parts = [int(p) for p in pair_str.split("-")]
    level_a, level_b = parts[0], parts[1]

    endorsements = compute_agent_endorsements(interactive_probes)
    solo_endorsements = compute_agent_endorsements(solo_probes) if solo_probes else {}

    # Checkpoints present
    checkpoints = sorted(list({r["checkpoint"] for r in interactive_probes}))
    final_cp = checkpoints[-1] if checkpoints else 8

    # Extract k=0 endorsements
    key_a_0 = (method, trait, pair_str, "A", level_a, 0)
    key_b_0 = (method, trait, pair_str, "B", level_b, 0)
    r_a_0 = endorsements.get(key_a_0, {}).get("mean", float("nan"))
    r_b_0 = endorsements.get(key_b_0, {}).get("mean", float("nan"))

    # Manipulation check at k=0
    # Higher level agent must have higher endorsement than lower level agent with CI > 0
    scores_a_0 = endorsements.get(key_a_0, {}).get("scores", [])
    scores_b_0 = endorsements.get(key_b_0, {}).get("scores", [])

    is_uninformative = False
    manipulation_p = float("nan")
    manipulation_ci = BootstrapResult(0.0, 0.0, 0.0)

    if level_a != level_b and scores_a_0 and scores_b_0:
        higher_scores = scores_b_0 if level_b > level_a else scores_a_0
        lower_scores = scores_a_0 if level_b > level_a else scores_b_0
        # Compute paired or difference CI
        n_min = min(len(higher_scores), len(lower_scores))
        diffs = [h - l for h, l in zip(higher_scores[:n_min], lower_scores[:n_min])]
        manipulation_ci = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)
        if manipulation_ci.ci_low <= 0:
            is_uninformative = True

    # Directions d_i
    d_a = 1 if level_b > level_a else (-1 if level_b < level_a else 0)
    d_b = 1 if level_a > level_b else (-1 if level_a < level_b else 0)

    # Headroom check
    # if r_0 >= 0.95 or <= 0.05 and d_i pushes further
    ceiling_limited_a = (r_a_0 >= 0.95 and d_a > 0) or (r_a_0 <= 0.05 and d_a < 0)
    ceiling_limited_b = (r_b_0 >= 0.95 and d_b > 0) or (r_b_0 <= 0.05 and d_b < 0)
    is_ceiling_limited = ceiling_limited_a or ceiling_limited_b

    # Behavioral shifts across checkpoints
    shifts_a: dict[int, float] = {}
    shifts_b: dict[int, float] = {}
    net_shifts_a: dict[int, float] = {}
    net_shifts_b: dict[int, float] = {}
    convergence_gaps: dict[int, float] = {}

    for cp in checkpoints:
        key_a_k = (method, trait, pair_str, "A", level_a, cp)
        key_b_k = (method, trait, pair_str, "B", level_b, cp)
        r_a_k = endorsements.get(key_a_k, {}).get("mean", float("nan"))
        r_b_k = endorsements.get(key_b_k, {}).get("mean", float("nan"))

        drift_a = r_a_k - r_a_0
        drift_b = r_b_k - r_b_0

        s_a = d_a * drift_a if d_a != 0 else abs(drift_a)
        s_b = d_b * drift_b if d_b != 0 else abs(drift_b)
        shifts_a[cp] = s_a
        shifts_b[cp] = s_b

        # Solo control drift
        s_solo_a = 0.0
        s_solo_b = 0.0
        if solo_endorsements:
            key_solo_a_0 = (method, trait, pair_str, "A", level_a, 0)
            key_solo_a_k = (method, trait, pair_str, "A", level_a, cp)
            r_solo_a_0 = solo_endorsements.get(key_solo_a_0, {}).get("mean", r_a_0)
            r_solo_a_k = solo_endorsements.get(key_solo_a_k, {}).get("mean", r_solo_a_0)
            s_solo_a = d_a * (r_solo_a_k - r_solo_a_0)

            key_solo_b_0 = (method, trait, pair_str, "B", level_b, 0)
            key_solo_b_k = (method, trait, pair_str, "B", level_b, cp)
            r_solo_b_0 = solo_endorsements.get(key_solo_b_0, {}).get("mean", r_b_0)
            r_solo_b_k = solo_endorsements.get(key_solo_b_k, {}).get("mean", r_solo_b_0)
            s_solo_b = d_b * (r_solo_b_k - r_solo_b_0)

        net_shifts_a[cp] = s_a - s_solo_a
        net_shifts_b[cp] = s_b - s_solo_b
        convergence_gaps[cp] = abs(r_a_k - r_b_k)

    # Bootstrap CIs for net social shift at final checkpoint
    # Compute per-dialogue social shifts
    dialogue_ids = sorted(list({r["dialogue_id"] for r in interactive_probes}))
    pair_net_shifts: list[float] = []

    for d_id in dialogue_ids:
        d_probes_0 = [r for r in interactive_probes if r["dialogue_id"] == d_id and r["checkpoint"] == 0]
        d_probes_k = [r for r in interactive_probes if r["dialogue_id"] == d_id and r["checkpoint"] == final_cp]

        # Agent A
        a0 = [float(r["score"]) for r in d_probes_0 if r["agent"] == "A" and r["score"] is not None]
        ak = [float(r["score"]) for r in d_probes_k if r["agent"] == "A" and r["score"] is not None]
        if a0 and ak:
            shift_a_val = d_a * (np.mean(ak) - np.mean(a0))
            pair_net_shifts.append(float(shift_a_val))

        # Agent B
        b0 = [float(r["score"]) for r in d_probes_0 if r["agent"] == "B" and r["score"] is not None]
        bk = [float(r["score"]) for r in d_probes_k if r["agent"] == "B" and r["score"] is not None]
        if b0 and bk:
            shift_b_val = d_b * (np.mean(bk) - np.mean(b0))
            pair_net_shifts.append(float(shift_b_val))

    # Subtract mean solo drift if available
    mean_solo_shift = (shifts_a.get(final_cp, 0.0) - net_shifts_a.get(final_cp, 0.0) +
                       shifts_b.get(final_cp, 0.0) - net_shifts_b.get(final_cp, 0.0)) / 2.0
    adjusted_shifts = [s - mean_solo_shift for s in pair_net_shifts] if pair_net_shifts else [0.0]
    behavioral_ci = bootstrap_ci(adjusted_shifts, n_resamples=n_resamples, seed=seed)

    # Linguistic shift analysis
    from .linguistic import compute_shifts_for_dialogue

    ling_shifts: list[float] = []
    if interactive_transcripts:
        # Group transcripts by dialogue_id
        d_trans: dict[int, list[dict[str, Any]]] = {}
        for t in interactive_transcripts:
            d_trans.setdefault(t["dialogue_id"], []).append(t)

        all_d_shifts = []
        for d_id, turns in d_trans.items():
            shifts = compute_shifts_for_dialogue(turns)
            all_d_shifts.append(shifts)

        # Standardize features across dialogues and compute composite
        feature_vals: dict[str, list[float]] = {}
        for s in all_d_shifts:
            for spk in ("A", "B"):
                for f, val in s.get(spk, {}).items():
                    feature_vals.setdefault(f, []).append(val)

        f_means = {f: float(np.mean(vals)) if vals else 0.0 for f, vals in feature_vals.items()}
        f_stds = {f: float(np.std(vals)) if len(vals) > 1 and np.std(vals) > 1e-6 else 1.0 for f, vals in feature_vals.items()}

        for s in all_d_shifts:
            for spk in ("A", "B"):
                spk_shifts = s.get(spk, {})
                if spk_shifts:
                    z_scores = [(spk_shifts[f] - f_means[f]) / f_stds[f] for f in spk_shifts]
                    ling_shifts.append(float(np.mean(z_scores)))

    linguistic_ci = bootstrap_ci(ling_shifts if ling_shifts else [0.0], n_resamples=n_resamples, seed=seed)

    # 2x2 Classification
    has_behavioral_drift = behavioral_ci.ci_low > 0
    has_linguistic_drift = linguistic_ci.ci_low > 0

    if has_behavioral_drift and has_linguistic_drift:
        classification = "both"
    elif has_behavioral_drift and not has_linguistic_drift:
        classification = "behavioral-only"
    elif not has_behavioral_drift and has_linguistic_drift:
        classification = "linguistic-only"
    else:
        classification = "neither"

    # Overall invalid rate across probe responses
    total_probes = len(interactive_probes)
    invalid_probes = sum(1 for r in interactive_probes if r.get("invalid"))
    overall_invalid_rate = invalid_probes / max(1, total_probes)

    return {
        "method": method,
        "trait": trait,
        "pair": pair_str,
        "level_a": level_a,
        "level_b": level_b,
        "final_checkpoint": final_cp,
        "r_a_0": r_a_0,
        "r_b_0": r_b_0,
        "is_uninformative": is_uninformative,
        "is_ceiling_limited": is_ceiling_limited,
        "manipulation_ci": manipulation_ci,
        "shifts_a": shifts_a,
        "shifts_b": shifts_b,
        "net_shifts_a": net_shifts_a,
        "net_shifts_b": net_shifts_b,
        "convergence_gaps": convergence_gaps,
        "behavioral_shift_mean": behavioral_ci.mean,
        "behavioral_ci_low": behavioral_ci.ci_low,
        "behavioral_ci_high": behavioral_ci.ci_high,
        "linguistic_shift_mean": linguistic_ci.mean,
        "linguistic_ci_low": linguistic_ci.ci_low,
        "linguistic_ci_high": linguistic_ci.ci_high,
        "classification": classification,
        "invalid_rate": overall_invalid_rate,
    }


def generate_summary_tables(
    analyses: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write summary.csv and summary.md containing metrics and 2x2 classifications."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    fields = [
        "method",
        "trait",
        "pair",
        "level_a",
        "level_b",
        "r_a_0",
        "r_b_0",
        "manipulation_check",
        "headroom_check",
        "net_social_shift",
        "behavioral_ci_low",
        "behavioral_ci_high",
        "linguistic_shift",
        "linguistic_ci_low",
        "linguistic_ci_high",
        "classification",
        "invalid_rate",
    ]

    rows = []
    for a in analyses:
        manip_flag = "WARN (uninformative)" if a["is_uninformative"] else "PASS"
        head_flag = "WARN (ceiling-limited)" if a["is_ceiling_limited"] else "PASS"
        rows.append({
            "method": a["method"],
            "trait": a["trait"],
            "pair": a["pair"],
            "level_a": a["level_a"],
            "level_b": a["level_b"],
            "r_a_0": round(a["r_a_0"], 4) if not math.isnan(a["r_a_0"]) else "",
            "r_b_0": round(a["r_b_0"], 4) if not math.isnan(a["r_b_0"]) else "",
            "manipulation_check": manip_flag,
            "headroom_check": head_flag,
            "net_social_shift": round(a["behavioral_shift_mean"], 4),
            "behavioral_ci_low": round(a["behavioral_ci_low"], 4),
            "behavioral_ci_high": round(a["behavioral_ci_high"], 4),
            "linguistic_shift": round(a["linguistic_shift_mean"], 4),
            "linguistic_ci_low": round(a["linguistic_ci_low"], 4),
            "linguistic_ci_high": round(a["linguistic_ci_high"], 4),
            "classification": a["classification"],
            "invalid_rate": round(a["invalid_rate"], 4),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Write Markdown summary
    md_lines = [
        "# Multi-Agent Social Robustness Summary",
        "",
        "## 2x2 Accommodation Classification and Behavioral Shift",
        "",
        "| Method | Trait | Pair | r_A(0) | r_B(0) | Manip Check | Headroom | Net Shift [95% CI] | Ling Shift [95% CI] | Classification | Invalid Rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b_ci = f"{r['net_social_shift']:.3f} [{r['behavioral_ci_low']:.3f}, {r['behavioral_ci_high']:.3f}]"
        l_ci = f"{r['linguistic_shift']:.3f} [{r['linguistic_ci_low']:.3f}, {r['linguistic_ci_high']:.3f}]"
        md_lines.append(
            f"| {r['method']} | {r['trait']} | {r['pair']} | {r['r_a_0']} | {r['r_b_0']} | "
            f"{r['manipulation_check']} | {r['headroom_check']} | {b_ci} | {l_ci} | "
            f"**{r['classification']}** | {r['invalid_rate']:.1%} |"
        )

    md_lines.extend([
        "",
        "### Interpretation Guide",
        "- **neither**: No significant shift in either behavioral decisions or linguistic style.",
        "- **linguistic-only**: Surface style accommodation without behavioral compromise.",
        "- **behavioral-only**: Changes in underlying trait-linked decisions without conversational style change.",
        "- **both**: Deeper social drift affecting both linguistic style and behavioral decisions.",
        "",
        "### Validity Status",
    ])

    for a in analyses:
        m_status = "WARN" if a["is_uninformative"] else "PASS"
        h_status = "WARN" if a["is_ceiling_limited"] else "PASS"
        inv_status = "PASS" if a["invalid_rate"] < 0.20 else "WARN"
        md_lines.append(
            f"- **{a['method']} - {a['trait']} ({a['pair']})**: "
            f"Manipulation Check: `{m_status}`, Headroom: `{h_status}`, Invalid Rate: `{inv_status}` ({a['invalid_rate']:.1%})"
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path
