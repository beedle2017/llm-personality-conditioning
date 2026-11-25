"""Statistical analysis, bootstrap confidence intervals, and 2x2 classification."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Baseline endorsement rates from artifacts/eval/default_behaviorbench.csv
BASELINE_RATES: dict[str, float] = {
    "C": 0.7667,  # Conscientiousness
    "A": 0.9000,  # Agreeableness
    "E": 1.0000,  # Extraversion
    "O": 0.9333,  # Openness
    "N": 0.4600,  # Neuroticism
}


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
    task_type = first.get("task_type", "collaborative")
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
    # Dedupe cached k=0 rows and bootstrap over items only
    is_uninformative = False
    manipulation_ci = BootstrapResult(0.0, 0.0, 0.0)

    if level_a != level_b:
        agent_high = "B" if level_b > level_a else "A"
        agent_low = "A" if level_b > level_a else "B"

        s0_items_high: dict[str, float] = {}
        s0_items_low: dict[str, float] = {}
        for r in interactive_probes:
            if r.get("checkpoint") == 0 and not r.get("invalid") and r.get("score") is not None:
                agent_role = r["agent"]
                it_id = r.get("item_id", "item_0")
                if agent_role == agent_high and it_id not in s0_items_high:
                    s0_items_high[it_id] = float(r["score"])
                elif agent_role == agent_low and it_id not in s0_items_low:
                    s0_items_low[it_id] = float(r["score"])

        common_items = [it for it in s0_items_high if it in s0_items_low]
        if common_items:
            diffs = [s0_items_high[it] - s0_items_low[it] for it in common_items]
            diffs_arr = np.asarray(diffs, dtype=float)
            rng_mc = np.random.default_rng(seed)
            n_items = len(diffs_arr)
            boot_diff_means = []
            for _ in range(n_resamples):
                idx_sample = rng_mc.integers(0, n_items, size=n_items)
                boot_diff_means.append(float(np.mean(diffs_arr[idx_sample])))

            manip_mean = float(np.mean(diffs_arr))
            manip_ci_low = float(np.percentile(boot_diff_means, 2.5))
            manip_ci_high = float(np.percentile(boot_diff_means, 97.5))
            manipulation_ci = BootstrapResult(mean=manip_mean, ci_low=manip_ci_low, ci_high=manip_ci_high)

            if manip_ci_low <= 0:
                is_uninformative = True
        else:
            is_uninformative = True

    # Directions d_i
    d_a = 1 if level_b > level_a else (-1 if level_b < level_a else 0)
    d_b = 1 if level_a > level_b else (-1 if level_a < level_b else 0)

    # Headroom check
    # if r_0 >= 0.95 or <= 0.05 and d_i pushes further
    ceiling_limited_a = (r_a_0 >= 0.95 and d_a > 0) or (r_a_0 <= 0.05 and d_a < 0)
    ceiling_limited_b = (r_b_0 >= 0.95 and d_b > 0) or (r_b_0 <= 0.05 and d_b < 0)
    is_ceiling_limited = ceiling_limited_a or ceiling_limited_b

    # Check for ceiling saturation at final checkpoint
    k_final_interactive_probes = [
        r for r in interactive_probes
        if r.get("checkpoint") == final_cp and not r.get("invalid") and r.get("score") is not None
    ]
    is_ceiling_saturated = (
        len(k_final_interactive_probes) > 0
        and all(float(r["score"]) >= 0.999 for r in k_final_interactive_probes)
    )

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

    # Fixed Bootstrap CIs for net social shift at final checkpoint:
    # "cached checkpoint-0 probes are identical across dialogues, so resample items for k=0 and dialogues×items only for k=6."
    # "For same-level pairs, report real absolute drift per agent instead of 0 by construction."
    unique_items = sorted(list({r.get("item_id", "item_0") for r in interactive_probes if not r.get("invalid") and r.get("score") is not None}))
    dialogue_ids = sorted(list({r["dialogue_id"] for r in interactive_probes}))

    # Map k=0 scores by agent and item
    s0_map_a: dict[str, float] = {}
    s0_map_b: dict[str, float] = {}
    for r in interactive_probes:
        if r.get("checkpoint") == 0 and not r.get("invalid") and r.get("score") is not None:
            it = r.get("item_id", "item_0")
            if r["agent"] == "A":
                s0_map_a[it] = float(r["score"])
            elif r["agent"] == "B":
                s0_map_b[it] = float(r["score"])

    # Map k=final_cp scores by (dialogue_id, item_id)
    sk_map_a: dict[tuple[int, str], float] = {}
    sk_map_b: dict[tuple[int, str], float] = {}
    for r in interactive_probes:
        if r.get("checkpoint") == final_cp and not r.get("invalid") and r.get("score") is not None:
            it = r.get("item_id", "item_0")
            if r["agent"] == "A":
                sk_map_a[(r["dialogue_id"], it)] = float(r["score"])
            elif r["agent"] == "B":
                sk_map_b[(r["dialogue_id"], it)] = float(r["score"])

    # Solo controls mappings
    solo_s0_a: dict[str, float] = {}
    solo_s0_b: dict[str, float] = {}
    solo_sk_a: dict[tuple[int, str], float] = {}
    solo_sk_b: dict[tuple[int, str], float] = {}
    if solo_probes:
        for r in solo_probes:
            if not r.get("invalid") and r.get("score") is not None:
                lvl = r.get("level")
                cp = r.get("checkpoint")
                it = r.get("item_id", "item_0")
                d_id = r.get("dialogue_id", 0)
                sc = float(r["score"])
                if lvl == level_a:
                    if cp == 0:
                        solo_s0_a[it] = sc
                    elif cp == final_cp:
                        solo_sk_a[(d_id, it)] = sc
                if lvl == level_b:
                    if cp == 0:
                        solo_s0_b[it] = sc
                    elif cp == final_cp:
                        solo_sk_b[(d_id, it)] = sc

    rng = np.random.default_rng(seed)
    bootstrap_net_shifts: list[float] = []

    if unique_items:
        for _ in range(n_resamples):
            # Resample items for k=0
            res_items = rng.choice(unique_items, size=len(unique_items), replace=True)
            # Resample dialogues for k=final_cp
            res_diags = rng.choice(dialogue_ids, size=len(dialogue_ids), replace=True)

            # k=0 means (resampled items only)
            s0_vals_a = [s0_map_a[it] for it in res_items if it in s0_map_a]
            s0_vals_b = [s0_map_b[it] for it in res_items if it in s0_map_b]
            m0_a = np.mean(s0_vals_a) if s0_vals_a else (r_a_0 if not math.isnan(r_a_0) else 0.0)
            m0_b = np.mean(s0_vals_b) if s0_vals_b else (r_b_0 if not math.isnan(r_b_0) else 0.0)

            # k=final_cp means (dialogues x items)
            sk_vals_a = [sk_map_a[(d, it)] for d in res_diags for it in res_items if (d, it) in sk_map_a]
            sk_vals_b = [sk_map_b[(d, it)] for d in res_diags for it in res_items if (d, it) in sk_map_b]
            mk_a = np.mean(sk_vals_a) if sk_vals_a else m0_a
            mk_b = np.mean(sk_vals_b) if sk_vals_b else m0_b

            # Solo controls
            solo_s0_vals_a = [solo_s0_a.get(it, m0_a) for it in res_items]
            solo_s0_vals_b = [solo_s0_b.get(it, m0_b) for it in res_items]
            solo_m0_a = np.mean(solo_s0_vals_a) if solo_s0_vals_a else m0_a
            solo_m0_b = np.mean(solo_s0_vals_b) if solo_s0_vals_b else m0_b

            solo_sk_vals_a = [solo_sk_a.get((d, it), solo_m0_a) for d in res_diags for it in res_items if solo_sk_a]
            solo_sk_vals_b = [solo_sk_b.get((d, it), solo_m0_b) for d in res_diags for it in res_items if solo_sk_b]
            solo_mk_a = np.mean(solo_sk_vals_a) if solo_sk_vals_a else solo_m0_a
            solo_mk_b = np.mean(solo_sk_vals_b) if solo_sk_vals_b else solo_m0_b

            if level_a != level_b:
                shift_a = d_a * (mk_a - m0_a)
                shift_b = d_b * (mk_b - m0_b)
                shift_solo_a = d_a * (solo_mk_a - solo_m0_a)
                shift_solo_b = d_b * (solo_mk_b - solo_m0_b)
                net_a = shift_a - shift_solo_a
                net_b = shift_b - shift_solo_b
                net_val = 0.5 * (net_a + net_b)
            else:
                # Same-level pairs: report real absolute drift per agent instead of 0 by construction
                drift_a = abs(mk_a - m0_a)
                drift_b = abs(mk_b - m0_b)
                drift_solo_a = abs(solo_mk_a - solo_m0_a)
                drift_solo_b = abs(solo_mk_b - solo_m0_b)
                net_a = drift_a - drift_solo_a
                net_b = drift_b - drift_solo_b
                net_val = 0.5 * (net_a + net_b)

            bootstrap_net_shifts.append(float(net_val))

    if bootstrap_net_shifts:
        behavioral_ci = BootstrapResult(
            mean=float(np.mean(bootstrap_net_shifts)),
            ci_low=float(np.percentile(bootstrap_net_shifts, 2.5)),
            ci_high=float(np.percentile(bootstrap_net_shifts, 97.5)),
        )
    else:
        behavioral_ci = BootstrapResult(0.0, 0.0, 0.0)

    # Direction-free linguistic style distance analysis compared against solo controls
    from .linguistic import (
        FEATURE_NAMES,
        compute_style_distances_for_dialogue,
        extract_turn_features,
    )

    feature_breakdown: dict[str, dict[str, float]] = {}
    d_net_composites: list[float] = []

    if interactive_transcripts:
        # 1. Compute feature standard deviations across all turns in the run
        all_trans = interactive_transcripts + (solo_transcripts or [])
        all_turn_features: dict[str, list[float]] = {f: [] for f in FEATURE_NAMES if f != "bow_cosine"}
        for tr in all_trans:
            feats = extract_turn_features(tr.get("text", ""))
            for f in all_turn_features:
                all_turn_features[f].append(feats[f])

        feature_stds = {
            f: float(np.std(vals)) if len(vals) > 1 and np.std(vals) > 1e-6 else 1.0
            for f, vals in all_turn_features.items()
        }
        # BoW cosine standard deviation across typical turns
        feature_stds["bow_cosine"] = 0.20

        # 2. Group interactive transcripts by dialogue_id
        d_inter: dict[int, list[dict[str, Any]]] = {}
        for t in interactive_transcripts:
            d_inter.setdefault(t["dialogue_id"], []).append(t)

        inter_dist_by_d: dict[int, dict[str, float]] = {}
        for d_id, turns in d_inter.items():
            ta = [t for t in turns if t["speaker"] == "A"]
            tb = [t for t in turns if t["speaker"] == "B"]
            inter_dist_by_d[d_id] = compute_style_distances_for_dialogue(ta, tb, feature_stds)

        # 3. Solo control baseline style distance changes
        solo_feat_baseline: dict[str, float] = {f: 0.0 for f in FEATURE_NAMES}
        if solo_transcripts:
            s_map: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for t in solo_transcripts:
                s_map.setdefault((t["level"], t["dialogue_id"]), []).append(t)

            if level_a != level_b:
                for f in FEATURE_NAMES:
                    vals = []
                    for d in range(max(1, len(d_inter))):
                        ta = s_map.get((level_a, d), [])
                        tb = s_map.get((level_b, d), [])
                        if ta and tb:
                            dists = compute_style_distances_for_dialogue(ta, tb, feature_stds)
                            vals.append(dists[f])
                    solo_feat_baseline[f] = float(np.mean(vals)) if vals else 0.0
            else:
                # Same-level solo: uncoupled pairwise distance changes across distinct dialogues of that level
                s_diags = sorted(list({t["dialogue_id"] for t in solo_transcripts if t.get("level") == level_a}))
                for f in FEATURE_NAMES:
                    vals = []
                    for i_idx, i_d in enumerate(s_diags):
                        for j_d in s_diags[i_idx + 1:]:
                            ta = s_map.get((level_a, i_d), [])
                            tb = s_map.get((level_b, j_d), [])
                            if ta and tb:
                                dists = compute_style_distances_for_dialogue(ta, tb, feature_stds)
                                vals.append(dists[f])
                    solo_feat_baseline[f] = float(np.mean(vals)) if vals else 0.0

        # 4. Net style distance reduction per dialogue and per feature
        feat_nets_by_d: dict[str, list[float]] = {f: [] for f in FEATURE_NAMES}
        for d_id in sorted(inter_dist_by_d.keys()):
            cur_dists = inter_dist_by_d[d_id]
            d_net_vals = [cur_dists[f] - solo_feat_baseline.get(f, 0.0) for f in FEATURE_NAMES]
            for f in FEATURE_NAMES:
                feat_nets_by_d[f].append(cur_dists[f] - solo_feat_baseline.get(f, 0.0))
            d_net_composites.append(float(np.mean(d_net_vals)))

        # Per-feature breakdown with bootstrap
        for f in FEATURE_NAMES:
            f_vals = feat_nets_by_d[f]
            if f_vals:
                f_boot = [np.mean(rng.choice(f_vals, size=len(f_vals), replace=True)) for _ in range(n_resamples)]
                feature_breakdown[f] = {
                    "net_mean": float(np.mean(f_vals)),
                    "ci_low": float(np.percentile(f_boot, 2.5)),
                    "ci_high": float(np.percentile(f_boot, 97.5)),
                }
            else:
                feature_breakdown[f] = {"net_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    # Bootstrap composite linguistic shift CI across dialogues
    if d_net_composites:
        ling_boot = [np.mean(rng.choice(d_net_composites, size=len(d_net_composites), replace=True)) for _ in range(n_resamples)]
        linguistic_ci = BootstrapResult(
            mean=float(np.mean(d_net_composites)),
            ci_low=float(np.percentile(ling_boot, 2.5)),
            ci_high=float(np.percentile(ling_boot, 97.5)),
        )
    else:
        linguistic_ci = BootstrapResult(0.0, 0.0, 0.0)

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

    # Endorsements at final checkpoint
    key_a_final = (method, trait, pair_str, "A", level_a, final_cp)
    key_b_final = (method, trait, pair_str, "B", level_b, final_cp)
    r_a_final = endorsements.get(key_a_final, {}).get("mean", r_a_0)
    r_b_final = endorsements.get(key_b_final, {}).get("mean", r_b_0)

    # BFI Error: |(1 + 4r) - target|
    bfi_err_a_0 = abs((1.0 + 4.0 * r_a_0) - float(level_a)) if not math.isnan(r_a_0) else float("nan")
    bfi_err_b_0 = abs((1.0 + 4.0 * r_b_0) - float(level_b)) if not math.isnan(r_b_0) else float("nan")
    errs_0 = [e for e in (bfi_err_a_0, bfi_err_b_0) if not math.isnan(e)]
    bfi_error_0 = float(np.mean(errs_0)) if errs_0 else float("nan")

    bfi_err_a_k = abs((1.0 + 4.0 * r_a_final) - float(level_a)) if not math.isnan(r_a_final) else float("nan")
    bfi_err_b_k = abs((1.0 + 4.0 * r_b_final) - float(level_b)) if not math.isnan(r_b_final) else float("nan")
    errs_k = [e for e in (bfi_err_a_k, bfi_err_b_k) if not math.isnan(e)]
    bfi_error_final = float(np.mean(errs_k)) if errs_k else float("nan")

    # PIS: sign(target - 3) * (r - baseline_rate)
    base_rate = BASELINE_RATES.get(trait, 0.5)

    def _agent_pis(r_val: float, tgt: int) -> float:
        if math.isnan(r_val) or tgt == 3:
            return float("nan")
        direction = 1.0 if tgt > 3 else -1.0
        return direction * (r_val - base_rate)

    pis_a_0 = _agent_pis(r_a_0, level_a)
    pis_b_0 = _agent_pis(r_b_0, level_b)
    valid_pis_0 = [p for p in (pis_a_0, pis_b_0) if not math.isnan(p)]
    pis_0 = float(np.mean(valid_pis_0)) if valid_pis_0 else float("nan")

    pis_a_k = _agent_pis(r_a_final, level_a)
    pis_b_k = _agent_pis(r_b_final, level_b)
    valid_pis_k = [p for p in (pis_a_k, pis_b_k) if not math.isnan(p)]
    pis_final = float(np.mean(valid_pis_k)) if valid_pis_k else float("nan")

    return {
        "method": method,
        "trait": trait,
        "pair": pair_str,
        "task_type": task_type,
        "level_a": level_a,
        "level_b": level_b,
        "final_checkpoint": final_cp,
        "r_a_0": r_a_0,
        "r_b_0": r_b_0,
        "is_uninformative": is_uninformative,
        "is_ceiling_limited": is_ceiling_limited,
        "is_ceiling_saturated": is_ceiling_saturated,
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
        "feature_breakdown": feature_breakdown,
        "classification": classification,
        "invalid_rate": overall_invalid_rate,
        "bfi_error": bfi_error_final,
        "pis": pis_final,
        "bfi_error_0": bfi_error_0,
        "pis_0": pis_0,
    }


def generate_summary_tables(
    analyses: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write summary.csv and summary.md containing metrics and 2x2 classifications."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    from .linguistic import FEATURE_NAMES

    fields = [
        "method",
        "trait",
        "pair",
        "task_type",
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
        "bfi_error",
        "pis",
        "bfi_error_0",
        "pis_0",
        "classification",
        "invalid_rate",
    ] + [f"ling_{f}" for f in FEATURE_NAMES]

    rows = []
    for a in analyses:
        if a["level_a"] == a["level_b"]:
            manip_flag = "N/A"
        elif a["is_uninformative"]:
            manip_flag = "WARN (uninformative)"
        else:
            manip_flag = "PASS"

        if a.get("is_ceiling_saturated"):
            head_flag = "WARN (ceiling-saturated)"
        elif a.get("is_ceiling_limited"):
            head_flag = "WARN (ceiling-limited)"
        else:
            head_flag = "PASS"

        row_dict = {
            "method": a["method"],
            "trait": a["trait"],
            "pair": a["pair"],
            "task_type": a.get("task_type", "collaborative"),
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
            "bfi_error": round(a["bfi_error"], 4) if not math.isnan(a.get("bfi_error", float("nan"))) else "",
            "pis": round(a["pis"], 4) if not math.isnan(a.get("pis", float("nan"))) else "",
            "bfi_error_0": round(a["bfi_error_0"], 4) if not math.isnan(a.get("bfi_error_0", float("nan"))) else "",
            "pis_0": round(a["pis_0"], 4) if not math.isnan(a.get("pis_0", float("nan"))) else "",
            "classification": a["classification"],
            "invalid_rate": round(a["invalid_rate"], 4),
        }
        fb = a.get("feature_breakdown", {})
        for f in FEATURE_NAMES:
            row_dict[f"ling_{f}"] = round(fb.get(f, {}).get("net_mean", 0.0), 4)

        rows.append(row_dict)

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
        "| Method | Trait | Pair | Task Type | r_A(0) | r_B(0) | Manip Check | Headroom | Net Shift [95% CI] | Ling Shift [95% CI] | BFI Err | PIS | Classification | Invalid Rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, a in zip(rows, analyses):
        if a.get("is_ceiling_saturated"):
            b_ci = f"{r['net_social_shift']:.3f} [ceiling-saturated]"
        else:
            b_ci = f"{r['net_social_shift']:.3f} [{r['behavioral_ci_low']:.3f}, {r['behavioral_ci_high']:.3f}]"
        l_ci = f"{r['linguistic_shift']:.3f} [{r['linguistic_ci_low']:.3f}, {r['linguistic_ci_high']:.3f}]"
        bfi_str = f"{r['bfi_error']:.3f}" if r["bfi_error"] != "" else "N/A"
        pis_str = f"{r['pis']:+.3f}" if r["pis"] != "" else "N/A"
        md_lines.append(
            f"| {r['method']} | {r['trait']} | {r['pair']} | {r['task_type']} | {r['r_a_0']} | {r['r_b_0']} | "
            f"{r['manipulation_check']} | {r['headroom_check']} | {b_ci} | {l_ci} | {bfi_str} | {pis_str} | "
            f"**{r['classification']}** | {r['invalid_rate']:.1%} |"
        )

    # Add Per-Feature Breakdown Table
    md_lines.extend([
        "",
        "## Per-Feature Linguistic Accommodation Breakdown (Style Distance Reduction)",
        "",
        "| Method | Trait | Pair | Task Type | Word Count | Hedges/100 | Contractions | Word Length | 1p Pronouns | Sentiment | BoW Cosine | Net Composite |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in rows:
        md_lines.append(
            f"| {r['method']} | {r['trait']} | {r['pair']} | {r['task_type']} | "
            f"{r.get('ling_word_count', 0.0):+.3f} | {r.get('ling_hedges_per_100', 0.0):+.3f} | "
            f"{r.get('ling_contraction_rate', 0.0):+.3f} | {r.get('ling_mean_word_length', 0.0):+.3f} | "
            f"{r.get('ling_pronoun_1p_rate', 0.0):+.3f} | {r.get('ling_sentiment', 0.0):+.3f} | "
            f"{r.get('ling_bow_cosine', 0.0):+.3f} | {r['linguistic_shift']:+.3f} |"
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
        if a["level_a"] == a["level_b"]:
            m_status = "N/A"
        elif a["is_uninformative"]:
            m_status = "WARN"
        else:
            m_status = "PASS"
        if a.get("is_ceiling_saturated"):
            h_status = "WARN (ceiling-saturated)"
        elif a.get("is_ceiling_limited"):
            h_status = "WARN (ceiling-limited)"
        else:
            h_status = "PASS"
        inv_status = "PASS" if a["invalid_rate"] < 0.20 else "WARN"
        md_lines.append(
            f"- **{a['method']} - {a['trait']} ({a['pair']}) [{a.get('task_type', 'collaborative')}]**: "
            f"Manipulation Check: `{m_status}`, Headroom: `{h_status}`, Invalid Rate: `{inv_status}` ({a['invalid_rate']:.1%})"
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path
