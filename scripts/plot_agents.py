"""Plotting script for multi-agent social robustness experiment.

Generates:
1. Drift curves vs checkpoint per method with 95% bootstrap CIs.
2. Grouped bar plot of net behavioral vs linguistic shift with error bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def plot_drift_curves(probes_df: pd.DataFrame, out_path: Path) -> None:
    """Plot behavioral endorsement curves across checkpoints with 95% CIs.

    Shows only (method, trait) panels that have data, with larger fonts,
    explicit checkpoint markers, n in panel title, and labeled lines
    for Agent L1, Agent L5, and Solo controls.
    """
    if probes_df.empty:
        return

    valid_df = probes_df[probes_df["score"].notna()].copy()
    valid_df["score"] = pd.to_numeric(valid_df["score"], errors="coerce")
    valid_df = valid_df.dropna(subset=["score"])

    # Identify panels (method, trait) that have interactive data
    interactive_df = valid_df[valid_df["condition"] == "interactive"]
    active_panels = []
    seen = set()
    for _, r in interactive_df.iterrows():
        key = (r["method"], r["trait"])
        if key not in seen:
            seen.add(key)
            active_panels.append(key)

    if not active_panels:
        return

    n_panels = len(active_panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.5 * nrows), squeeze=False)

    colors_ag = {"1": "#1f77b4", "5": "#d62728", "3": "#9467bd"}
    markers_ag = {"1": "o", "5": "s", "3": "D"}
    solo_colors = {"1": "#5c768d", "5": "#b35959", "3": "#75598d"}

    for idx, (method, trait) in enumerate(active_panels):
        r_idx = idx // ncols
        c_idx = idx % ncols
        ax = axes[r_idx, c_idx]

        sub_inter = valid_df[(valid_df["method"] == method) & (valid_df["trait"] == trait) & (valid_df["condition"] == "interactive")]
        sub_solo = valid_df[(valid_df["method"] == method) & (valid_df["trait"] == trait) & (valid_df["condition"] == "solo_control")]

        n_diag = sub_inter["dialogue_id"].nunique()
        n_items = sub_inter["item_id"].nunique()
        ax.set_title(f"{method.capitalize()} · Trait {trait} (n={n_diag} dialogues, {n_items} items)", fontsize=13, fontweight="bold", pad=10)

        # Plot interactive lines
        pairs = sorted(sub_inter["pair"].unique())
        for pair in pairs:
            pair_sub = sub_inter[sub_inter["pair"] == pair]
            for agent in ("A", "B"):
                ag_sub = pair_sub[pair_sub["agent"] == agent]
                if ag_sub.empty:
                    continue
                lvl = int(ag_sub["level"].iloc[0])
                cps = sorted(ag_sub["checkpoint"].unique())
                means = []
                lows = []
                highs = []
                for cp in cps:
                    vals = ag_sub[ag_sub["checkpoint"] == cp]["score"].values
                    if len(vals) > 0:
                        m = float(np.mean(vals))
                        se = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                        means.append(m)
                        lows.append(m - 1.96 * se)
                        highs.append(m + 1.96 * se)
                    else:
                        means.append(np.nan)
                        lows.append(np.nan)
                        highs.append(np.nan)

                label_base = f"Agent L{lvl}"
                if pair == "3-3":
                    label_base = f"Agent L3 ({agent})"
                elif len(pairs) > 1 and pair != "1-5":
                    label_base = f"Agent L{lvl} ({pair})"

                c = colors_ag.get(str(lvl), "#2ca02c")
                m = markers_ag.get(str(lvl), "o")
                ax.plot(cps, means, label=label_base, color=c, marker=m, markersize=8, markeredgewidth=1.5, linewidth=2.0, alpha=0.9)
                ax.fill_between(cps, lows, highs, color=c, alpha=0.15)

        # Plot solo control lines if present
        if not sub_solo.empty:
            solo_lvls = sorted(sub_solo["level"].unique())
            for s_lvl in solo_lvls:
                s_sub = sub_solo[sub_solo["level"] == s_lvl]
                s_cps = sorted(s_sub["checkpoint"].unique())
                s_means = []
                for cp in s_cps:
                    s_vals = s_sub[s_sub["checkpoint"] == cp]["score"].values
                    s_means.append(float(np.mean(s_vals)) if len(s_vals) > 0 else np.nan)

                s_col = solo_colors.get(str(int(s_lvl)), "#7f7f7f")
                ax.plot(
                    s_cps,
                    s_means,
                    label=f"Solo control L{int(s_lvl)}",
                    color=s_col,
                    linestyle="--",
                    marker="^",
                    markersize=7,
                    linewidth=1.8,
                    alpha=0.75,
                )

        ax.set_xlabel("Dialogue Checkpoint", fontsize=11, fontweight="bold")
        ax.set_ylabel("Endorsement Rate", fontsize=11, fontweight="bold")
        all_cps = sorted(sub_inter["checkpoint"].unique())
        ax.set_xticks(all_cps)
        ax.set_xticklabels([f"k={cp}" for cp in all_cps], fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

    # Hide any unused subplots
    for empty_idx in range(n_panels, nrows * ncols):
        er_idx = empty_idx // ncols
        ec_idx = empty_idx % ncols
        axes[er_idx, ec_idx].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_net_shifts_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart comparing net behavioral shift vs linguistic accommodation shift."""
    if summary_df.empty:
        return

    df = summary_df.copy()
    # Format readable labels such as "Prompting · C · 1 vs 5"
    labels = []
    for _, r in df.iterrows():
        m_name = str(r["method"]).capitalize()
        t_name = str(r["trait"])
        la = r.get("level_a", "")
        lb = r.get("level_b", "")
        labels.append(f"{m_name} · {t_name} · {la} vs {lb}")

    x = np.arange(len(labels))
    width = 0.35

    b_shifts = df["net_social_shift"].values
    b_err_low = np.maximum(0, b_shifts - df["behavioral_ci_low"].values)
    b_err_high = np.maximum(0, df["behavioral_ci_high"].values - b_shifts)

    l_shifts = df["linguistic_shift"].values
    l_err_low = np.maximum(0, l_shifts - df["linguistic_ci_low"].values)
    l_err_high = np.maximum(0, df["linguistic_ci_high"].values - l_shifts)

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.5), 5.5))

    rects1 = ax.bar(
        x - width / 2,
        b_shifts,
        width,
        yerr=[b_err_low, b_err_high],
        label="Net Behavioral Shift",
        color="#2ca02c",
        capsize=4,
        alpha=0.88,
        edgecolor="#1e731e",
        linewidth=1.0,
    )
    rects2 = ax.bar(
        x + width / 2,
        l_shifts,
        width,
        yerr=[l_err_low, l_err_high],
        label="Net Linguistic Shift (Style Distance Reduction)",
        color="#1f77b4",
        capsize=4,
        alpha=0.88,
        edgecolor="#144d75",
        linewidth=1.0,
    )

    ax.set_ylabel("Standardized Shift / Distance Reduction", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Agent Social Robustness: Behavioral vs. Linguistic Accommodation", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for multi-agent robustness")
    parser.add_argument("--probes", type=Path, default=Path("artifacts/agents/probes.csv"), help="Probes CSV")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/agents/summary.csv"), help="Summary CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/agents/figures"), help="Output figures directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.probes.exists():
        probes_df = pd.read_csv(args.probes)
        plot_drift_curves(probes_df, args.out_dir / "drift_curves.png")
        print(f"Drift curves saved to {args.out_dir / 'drift_curves.png'}")

    if args.summary.exists():
        summary_df = pd.read_csv(args.summary)
        plot_net_shifts_bar(summary_df, args.out_dir / "net_shifts_bar.png")
        print(f"Net shifts bar chart saved to {args.out_dir / 'net_shifts_bar.png'}")


if __name__ == "__main__":
    main()
