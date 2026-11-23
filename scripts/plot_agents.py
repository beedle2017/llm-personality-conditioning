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
    """Plot behavioral endorsement curves across checkpoints with 95% CIs."""
    if probes_df.empty:
        return

    # Filter to interactive condition and valid scores
    valid_df = probes_df[(probes_df["condition"] == "interactive") & (probes_df["score"].notna())].copy()
    valid_df["score"] = pd.to_numeric(valid_df["score"], errors="coerce")
    valid_df = valid_df.dropna(subset=["score"])

    methods = valid_df["method"].unique()
    traits = valid_df["trait"].unique()

    fig, axes = plt.subplots(len(methods), len(traits), figsize=(4 * len(traits), 3.5 * len(methods)), squeeze=False)

    for i, method in enumerate(methods):
        for j, trait in enumerate(traits):
            ax = axes[i, j]
            sub = valid_df[(valid_df["method"] == method) & (valid_df["trait"] == trait)]
            if sub.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                continue

            pairs = sub["pair"].unique()
            for pair in sorted(pairs):
                pair_sub = sub[sub["pair"] == pair]
                for agent, color, marker in [("A", "#1f77b4", "o"), ("B", "#d62728", "s")]:
                    ag_sub = pair_sub[pair_sub["agent"] == agent]
                    if ag_sub.empty:
                        continue
                    lvl = ag_sub["level"].iloc[0]
                    cps = sorted(ag_sub["checkpoint"].unique())
                    means = []
                    lows = []
                    highs = []
                    for cp in cps:
                        vals = ag_sub[ag_sub["checkpoint"] == cp]["score"].values
                        if len(vals) > 0:
                            m = np.mean(vals)
                            # Simple standard error approx for curve plotting
                            se = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                            means.append(m)
                            lows.append(m - 1.96 * se)
                            highs.append(m + 1.96 * se)
                        else:
                            means.append(np.nan)
                            lows.append(np.nan)
                            highs.append(np.nan)

                    label = f"Pair {pair} - Ag {agent} (lvl {lvl})"
                    ax.plot(cps, means, label=label, marker=marker, linewidth=1.5, alpha=0.85)
                    ax.fill_between(cps, lows, highs, alpha=0.15)

            ax.set_title(f"{method.capitalize()} - Trait {trait}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Dialogue Round (Checkpoint)", fontsize=9)
            ax.set_ylabel("Endorsement Rate", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", alpha=0.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_net_shifts_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot bar chart comparing net behavioral shift vs linguistic accommodation shift."""
    if summary_df.empty:
        return

    df = summary_df.copy()
    labels = [f"{r['method'][:4]}-{r['trait']}-{r['pair']}" for _, r in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    b_shifts = df["net_social_shift"].values
    b_err_low = np.maximum(0, b_shifts - df["behavioral_ci_low"].values)
    b_err_high = np.maximum(0, df["behavioral_ci_high"].values - b_shifts)

    l_shifts = df["linguistic_shift"].values
    l_err_low = np.maximum(0, l_shifts - df["linguistic_ci_low"].values)
    l_err_high = np.maximum(0, df["linguistic_ci_high"].values - l_shifts)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))

    rects1 = ax.bar(
        x - width / 2,
        b_shifts,
        width,
        yerr=[b_err_low, b_err_high],
        label="Net Behavioral Shift",
        color="#2ca02c",
        capsize=3,
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        l_shifts,
        width,
        yerr=[l_err_low, l_err_high],
        label="Net Linguistic Shift (Composite)",
        color="#1f77b4",
        capsize=3,
        alpha=0.85,
    )

    ax.set_ylabel("Standardized Shift toward Partner", fontsize=10)
    ax.set_title("Multi-Agent Social Robustness: Behavioral vs Linguistic Accommodation", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9)
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
