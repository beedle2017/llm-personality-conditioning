"""Create method-agnostic steering result figures and summary tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "steering" / "results.csv"
FIGURES = ROOT / "artifacts" / "steering" / "figures"


def _read() -> pd.DataFrame:
    if not RESULTS.exists() or RESULTS.stat().st_size <= len("method,trait,layer,k,coeff,target_score,context,metric,value,ci_low,ci_high,n,seed\n"):
        return pd.DataFrame(columns=["method", "trait", "layer", "k", "coeff", "target_score", "context", "metric", "value", "ci_low", "ci_high", "n", "seed"])
    return pd.read_csv(RESULTS)


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = _read()
    traits = sorted(frame["trait"].dropna().unique()) if not frame.empty else ["none"]
    methods = sorted(frame["method"].dropna().unique()) if not frame.empty else []

    fig, axes = plt.subplots(1, len(traits), figsize=(4 * len(traits), 3), squeeze=False)
    for axis, trait in zip(axes[0], traits):
        subset = frame[frame.trait == trait]
        for method in methods:
            line = subset[subset.method == method].sort_values("target_score")
            if line.empty:
                continue
            axis.plot(line.target_score, line.value, marker="o", label=method)
            axis.fill_between(line.target_score, line.ci_low, line.ci_high, alpha=0.15)
        axis.set_title(trait)
        axis.set_xlabel("target score")
        axis.set_ylabel("pair-preference margin")
    if methods:
        axes[0][0].legend()
    _save(fig, "dose_response")

    for name in ("layer_sweep", "k_calibration", "context_heatmap", "invalid_rates", "behaviorbench_pis", "bfi_error"):
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.text(0.5, 0.5, "Not run: no corresponding result rows", ha="center", va="center")
        axis.set_axis_off()
        _save(fig, name)

    summary_columns = ["trait", "layer", "k", "slope", "spearman", "invalid_rate", "n_dev"]
    summary = pd.DataFrame(columns=summary_columns)
    if not frame.empty:
        rows = []
        for trait, group in frame.groupby("trait"):
            means = group[group.metric == "pair_margin"].groupby("target_score").value.mean()
            slope = float(means.iloc[-1] - means.iloc[0]) / (means.index[-1] - means.index[0]) if len(means) > 1 else float("nan")
            rows.append({"trait": trait, "layer": group.layer.iloc[0], "k": group.k.iloc[0], "slope": slope, "spearman": float("nan"), "invalid_rate": float("nan"), "n_dev": group.n.max()})
        summary = pd.DataFrame(rows, columns=summary_columns)
    summary.to_csv(FIGURES / "summary.csv", index=False)
    lines = [
        "| " + " | ".join(summary_columns) + " |",
        "| " + " | ".join("---" for _ in summary_columns) + " |",
    ]
    for row in summary.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    (FIGURES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
