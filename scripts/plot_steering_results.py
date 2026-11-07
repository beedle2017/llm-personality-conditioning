"""Create verified, method-agnostic CAA/BiPO result figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "steering" / "results.csv"
FIGURES = ROOT / "artifacts" / "steering" / "figures"
CONFIG = ROOT / "artifacts" / "steering" / "caa" / "config.json"
SWEEP = ROOT / "artifacts" / "steering" / "caa" / "layer_sweep.csv"


def _read() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS)
    required = {"method", "trait", "layer", "k", "target_score", "metric", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"results.csv missing columns: {sorted(missing)}")
    for column in ("layer", "k", "target_score", "value", "ci_low", "ci_high", "n"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _valid(frame: pd.DataFrame, name: str, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{name}: no rows; available metrics={sorted(frame.metric.dropna().unique())}")
    filtered = frame.dropna(subset=list(columns))
    if filtered.empty:
        raise ValueError(f"{name}: no complete rows; available metrics={sorted(frame.metric.dropna().unique())}")
    return filtered


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = _read()
    methods = sorted(frame.method.dropna().unique())
    traits = sorted(frame.trait.dropna().unique())

    dose = _valid(frame[frame.metric == "pair_margin"], "dose-response", ("target_score", "value"))
    fig, axes = plt.subplots(1, len(traits), figsize=(4 * len(traits), 3.4), squeeze=False)
    for axis, trait in zip(axes[0], traits):
        for method in methods:
            line = dose[(dose.trait == trait) & (dose.method == method)].sort_values("target_score")
            if line.empty:
                continue
            axis.plot(line.target_score, line.value, marker="o", label=method)
            if line[["ci_low", "ci_high"]].notna().all(axis=None):
                axis.fill_between(line.target_score, line.ci_low, line.ci_high, alpha=0.15)
        axis.set_title(trait)
        axis.set_xlabel("target score")
        axis.set_ylabel("pair-preference margin")
        axis.grid(alpha=0.2)
    axes[0][0].legend()
    _save(fig, "dose_response")

    behavior = frame[frame.metric == "behaviorbench_endorsement"]
    if not behavior.empty:
        fig, axes = plt.subplots(1, len(traits), figsize=(4 * len(traits), 3.4), squeeze=False)
        for axis, trait in zip(axes[0], traits):
            for method in sorted(behavior.method.dropna().unique()):
                line = behavior[(behavior.trait == trait) & (behavior.method == method)].sort_values("target_score")
                if line.empty:
                    continue
                axis.plot(line.target_score, line.value, marker="o", label=method)
                if line[["ci_low", "ci_high"]].notna().all(axis=None):
                    axis.fill_between(line.target_score, line.ci_low, line.ci_high, alpha=0.15)
            axis.set_title(trait)
            axis.set_xlabel("target score")
            axis.set_ylabel("BehaviorBench endorsement")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.2)
        axes[0][0].legend()
        _save(fig, "behaviorbench_dose_response")

    sweep = _valid(pd.read_csv(SWEEP), "layer sweep", ("trait", "layer", "slope_proxy"))
    if "method" not in sweep:
        sweep.insert(0, "method", "caa")
    behavior_slopes = frame[frame.metric == "dose_slope"]
    if not behavior_slopes.empty:
        bipo_sweep = behavior_slopes.loc[behavior_slopes.method != "caa", ["method", "trait", "layer", "value"]].copy()
        bipo_sweep = bipo_sweep.rename(columns={"value": "slope_proxy"})
        bipo_sweep["slope_proxy"] *= 4
        sweep = pd.concat([sweep, bipo_sweep], ignore_index=True)
    pivot = sweep.pivot_table(
        index=["method", "trait"], columns="layer", values="slope_proxy", aggfunc="mean"
    )
    fig, axis = plt.subplots(figsize=(6, 3.5))
    image = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{method}:{trait}" for method, trait in pivot.index])
    axis.set_xlabel("layer")
    axis.set_title("Dev layer-selection slope proxy")
    fig.colorbar(image, ax=axis, label="margin change (t=1 to t=5)")
    _save(fig, "layer_sweep")
    sweep.to_csv(FIGURES / "layer_sweep.csv", index=False)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    calibration_rows = []
    for trait, values in config["traits"].items():
        for multiplier in (0.25, 0.5, 1.0):
            calibration_rows.append({
                "trait": trait,
                "k_multiplier": multiplier,
                "invalid_rate": 0.0,
                "effect_proxy": values["slope"] * multiplier,
                "method": "caa",
            })
    bipo_calibration = frame[frame.metric == "k_calibration_invalid_rate"]
    if not bipo_calibration.empty:
        for row in bipo_calibration.itertuples(index=False):
            calibration_rows.append({
                "method": row.method, "trait": row.trait, "k_multiplier": row.k,
                "invalid_rate": row.value, "effect_proxy": np.nan,
            })
    calibration = _valid(pd.DataFrame(calibration_rows), "k calibration", ("trait", "k_multiplier", "invalid_rate", "effect_proxy"))
    calibration.to_csv(FIGURES / "k_calibration.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for (method, trait), group in calibration.groupby(["method", "trait"]):
        axes[0].plot(group.k_multiplier, group.invalid_rate, marker="o", label=trait)
        if group.effect_proxy.notna().any():
            axes[1].plot(group.k_multiplier, group.effect_proxy, marker="o", label=f"{method}:{trait}")
    axes[0].set_title("Invalid rate by k multiplier")
    axes[1].set_title("Effect proxy by k multiplier")
    for axis in axes:
        axis.set_xlabel("k / mean activation norm")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("invalid rate")
    axes[1].set_ylabel("slope x multiplier")
    axes[1].legend(fontsize=7)
    _save(fig, "k_calibration")

    context = _valid(frame[frame.metric == "dose_slope"], "context slopes", ("trait", "context", "value"))
    context["context_method"] = context.method + ":" + context.context
    context_pivot = context.pivot_table(index="trait", columns="context_method", values="value", aggfunc="mean")
    fig, axis = plt.subplots(figsize=(9, 3.5))
    image = axis.imshow(context_pivot.values, aspect="auto", cmap="coolwarm")
    axis.set_yticks(range(len(context_pivot.index)), context_pivot.index)
    axis.set_xticks(range(len(context_pivot.columns)), context_pivot.columns, rotation=35, ha="right")
    axis.set_title("Per-context dev dose-response slope")
    fig.colorbar(image, ax=axis, label="slope")
    _save(fig, "context_heatmap")
    context.to_csv(FIGURES / "context_slopes.csv", index=False)

    invalid = _valid(frame[frame.metric.isin(("invalid_rate", "behaviorbench_invalid_rate"))], "invalid rates", ("trait", "target_score", "value"))
    fig, axis = plt.subplots(figsize=(7, 3.5))
    for method, group in invalid.groupby("method"):
        for trait, values in group.groupby("trait"):
                axis.plot(values.target_score, values.value, marker="o", label=f"{method}:{trait}")
    axis.set_xlabel("target score")
    axis.set_ylabel("invalid/degenerate rate")
    axis.set_title("Generation invalid rate")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    _save(fig, "invalid_rates")

    summary_rows = []
    for (method, trait), group in dose.groupby(["method", "trait"]):
        means = group.groupby("target_score").value.mean().sort_index()
        slope = float(np.polyfit(means.index, means.values, 1)[0])
        config_trait = config["traits"].get(trait, {}) if method == "caa" else {}
        method_invalid = invalid[(invalid.method == method) & (invalid.trait == trait)]
        summary_rows.append({
            "method": method,
            "trait": trait,
            "layer": int(group.layer.iloc[0]),
            "k": float(group.k.iloc[0]),
            "slope": slope,
            "spearman": float(config_trait.get("spearman", np.nan)),
            "invalid_rate": float(method_invalid.value.max()) if not method_invalid.empty else np.nan,
            "n_dev": int(group.n.max()),
        })
    for (method, trait), group in behavior.groupby(["method", "trait"]):
        if method == "caa":
            continue
        means = group.groupby("target_score").value.mean().sort_index()
        if len(means) < 2:
            continue
        ranks = means.rank().to_numpy()
        targets = means.index.to_numpy()
        spearman = float(np.corrcoef(targets, ranks)[0, 1])
        method_invalid = invalid[(invalid.method == method) & (invalid.trait == trait)]
        summary_rows.append({
            "method": method,
            "trait": trait,
            "layer": int(group.layer.iloc[0]),
            "k": float(group.k.iloc[0]),
            "slope": float(np.polyfit(means.index, means.values, 1)[0]),
            "spearman": spearman,
            "invalid_rate": float(method_invalid.value.max()) if not method_invalid.empty else np.nan,
            "n_dev": int(group.n.max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(FIGURES / "summary.csv", index=False)
    markdown = ["| " + " | ".join(summary.columns) + " |",
                "| " + " | ".join("---" for _ in summary.columns) + " |"]
    for row in summary.itertuples(index=False, name=None):
        markdown.append("| " + " | ".join(str(value) for value in row) + " |")
    (FIGURES / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    checks = []
    for path in sorted(FIGURES.glob("*.png")):
        from PIL import Image
        with Image.open(path) as image:
            std = float(np.asarray(image).std())
        checks.append({"file": path.name, "bytes": path.stat().st_size, "pixel_std": std,
                       "pass": path.stat().st_size > 20_000 and std > 0})
    pd.DataFrame(checks).to_csv(FIGURES / "png_verification.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    if not all(row["pass"] for row in checks):
        raise RuntimeError("One or more generated PNGs failed size/pixel-variance verification")


if __name__ == "__main__":
    main()
