"""Plot shared CAA/BiPO steering and BehaviorBench results."""

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
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
COLORS = {"default": "black", "caa": "#0072B2", "bipo": "#D55E00"}


def _read_results() -> pd.DataFrame:
    sources = [RESULTS]
    bipo_reference = ROOT / "artifacts" / "steering" / "bipo" / "results_bipo_ref.csv"
    if bipo_reference.exists():
        sources.append(bipo_reference)
    else:
        print(f"Warning: BiPO reference results not found at {bipo_reference}")
    frame = pd.concat([pd.read_csv(path) for path in sources], ignore_index=True)
    required = {"method", "trait", "layer", "k", "target_score", "metric", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"results.csv missing columns: {sorted(missing)}")
    for column in ("layer", "k", "target_score", "value", "ci_low", "ci_high", "n"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _baseline() -> pd.DataFrame:
    path = ROOT / "artifacts" / "eval" / "default_behaviorbench.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame.rename(columns={"endorsement_rate": "value"})


def _selected_k(frame: pd.DataFrame, method: str, trait: str) -> float | None:
    summary_path = FIGURES / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if {"method", "trait", "k"} <= set(summary.columns):
            match = summary[
                (summary["method"].str.lower() == method)
                & (summary["trait"].str.lower() == trait)
            ]
            if not match.empty and pd.notna(match.iloc[-1]["k"]):
                return float(match.iloc[-1]["k"])
    match = frame[
        (frame["method"].str.lower() == method)
        & (frame["trait"].str.lower() == trait)
        & frame["k"].notna()
    ]
    return float(match.iloc[-1]["k"]) if not match.empty else None


def _selected_behaviorbench_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        frame["metric"].eq("behaviorbench_endorsement")
        & frame["context"].eq("test")
        & frame["method"].str.lower().isin(("bipo", "caa"))
    ].copy()
    selected["method"] = selected["method"].str.lower()
    selected["trait"] = selected["trait"].str.lower()
    selected["_row_order"] = np.arange(len(selected))
    selected_k = {
        (method, trait): _selected_k(frame, method, trait)
        for method in ("bipo", "caa")
        for trait in TRAITS
    }
    selected = selected[
        selected.apply(
            lambda row: selected_k[(row["method"], row["trait"])] == row["k"],
            axis=1,
        )
    ]
    return (
        selected.sort_values("_row_order")
        .drop_duplicates(["method", "trait", "target_score"], keep="last")
        .drop(columns="_row_order")
    )


def _dose_response_compare(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_behaviorbench_rows(frame)
    baseline = _baseline()
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(17, 3.6), sharey=True, squeeze=False
    )
    table_rows = []
    for axis, trait in zip(axes[0], TRAITS):
        for method in ("caa", "bipo"):
            group = selected[
                (selected["method"] == method) & (selected["trait"] == trait)
            ].sort_values("target_score")
            if group.empty:
                continue
            color = COLORS[method]
            axis.plot(
                group["target_score"], group["value"], "o-",
                color=color, label=method.upper(),
            )
            ci = group.dropna(subset=["ci_low", "ci_high"])
            if not ci.empty:
                axis.fill_between(
                    ci["target_score"], ci["ci_low"], ci["ci_high"],
                    color=color, alpha=0.16,
                )
            slope = float(np.polyfit(group["target_score"], group["value"], 1)[0])
            axis.text(
                0.04, 0.08 if method == "caa" else 0.18,
                f"{method.upper()} slope={slope:.3f}",
                transform=axis.transAxes, color=color, fontsize=8,
            )
            at_one = group[group["target_score"] == 1]["value"]
            at_five = group[group["target_score"] == 5]["value"]
            table_rows.append({
                "method": method,
                "trait": trait,
                "k": _selected_k(frame, method, trait),
                "endorsement_t1": float(at_one.iloc[-1]) if not at_one.empty else np.nan,
                "endorsement_t5": float(at_five.iloc[-1]) if not at_five.empty else np.nan,
                "slope": slope,
            })
        if not baseline.empty:
            default = baseline[baseline["trait"].str.lower() == trait].sort_values("target_score")
            if not default.empty:
                axis.plot(
                    default["target_score"], default["value"],
                    "--", color="#777777", linewidth=1, label="default",
                )
        axis.set_title(f"{trait} (exploratory)" if trait == "neuroticism" else trait)
        axis.set_xlabel("target score")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("BehaviorBench endorsement")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    output = FIGURES / "dose_response_compare.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(table_rows)


def _behaviorbench_plots(frame: pd.DataFrame) -> pd.DataFrame:
    endorsement = frame[frame.metric == "behaviorbench_endorsement"].copy()
    invalid = frame[frame.metric == "behaviorbench_invalid_rate"].copy()
    if endorsement.empty:
        raise ValueError("No behaviorbench_endorsement rows found")
    methods = [method for method in ("default", "bipo", "caa") if method in endorsement.method.unique()]
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(4 * len(TRAITS), 3.5), squeeze=False)
    baseline = _baseline()
    for axis, trait in zip(axes[0], TRAITS):
        for method in methods:
            group = endorsement[(endorsement.method == method) & (endorsement.trait.str.lower() == trait)]
            group = group.sort_values("target_score")
            if group.empty:
                continue
            color = COLORS.get(method, None)
            axis.plot(group.target_score, group.value, marker="o", color=color, label=method)
            if {"ci_low", "ci_high"} <= set(group.columns):
                ci = group.dropna(subset=["ci_low", "ci_high"])
                if not ci.empty:
                    axis.fill_between(ci.target_score, ci.ci_low, ci.ci_high, color=color, alpha=0.15)
        default = baseline[baseline.trait.str.lower() == trait] if not baseline.empty else pd.DataFrame()
        if not default.empty:
            axis.plot(default.target_score, default.value, "k--", label="default")
        axis.set_title(trait)
        axis.set_xlabel("target score t")
        axis.set_ylabel("E(tau, t) endorsement")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2)
    axes[0][0].legend(fontsize=8)
    _save(fig, "behaviorbench_dose_response")
    _save(plt.figure(), "dose_response") if False else None

    fig, axis = plt.subplots(figsize=(8, 4))
    for method in methods:
        group = invalid[invalid.method == method]
        for trait in TRAITS:
            line = group[group.trait.str.lower() == trait].sort_values("target_score")
            if not line.empty:
                axis.plot(line.target_score, line.value, marker="o", label=f"{method}:{trait}")
    axis.set_xlabel("target score t")
    axis.set_ylabel("invalid rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    _save(fig, "invalid_rates")

    rows = []
    for (method, trait), group in endorsement.groupby(["method", endorsement.trait.str.lower()]):
        group = group.sort_values("target_score")
        if group.empty:
            continue
        means = group.groupby("target_score").value.mean()
        slope = float(np.polyfit(means.index, means.values, 1)[0]) if len(means) > 1 else float("nan")
        invalid_group = invalid[(invalid.method == method) & (invalid.trait.str.lower() == trait)]
        rows.append({
            "method": method,
            "trait": trait,
            "layer": int(group.layer.dropna().iloc[0]) if group.layer.notna().any() else "",
            "k": float(group.k.dropna().iloc[0]) if group.k.notna().any() else "",
            "dose_slope": slope,
            "spearman": "",
            "invalid_rate": float(invalid_group.value.max()) if not invalid_group.empty else "",
            "n": int(group.n.max()) if group.n.notna().any() else "",
        })
    summary = pd.DataFrame(rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FIGURES / "summary.csv", index=False)
    markdown = ["| " + " | ".join(summary.columns) + " |",
                "| " + " | ".join("---" for _ in summary.columns) + " |"]
    markdown.extend("| " + " | ".join(str(value) for value in row) + " |"
                    for row in summary.itertuples(index=False, name=None))
    (FIGURES / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def _k_calibration_plot() -> None:
    files = [
        ("caa", ROOT / "artifacts" / "steering" / "caa" / "behaviorbench_k_calibration.csv"),
        ("bipo", ROOT / "artifacts" / "steering" / "bipo" / "behaviorbench_k_calibration.csv"),
    ]
    frames = []
    for method, path in files:
        if path.exists():
            frame = pd.read_csv(path)
            frame["method"] = method
            frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (method, trait), group in data.groupby(["method", "trait"]):
        color = COLORS.get(method)
        axes[0].plot(group.k, group.slope, marker="o", color=color, alpha=0.8, label=f"{method}:{trait}")
        axes[1].plot(group.k, group.invalid_rate, marker="o", color=color, alpha=0.8)
    axes[0].set_title("Development calibration slope")
    axes[1].set_title("Development invalid rate")
    for axis in axes:
        axis.set_xlabel("k")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("slope")
    axes[1].set_ylabel("invalid rate")
    axes[0].legend(fontsize=7, ncol=2)
    _save(fig, "k_calibration")
    data.to_csv(FIGURES / "k_calibration.csv", index=False)


def _k_calibration_compare(frame: pd.DataFrame) -> None:
    files = [
        ("caa", ROOT / "artifacts" / "steering" / "caa" / "behaviorbench_k_calibration.csv"),
        ("bipo", ROOT / "artifacts" / "steering" / "bipo" / "behaviorbench_k_calibration.csv"),
    ]
    frames = []
    for method, path in files:
        if path.exists():
            values = pd.read_csv(path)
            values["method"] = method
            values["trait"] = values["trait"].str.lower()
            frames.append(values)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    finite_k = data.loc[data["k"] > 0, "k"]
    x_limits = (float(finite_k.min()) * 0.8, float(finite_k.max()) * 1.25)
    fig, axes = plt.subplots(
        2, len(TRAITS), figsize=(17, 6), sharex="col", squeeze=False
    )
    for column, trait in enumerate(TRAITS):
        for method in ("caa", "bipo"):
            group = data[(data["method"] == method) & (data["trait"] == trait)].sort_values("k")
            if group.empty:
                continue
            color = COLORS[method]
            axes[0, column].plot(group["k"], group["slope"], "o-", color=color, label=method.upper())
            axes[1, column].plot(group["k"], group["invalid_rate"], "o-", color=color, label=method.upper())
            selected = _selected_k(frame, method, trait)
            if selected is not None:
                axes[0, column].axvline(selected, color=color, linestyle=":", linewidth=1)
                axes[1, column].axvline(selected, color=color, linestyle=":", linewidth=1)
        axes[0, column].set_title(f"{trait} (exploratory)" if trait == "neuroticism" else trait)
        axes[0, column].set_xscale("log")
        axes[1, column].set_xscale("log")
        axes[0, column].set_xlim(x_limits)
        axes[1, column].set_xlim(x_limits)
        axes[1, column].set_xlabel("k (log scale)")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("development slope")
    axes[1, 0].set_ylabel("development invalid rate")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    output = FIGURES / "k_calibration_compare.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _comparison(summary: pd.DataFrame, frame: pd.DataFrame) -> None:
    lines = ["# CAA vs BiPO", "", "| trait | method | E(1) | E(5) | gap | default E(1) | default E(5) | dose slope | invalid rate | k | layer | t5-t1 CIs separate |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    endorsement = frame[frame.metric == "behaviorbench_endorsement"]
    baseline = _baseline()
    for trait in TRAITS:
        for method in ("caa", "bipo"):
            group = endorsement[(endorsement.method == method) & (endorsement.trait.str.lower() == trait)]
            if group.empty:
                continue
            at1, at5 = group[group.target_score == 1], group[group.target_score == 5]
            e1, e5 = float(at1.value.mean()), float(at5.value.mean())
            separate = False
            if not at1[["ci_low", "ci_high"]].isna().any(axis=None) and not at5[["ci_low", "ci_high"]].isna().any(axis=None):
                separate = float(at5.ci_low.mean()) > float(at1.ci_high.mean()) or float(at1.ci_low.mean()) > float(at5.ci_high.mean())
            row = summary[(summary.method == method) & (summary.trait == trait)].iloc[0]
            default = baseline[baseline.trait.str.lower() == trait] if not baseline.empty else pd.DataFrame()
            default_1 = default[default.target_score == 1].value.mean() if not default.empty else float("nan")
            default_5 = default[default.target_score == 5].value.mean() if not default.empty else float("nan")
            lines.append(f"| {trait} | {method} | {e1:.3f} | {e5:.3f} | {e5-e1:.3f} | {default_1:.3f} | {default_5:.3f} | {row.dose_slope:.3f} | {row.invalid_rate} | {row.k} | {row.layer} | {'yes' if separate else 'no'} |")
    lines += ["", "## Caveats", "", "- CAA uses per-trait layers 14/16; BiPO uses fixed layer 15.",
              "- CAA uses 4-bit NF4; BiPO used an unquantized fp16 fallback.",
              "- CAA vectors are unit norm; BiPO vectors may be raw, so k scales are not directly equal.",
              "- CAA uses k grid (1, 2, 4, 6, 8, 12, 16, 24); BiPO uses its notebook grid.",
              "- Neuroticism is an exploratory 50/50 heuristic-choice proxy.",
              "- Bootstrap-CI separation is descriptive and is not a winner claim."]
    (FIGURES / "caa_vs_bipo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = _read_results()
    summary = _behaviorbench_plots(frame)
    _k_calibration_plot()
    dose_table = _dose_response_compare(frame)
    _k_calibration_compare(frame)
    _comparison(summary, frame)
    print(summary.to_string(index=False))
    print("\nDose-response comparison:")
    print(dose_table.to_string(index=False))
    print(f"\nSaved {FIGURES / 'dose_response_compare.png'}")
    print(f"Saved {FIGURES / 'k_calibration_compare.png'}")
    print(f"Default baseline lookup: {ROOT / 'artifacts' / 'eval' / 'default_behaviorbench.csv'}")


if __name__ == "__main__":
    main()
