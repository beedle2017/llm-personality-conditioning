"""Plot post-generation rewriting results, in the style of plot_steering_results.py.

python3 scripts/plot_rewriting_results.py            # test split (default)
python3 scripts/plot_rewriting_results.py --split development

Reads artifacts/rewriting/results.csv (written by rewriting_report.py --results-csv) and the
run JSONL files for preservation metrics. Writes to artifacts/rewriting/figures/.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "rewriting" / "results.csv"
FIGURES = ROOT / "artifacts" / "rewriting" / "figures"
TRAITS = ("Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism")
LEVELS = (1, 2, 3, 4, 5)
COLORS = {"free": "#009E73", "no_draft": "#0072B2", "locked": "#CC79A7"}
LABELS = {"free": "free rewrite", "no_draft": "no draft", "locked": "locked (tone only)"}


def _num(value):
    return float(value) if value not in ("", None) else np.nan


def read_results(split):
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == split]
    if not rows:
        raise SystemExit(f"no {split} rows in {RESULTS}")
    return rows


def pick(rows, trait, condition):
    """Rows for one trait and condition, ordered by level (levels are '' for draft/paraphrase)."""
    group = [r for r in rows if r["trait"] == trait and r["condition"] == condition]
    return sorted(group, key=lambda r: int(r["target_score"] or 0))


def curve(rows, trait, condition):
    group = pick(rows, trait, condition)
    x = np.array([int(r["target_score"]) for r in group])
    y = np.array([_num(r["endorsement_rate"]) for r in group])
    low = np.array([_num(r["ci_low"]) for r in group])
    high = np.array([_num(r["ci_high"]) for r in group])
    return x, y, low, high


def slope(x, y):
    keep = ~np.isnan(y)
    return float(np.polyfit(x[keep], y[keep], 1)[0]) if keep.sum() > 1 else np.nan


def dose_response(rows, split):
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(17, 3.6), sharey=True, squeeze=False)
    summary = []
    for axis, trait in zip(axes[0], TRAITS):
        draft = pick(rows, trait, "draft")[0]
        paraphrase = pick(rows, trait, "paraphrase")[0]
        axis.axhline(_num(draft["endorsement_rate"]), color="black", linestyle="--", linewidth=1,
                     label="draft (unconditioned)")
        axis.axhline(_num(paraphrase["endorsement_rate"]), color="#777777", linestyle=":", linewidth=1.2,
                     label="paraphrase (no target)")
        for position, condition in enumerate(("free", "no_draft", "locked")):
            x, y, low, high = curve(rows, trait, condition)
            color = COLORS[condition]
            axis.plot(x, y, "o-", color=color, label=LABELS[condition], markersize=4)
            axis.fill_between(x, low, high, color=color, alpha=0.14)
            value = slope(x, y)
            axis.text(0.96, 0.06 + 0.09 * position, f"{condition} slope={value:.3f}", ha="right",
                      transform=axis.transAxes, color=color, fontsize=7.5,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
            summary.append({"split": split, "trait": trait, "condition": condition,
                            "draft": _num(draft["endorsement_rate"]),
                            "endorsement_t1": y[0], "endorsement_t5": y[-1], "slope": value,
                            "ci_separate": bool(low[-1] > high[0] or low[0] > high[-1]),
                            "max_invalid": max(_num(r["invalid_rate"]) for r in pick(rows, trait, condition))})
        axis.set_title(f"{trait.lower()} (exploratory)" if trait == "Neuroticism" else trait.lower())
        axis.set_xlabel("target score")
        axis.set_xticks(LEVELS)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("BehaviorBench endorsement")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, f"dose_response_{split}")
    return summary


def flip_heatmap(rows, split):
    """Share of decisions that differ from the item's own draft, averaged over levels 1-5."""
    conditions = ("paraphrase", "locked", "free", "no_draft")
    grid = np.zeros((len(TRAITS), len(conditions)))
    for i, trait in enumerate(TRAITS):
        for j, condition in enumerate(conditions):
            group = pick(rows, trait, condition)
            grid[i, j] = mean(_num(r["flip_up"]) + _num(r["flip_down"]) for r in group)
    fig, axis = plt.subplots(figsize=(5.6, 3.6))
    image = axis.imshow(grid, cmap="Greens", vmin=0, vmax=max(0.5, grid.max()))
    for i in range(len(TRAITS)):
        for j in range(len(conditions)):
            axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9,
                      color="white" if grid[i, j] > 0.3 else "black")
    axis.set_xticks(range(len(conditions)), ["paraphrase", "locked", "free", "no draft"],
                  rotation=20, ha="right")
    axis.set_yticks(range(len(TRAITS)), [t.lower() for t in TRAITS])
    axis.set_title("Decision changed vs. draft (mean over levels)", fontsize=10)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="flip rate")
    fig.tight_layout()
    save(fig, f"flip_heatmap_{split}")
    return conditions, grid


def invalid_rates(rows, split):
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(17, 2.8), sharey=True, squeeze=False)
    for axis, trait in zip(axes[0], TRAITS):
        for condition in ("free", "no_draft", "locked"):
            group = pick(rows, trait, condition)
            axis.plot([int(r["target_score"]) for r in group], [_num(r["invalid_rate"]) for r in group],
                      "o-", color=COLORS[condition], label=LABELS[condition], markersize=4)
        axis.set_title(trait.lower())
        axis.set_xlabel("target score")
        axis.set_xticks(LEVELS)
        axis.set_ylim(-0.02, 0.5)
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("invalid rate")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"invalid_rates_{split}")


def preservation(split):
    """Mean preservation metrics per condition from the run files of one split."""
    prefix = "test" if split == "test" else "dev"
    records = []
    for trait in "OCEAN":
        path = ROOT / "artifacts" / "rewriting" / f"{prefix}_{trait}.jsonl"
        if path.exists():
            records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    table = []
    for condition in ("paraphrase", "locked", "free"):
        group = [r for r in records if r["condition"] == condition and "preservation" in r]
        row = {"condition": condition, "n": len(group),
               "refusal_rate": mean(r["refusal"] for r in group) if group else np.nan}
        for key in ("length_ratio", "word_jaccard", "numbers_kept"):
            values = [r["preservation"][key] for r in group if r["preservation"][key] is not None]
            row[key] = mean(values) if values else np.nan
        table.append(row)
    return table


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("test", "development"))
    split = parser.parse_args().split
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_results(split)
    summary = dose_response(rows, split)
    conditions, grid = flip_heatmap(rows, split)
    invalid_rates(rows, split)
    kept = preservation(split)
    write_csv(summary, FIGURES / f"summary_{split}.csv")
    write_csv(kept, FIGURES / f"preservation_{split}.csv")
    for row in summary:
        print(f"{row['trait']:17} {row['condition']:9} draft={row['draft']:.3f} "
              f"E1={row['endorsement_t1']:.3f} E5={row['endorsement_t5']:.3f} slope={row['slope']:.3f} "
              f"sep={row['ci_separate']} inv={row['max_invalid']:.2f}")
    print("\nflip rate", conditions)
    for trait, values in zip(TRAITS, grid):
        print(f"{trait:17}", " ".join(f"{v:.3f}" for v in values))
    print()
    for row in kept:
        print(row)
    print(f"\nSaved figures to {FIGURES}")


if __name__ == "__main__":
    main()
