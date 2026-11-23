"""Cross-context consistency for post-generation rewriting.

No new generation: joins the single-turn records with the context labels in
data/behaviorbench/adapted/data/all_items_labeled.json and breaks results down by
trait x context.

    python3 scripts/rewriting_context.py                      # free, development + test
    python3 scripts/rewriting_context.py --condition no_draft --split test

Writes artifacts/rewriting/context/{context_results.csv, context_summary.csv, context_heatmap.png}.

Rules: contexts are compared within a trait only (contexts are unevenly spread across traits),
and cells with fewer than --min-items items are reported but greyed out. Neuroticism context is
confounded with the heuristic family (development: availability, labeled no_social; test:
representativeness, mostly public_community).
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "behaviorbench" / "adapted" / "data" / "all_items_labeled.json"
RECORDS = ROOT / "artifacts" / "rewriting"
OUT = RECORDS / "context"
TRAITS = ("O", "C", "E", "A", "N")
TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
               "A": "Agreeableness", "N": "Neuroticism"}
CONTEXTS = ("friends_family", "public_community", "work_school", "home_personal", "no_social")
LEVELS = (1, 2, 3, 4, 5)
SEED = 20261026


def load_labels(path=LABELS):
    """Item id -> list of context labels."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["id"]: list(item.get("context_labels") or []) for item in items}


def default_record_files():
    """Final single-turn runs: dev_X.jsonl and test_X.jsonl (dev_O_v1 is the superseded O run)."""
    return [RECORDS / f"{split}_{trait}.jsonl" for split in ("dev", "test") for trait in TRAITS
            if (RECORDS / f"{split}_{trait}.jsonl").exists()]


def load_records(paths):
    return [json.loads(line) for path in paths
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def item_scores(records, labels, condition, splits):
    """{(trait, context): {item_id: {level: [scores]}}} over valid, labeled records."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        if record.get("kind") == "bfi" or record["condition"] != condition:
            continue
        if splits and record.get("split") not in splits:
            continue
        if record["score"] is None or record.get("level") not in LEVELS:
            continue
        for context in labels.get(record["item_id"], []):
            cells[(record["item_trait"], context)][record["item_id"]][record["level"]].append(record["score"])
    return cells


def _slope(points):
    """Least-squares slope of endorsement on level, from (level, rate) pairs."""
    xs = [x for x, _ in points]
    if len(set(xs)) < 2:
        return None
    mx, my = mean(xs), mean(y for _, y in points)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def cell_stats(items):
    """Endorsement per level, gap E(5) - E(1) and slope for one trait x context cell."""
    rates = {}
    for level in LEVELS:
        values = [s for levels in items.values() for s in levels.get(level, [])]
        rates[level] = mean(values) if values else None
    points = [(level, rate) for level, rate in rates.items() if rate is not None]
    gap = None if rates[1] is None or rates[5] is None else rates[5] - rates[1]
    return rates, gap, _slope(points)


def bootstrap(items, resamples=1000, seed=SEED):
    """95% intervals for each level's rate, the gap and the slope, resampling items."""
    ids = sorted(items)
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(resamples):
        sample = {n: items[item] for n, item in enumerate(rng.choices(ids, k=len(ids)))}
        rates, gap, slope = cell_stats(sample)
        for level, rate in rates.items():
            if rate is not None:
                draws[level].append(rate)
        if gap is not None:
            draws["gap"].append(gap)
        if slope is not None:
            draws["slope"].append(slope)

    def interval(values):
        if not values:
            return None, None
        values = sorted(values)
        return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]
    return {key: interval(values) for key, values in draws.items()}


def analyze(records, labels, condition="free", splits=("development", "test"), min_items=5):
    """Long rows (one per metric) and one summary row per trait x context."""
    split_name = "+".join(splits) if splits else "all"
    rows, summary = [], []
    cells = item_scores(records, labels, condition, set(splits) if splits else None)
    for trait in TRAITS:
        for context in CONTEXTS:
            items = cells.get((trait, context))
            if not items:
                continue
            rates, gap, slope = cell_stats(items)
            intervals = bootstrap(items)
            base = {"method": "rewriting", "condition": condition, "trait": TRAIT_NAMES[trait],
                    "context": context, "split": split_name, "n_items": len(items), "seed": SEED}
            for level in LEVELS:
                low, high = intervals.get(level, (None, None))
                rows.append({**base, "target_score": level, "metric": "endorsement_rate",
                             "value": rates[level], "ci_low": low, "ci_high": high})
            for metric, value in (("gap_t5_t1", gap), ("dose_slope", slope)):
                low, high = intervals.get("gap" if metric == "gap_t5_t1" else "slope", (None, None))
                rows.append({**base, "target_score": "", "metric": metric, "value": value,
                             "ci_low": low, "ci_high": high})
            summary.append({"trait": TRAIT_NAMES[trait], "context": context, "condition": condition,
                            "split": split_name, "n_items": len(items), "e1": rates[1], "e5": rates[5],
                            "gap": gap, "slope": slope,
                            "slope_ci_low": intervals.get("slope", (None, None))[0],
                            "slope_ci_high": intervals.get("slope", (None, None))[1],
                            "reportable": len(items) >= min_items})
    return rows, summary


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def heatmap(summary, path, min_items, title):
    """Slope by context (rows) and trait (columns); small cells greyed, n shown in each cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(row["context"], row["trait"]): row for row in summary}
    grid = np.full((len(CONTEXTS), len(TRAITS)), np.nan)
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for i, context in enumerate(CONTEXTS):
        for j, trait in enumerate(TRAITS):
            row = lookup.get((context, TRAIT_NAMES[trait]))
            if row is None:
                continue
            small = not row["reportable"] or row["slope"] is None
            if not small:
                grid[i, j] = row["slope"]
            label = "-" if row["slope"] is None else f"{row['slope']:.2f}"
            axis.text(j, i, f"{label}\nn={row['n_items']}", ha="center", va="center", fontsize=7.5,
                      color="#999999" if small else ("white" if row["slope"] > 0.15 else "black"))
    limit = max(0.2, float(np.nanmax(np.abs(grid)))) if not np.all(np.isnan(grid)) else 0.2
    image = axis.imshow(np.ma.masked_invalid(grid), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_facecolor("#f2f2f2")
    axis.set_xticks(range(len(TRAITS)), [TRAIT_NAMES[t].lower() for t in TRAITS], rotation=20, ha="right")
    axis.set_yticks(range(len(CONTEXTS)), [c.replace("_", " ") for c in CONTEXTS])
    axis.set_title(title, fontsize=10)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="dose-response slope")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", nargs="*", type=Path, help="JSONL files (default: final dev and test runs)")
    parser.add_argument("--condition", default="free", choices=("free", "no_draft", "locked"))
    parser.add_argument("--split", default="both", choices=("both", "development", "test"))
    parser.add_argument("--min-items", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    splits = ("development", "test") if args.split == "both" else (args.split,)
    records = load_records(args.records or default_record_files())
    rows, summary = analyze(records, load_labels(), args.condition, splits, args.min_items)
    if not rows:
        sys.exit("no labeled records matched")
    suffix = "" if (args.condition, args.split) == ("free", "both") else f"_{args.condition}_{args.split}"
    write_csv(rows, args.out / f"context_results{suffix}.csv")
    write_csv(summary, args.out / f"context_summary{suffix}.csv")
    split_text = "development + test" if args.split == "both" else args.split
    heatmap(summary, args.out / f"context_heatmap{suffix}.png", args.min_items,
            f"Rewriting ({args.condition}) slope by context and trait ({split_text})")

    print(f"{'trait':17} {'context':17} {'n':>3} {'E1':>5} {'E5':>5} {'gap':>6} {'slope':>6}  95% CI slope")
    for row in summary:
        fmt = lambda v: "   -" if v is None else f"{v:.2f}"
        flag = "" if row["reportable"] else "  (small)"
        print(f"{row['trait']:17} {row['context']:17} {row['n_items']:>3} {fmt(row['e1']):>5} {fmt(row['e5']):>5} "
              f"{fmt(row['gap']):>6} {fmt(row['slope']):>6}  [{fmt(row['slope_ci_low'])}, {fmt(row['slope_ci_high'])}]{flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
