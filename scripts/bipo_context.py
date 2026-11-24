"""Cross-context consistency analysis for BiPO BehaviorBench results.

No new generation: analyzes the item-level CSV produced by
``notebooks/evaluate_bipo_behaviorbench.ipynb`` using the existing BiPO vectors.
Results are broken down by trait x context within the held-out test set.

    python3 scripts/bipo_context.py
    python3 scripts/bipo_context.py path/to/behaviorbench_items.csv --min-items 10

Writes context_results.csv, context_summary.csv, and context_heatmap.png under
artifacts/steering/bipo/context by default. Contexts are compared within each
trait because their distribution differs across traits; cells below --min-items
are reported but greyed out.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "behaviorbench" / "adapted" / "data" / "all_items_labeled.json"
RECORDS = ROOT / "artifacts" / "steering" / "bipo" / "behaviorbench_items.csv"
OUT = ROOT / "artifacts" / "steering" / "bipo" / "context"
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
TRAIT_NAMES = {trait: trait.title() for trait in TRAITS}
CONTEXTS = ("friends_family", "public_community", "work_school", "home_personal", "no_social")
LEVELS = (1, 2, 3, 4, 5)
SEED = 20261026


def load_labels(path=LABELS):
    """Map item IDs to their reviewed context labels."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["id"]: list(item.get("context_labels") or []) for item in items}


def load_records(paths):
    """Load item-level result rows from one or more BiPO evaluation CSVs."""
    records = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            records.extend(csv.DictReader(handle))
    return records


def _valid_score(record):
    valid = record.get("valid")
    if valid is not None and str(valid).strip().lower() not in {"true", "1", "yes"}:
        return None
    try:
        score = float(record["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def item_scores(records, labels):
    """Return {(trait, context): {item_id: {level: [scores]}}} for valid labeled rows."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        item_id = record.get("item_id")
        trait = str(record.get("trait", "")).lower()
        try:
            level = int(record.get("target_score"))
        except (TypeError, ValueError):
            continue
        score = _valid_score(record)
        if not item_id or trait not in TRAITS or level not in LEVELS or score is None:
            continue
        for context in labels.get(item_id, []):
            if context in CONTEXTS:
                cells[(trait, context)][item_id][level].append(score)
    return cells


def _slope(points):
    """Least-squares slope of endorsement on target score."""
    xs = [x for x, _ in points]
    if len(set(xs)) < 2:
        return None
    mx, my = mean(xs), mean(y for _, y in points)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def cell_stats(items):
    """Endorsement per level, the level-5 minus level-1 gap, and dose slope."""
    rates = {}
    for level in LEVELS:
        values = [score for levels in items.values() for score in levels.get(level, [])]
        rates[level] = mean(values) if values else None
    points = [(level, rate) for level, rate in rates.items() if rate is not None]
    gap = None if rates[1] is None or rates[5] is None else rates[5] - rates[1]
    return rates, gap, _slope(points)


def bootstrap(items, resamples=1000, seed=SEED):
    """Calculate percentile intervals by resampling items with all their levels."""
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


def analyze(records, labels, min_items=5):
    """Return long-form metric rows and a summary row per trait x context."""
    rows, summary = [], []
    cells = item_scores(records, labels)
    for trait in TRAITS:
        for context in CONTEXTS:
            items = cells.get((trait, context))
            if not items:
                continue
            rates, gap, slope = cell_stats(items)
            intervals = bootstrap(items)
            base = {
                "method": "bipo",
                "trait": TRAIT_NAMES[trait],
                "context": context,
                "split": "test",
                "n_items": len(items),
                "seed": SEED,
            }
            for level in LEVELS:
                low, high = intervals.get(level, (None, None))
                rows.append(
                    {
                        **base,
                        "target_score": level,
                        "metric": "endorsement_rate",
                        "value": rates[level],
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            for metric, value, interval_key in (
                ("gap_t5_t1", gap, "gap"),
                ("dose_slope", slope, "slope"),
            ):
                low, high = intervals.get(interval_key, (None, None))
                rows.append(
                    {
                        **base,
                        "target_score": "",
                        "metric": metric,
                        "value": value,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            summary.append(
                {
                    "trait": TRAIT_NAMES[trait],
                    "context": context,
                    "split": "test",
                    "n_items": len(items),
                    "e1": rates[1],
                    "e5": rates[5],
                    "gap": gap,
                    "slope": slope,
                    "slope_ci_low": intervals.get("slope", (None, None))[0],
                    "slope_ci_high": intervals.get("slope", (None, None))[1],
                    "reportable": len(items) >= min_items,
                }
            )
    return rows, summary


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def heatmap(summary, path, title):
    """Plot slope by context and trait; grey out small cells and show item counts."""
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
            axis.text(
                j,
                i,
                f"{label}\nn={row['n_items']}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#999999" if small else ("white" if row["slope"] > 0.15 else "black"),
            )
    limit = max(0.2, float(np.nanmax(np.abs(grid)))) if not np.all(np.isnan(grid)) else 0.2
    image = axis.imshow(np.ma.masked_invalid(grid), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_facecolor("#f2f2f2")
    axis.set_xticks(range(len(TRAITS)), [TRAIT_NAMES[t].lower() for t in TRAITS], rotation=20, ha="right")
    axis.set_yticks(range(len(CONTEXTS)), [context.replace("_", " ") for context in CONTEXTS])
    axis.set_title(title, fontsize=10)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="dose-response slope")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", nargs="*", type=Path, help="BiPO item-level CSVs (default: notebook output)")
    parser.add_argument("--min-items", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    records = load_records(args.records or [RECORDS])
    rows, summary = analyze(records, load_labels(), args.min_items)
    if not rows:
        sys.exit("no valid labeled BiPO records matched")
    write_csv(rows, args.out / "context_results.csv")
    write_csv(summary, args.out / "context_summary.csv")
    heatmap(summary, args.out / "context_heatmap.png", "BiPO dose-response slope by context and trait (test)")

    print(f"{'trait':17} {'context':17} {'n':>3} {'E1':>5} {'E5':>5} {'gap':>6} {'slope':>6}  95% CI slope")
    for row in summary:
        fmt = lambda value: "   -" if value is None else f"{value:.2f}"
        flag = "" if row["reportable"] else "  (small)"
        print(
            f"{row['trait']:17} {row['context']:17} {row['n_items']:>3} "
            f"{fmt(row['e1']):>5} {fmt(row['e5']):>5} {fmt(row['gap']):>6} "
            f"{fmt(row['slope']):>6}  [{fmt(row['slope_ci_low'])}, {fmt(row['slope_ci_high'])}]{flag}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
