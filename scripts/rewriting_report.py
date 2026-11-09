"""Summarize rewriting records and manage the extractor's hand-label check.

Summary:      python3 scripts/rewriting_report.py run.jsonl
Label sample: python3 scripts/rewriting_report.py run.jsonl --label-sample 100 --labels labels.csv
Agreement:    python3 scripts/rewriting_report.py run.jsonl --labels labels.csv
Results CSV:  python3 scripts/rewriting_report.py dev_O.jsonl dev_C.jsonl ... --results-csv results.csv
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
sys.path.insert(0, str(ROOT))

from scoring import pis  # noqa: E402


def rate(records):
    """Return percent endorsement over valid records, or None with none valid."""
    valid = [r["score"] for r in records if r["score"] is not None]
    return 100 * mean(valid) if valid else None


def invalid_rate(records):
    return 100 * mean(r["score"] is None for r in records) if records else None


def flips(records, baseline):
    """Percent of records with valid decisions that differ from the item's draft."""
    pairs = [(baseline.get((r["item_id"], r["seed"])), r["score"]) for r in records]
    pairs = [(old, new) for old, new in pairs if old is not None and new is not None]
    return {"up": 100 * mean(old == 0 and new == 1 for old, new in pairs) if pairs else None,
            "down": 100 * mean(old == 1 and new == 0 for old, new in pairs) if pairs else None}


def fmt(value):
    return "-" if value is None else f"{value:.1f}"


def behavior_summary(records):
    records = [r for r in records if r["kind"] != "bfi"]
    print("trait condition level  n   rate  PIS  invalid flip_up flip_down")
    for trait in sorted({r["item_trait"] for r in records}):
        subset = [r for r in records if r["item_trait"] == trait]
        drafts = [r for r in subset if r["condition"] == "draft"]
        baseline = {(r["item_id"], r["seed"]): r["score"] for r in drafts}
        base_rate = rate(drafts)
        groups = defaultdict(list)
        for record in subset:
            groups[(record["condition"], record["level"] or 0)].append(record)
        for (condition, level), rows in sorted(groups.items()):
            value = rate(rows)
            # t = 3 is analyzed separately, so PIS is undefined there.
            score = None if not level or level == 3 or value is None or base_rate is None \
                else float(pis(value, level, base_rate))
            change = flips(rows, baseline)
            print(f"{trait:5} {condition:10} {level or '-':>2} {len(rows):4} {fmt(value):>6} {fmt(score):>6} "
                  f"{fmt(invalid_rate(rows)):>6} {fmt(change['up']):>6} {fmt(change['down']):>6}")


def bfi_summary(records):
    records = [r for r in records if r["kind"] == "bfi" and r["target_trait"]]
    if not records:
        return
    print("\nBFI: target trait, condition, level, measured score, absolute error, invalid %")
    groups = defaultdict(list)
    for record in records:
        if record["item_trait"] == record["target_trait"]:
            groups[(record["target_trait"], record["condition"], record["level"])].append(record)
    for (trait, condition, level), rows in sorted(groups.items()):
        valid = [r["score"] for r in rows if r["score"] is not None]
        measured = mean(valid) if valid else None
        error = abs(measured - level) if valid else None
        print(f"{trait} {condition:10} {level} {fmt(measured):>5} {fmt(error):>5} {fmt(invalid_rate(rows)):>6}")


def preservation_summary(records):
    print("\ncondition  length_ratio  word_jaccard  numbers_kept  refusal %")
    for condition in sorted({r["condition"] for r in records if "preservation" in r}):
        rows = [r for r in records if r["condition"] == condition]
        values = {key: [r["preservation"][key] for r in rows if r["preservation"][key] is not None]
                  for key in ("length_ratio", "word_jaccard", "numbers_kept")}
        means = [f"{mean(v):.2f}" if v else "-" for v in values.values()]
        print(f"{condition:10} {means[0]:>10} {means[1]:>12} {means[2]:>12} "
              f"{100 * mean(r['refusal'] for r in rows):>9.1f}")


TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
               "A": "Agreeableness", "N": "Neuroticism"}
RESULT_FIELDS = ["method", "condition", "trait", "split", "target_score", "endorsement_rate",
                 "invalid_rate", "ci_low", "ci_high", "n", "valid_n", "pis", "flip_up", "flip_down", "seed"]


def _bootstrap(values, resamples=1000, seed=20261026):
    """Percentile 95% interval of the mean, as in the default-baseline table."""
    if not values:
        return None, None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def write_results_csv(records, path):
    """One row per split, trait, condition and level; rates are fractions (0-1) like artifacts/eval.

    Each split is compared with its own drafts, so development and test can share one file.
    """
    records = [r for r in records if r["kind"] != "bfi"]
    rows = []
    splits = sorted({r.get("split") or "" for r in records}, key=lambda s: (s != "development", s))
    for split, trait in [(s, t) for s in splits for t in "OCEAN"]:
        subset = [r for r in records if r["item_trait"] == trait and (r.get("split") or "") == split]
        if not subset:
            continue
        drafts = [r for r in subset if r["condition"] == "draft"]
        baseline = {(r["item_id"], r["seed"]): r["score"] for r in drafts}
        base_rate = rate(drafts)
        groups = defaultdict(list)
        for record in subset:
            groups[(record["condition"], record["level"] or 0)].append(record)
        for (condition, level), group in sorted(groups.items()):
            valid = [r["score"] for r in group if r["score"] is not None]
            value = rate(group)
            score = None if not level or level == 3 or value is None or base_rate is None \
                else float(pis(value, level, base_rate))
            change = flips(group, baseline) if condition != "draft" else {"up": None, "down": None}
            low, high = _bootstrap(valid)
            rows.append({"method": "rewriting", "condition": condition, "trait": TRAIT_NAMES[trait],
                         "split": split, "target_score": level or "",
                         "endorsement_rate": mean(valid) if valid else "",
                         "invalid_rate": invalid_rate(group) / 100, "ci_low": low, "ci_high": high,
                         "n": len(group), "valid_n": len(valid),
                         "pis": "" if score is None else score / 100,
                         "flip_up": "" if change["up"] is None else change["up"] / 100,
                         "flip_down": "" if change["down"] is None else change["down"] / 100,
                         "seed": group[0]["seed"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def _items():
    """Adapted BehaviorBench items by id, so a label sheet can show each question."""
    items = {}
    for split in ("development", "test"):
        path = ROOT / "data" / "behaviorbench" / "adapted" / "data" / f"{split}.json"
        if path.exists():
            items.update({item["id"]: item for item in json.loads(path.read_text(encoding="utf-8"))})
    return items


def _choices(record):
    options = record.get("displayed_options") or []
    return options if record["kind"] == "forced_choice" and options else ["Yes", "No"]


def _question(record, items):
    """The situation and the decision being labeled, as the extractor saw them."""
    item = items.get(record["item_id"])
    if item is None:
        return ""
    if record["kind"] == "forced_choice":
        return f"{item['scenario']} Choice: {' or '.join(_choices(record))}?"
    return f"{item['scenario']} Did the speaker decide to do this: '{item['behavior']}'?"


def _normalize(label, record):
    """Lowercase, drop end punctuation, and map a choice number to its option text."""
    label = (label or "").strip().rstrip(".!?").strip().casefold()
    choices = _choices(record)
    if label.isdigit() and 1 <= int(label) <= len(choices):
        label = choices[int(label) - 1].casefold()
    return label


LABEL_FIELDS = ["item_id", "item_trait", "seed", "condition", "target_trait", "level",
                "question", "answer_with", "text", "human_label"]


def write_label_sample(records, path, size, seed=0):
    """Write texts without the extractor's output, so hand labels stay independent."""
    rows = [r for r in records if r["kind"] != "bfi" and r["condition"] != "draft"]
    sample = random.Random(seed).sample(rows, min(size, len(rows)))
    items = _items()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, LABEL_FIELDS)
        writer.writeheader()
        for row in sample:
            out = {key: row.get(key) for key in LABEL_FIELDS[:6]}
            out.update(question=_question(row, items), text=row.get("text"),
                       answer_with=" / ".join(f"{i}={c}" for i, c in enumerate(_choices(row), 1)) + " / Unclear")
            writer.writerow(out)
    print(f"wrote {len(sample)} rows to {path}; fill human_label, then rerun with --labels only")


def agreement(records, path):
    lookup = {(r["item_id"], str(r["seed"]), r["condition"], r["target_trait"] or "", str(r["level"] or "")): r
              for r in records}
    by_trait = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["item_id"], row["seed"], row["condition"], row["target_trait"],
                   "" if row["level"] in ("", "None") else row["level"])
            record = lookup.get(key)
            label = _normalize(row["human_label"], record) if record else ""
            if label:
                by_trait[record.get("item_trait", "?")].append(_normalize(record["extracted"], record) == label)
    matches = [m for values in by_trait.values() for m in values]
    if not matches:
        print("no filled labels found")
        return
    print(f"extractor agreement: {100 * mean(matches):.1f}% over {len(matches)} labels")
    for trait in sorted(by_trait):
        values = by_trait[trait]
        print(f"  {trait}: {100 * mean(values):.1f}% over {len(values)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path)
    parser.add_argument("--label-sample", type=int)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--results-csv", type=Path, help="write the summary as a CSV table")
    parser.add_argument("more_records", type=Path, nargs="*", help="further JSONL files to combine")
    args = parser.parse_args()
    records = [json.loads(line) for path in [args.records, *args.more_records]
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.results_csv:
        write_results_csv(records, args.results_csv)
    elif args.label_sample:
        write_label_sample(records, args.labels or Path("labels.csv"), args.label_sample)
    elif args.labels:
        agreement(records, args.labels)
    else:
        behavior_summary(records)
        bfi_summary(records)
        preservation_summary(records)


if __name__ == "__main__":
    main()
