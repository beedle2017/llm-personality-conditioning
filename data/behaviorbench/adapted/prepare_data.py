"""Validate and adapt local BehaviorBench data; no model calls or dependencies.

Run from any directory: python3 /path/to/adapted/prepare_data.py
Generated JSON files are deterministic. Source datasets are never modified.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "risk-taking_v2.json": ("O", "Openness", "risk-taking"),
    "self-control_v2.json": ("C", "Conscientiousness", "self-control"),
    "social_media_v2.json": ("E", "Extraversion", "social-media engagement"),
    "altruism_v2.json": ("A", "Agreeableness", "altruism"),
    "heuristic_v2.json": ("N", "Neuroticism", "heuristic choice"),
}


def normalized(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score_response(item, response):
    """Return endorsement 0/1 or None for invalid output; never default to No.

    For forced choices accept A/B or the exact choice text, not numeric positions.
    Option order must be recorded by the caller if counterbalanced at inference.
    """
    if not isinstance(response, str):
        return None
    answer = response.strip().rstrip(".!?").strip().casefold()
    for option in item["options"]:
        if answer in {option["key"].casefold(), option["text"].casefold()}:
            return option["endorsement"]
    return None


def load_items(root):
    items, hashes = [], {}
    for filename, (code, trait, proxy) in SOURCES.items():
        raw = (root / filename).read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        require(isinstance(data, list) and data, f"{filename}: expected nonempty list")
        for record, entry in enumerate(data, 1):
            require(isinstance(entry, dict), f"{filename}:{record}: expected object")
            if code == "N":
                fields = ["question", "choice1", "choice2", "choice1_label", "choice2_label"]
                require(all(isinstance(entry.get(k), str) and entry[k].strip() for k in fields),
                        f"{filename}:{record}: missing/empty heuristic field")
                require({entry["choice1_label"], entry["choice2_label"]} == {"Y", "N"},
                        f"{filename}:{record}: invalid labels")
                require(normalized(entry["choice1"]) != normalized(entry["choice2"]),
                        f"{filename}:{record}: identical choices")
                # Manual inspection of all source questions finds two templates:
                # 1-50 salient anecdotes versus statistics; 51-100 descriptions versus base rates.
                family = "availability" if record <= 50 else "representativeness"
                group = f"N-{family}"
                options = [{"key": key, "text": entry[f"choice{i}"],
                            "source_label": entry[f"choice{i}_label"],
                            "endorsement": int(entry[f"choice{i}_label"] == "Y")}
                           for i, key in [(1, "A"), (2, "B")]]
                texts, behavior, reverse = [entry["question"]], None, None
                flags = ["exploratory_proxy", "heuristic_family_holdout",
                         "choice_order_confounded_with_family", "semantic_review_required"]
                if record in {62, 74}:
                    flags.append("question_explicitly_reveals_choice_attribute")
                if record == 68:
                    flags.append("label_interpretation_requires_review")
            else:
                require(isinstance(entry.get("behavior"), str) and entry["behavior"].strip(),
                        f"{filename}:{record}: empty behavior")
                texts = entry.get("scenarios")
                require(isinstance(texts, list) and texts, f"{filename}:{record}: empty scenarios")
                if code == "C":
                    require(entry.get("label") in {"", "R"}, f"{filename}:{record}: bad polarity")
                reverse = entry.get("label") == "R"
                options = [{"key": "Yes", "text": "Yes", "endorsement": int(not reverse)},
                           {"key": "No", "text": "No", "endorsement": int(reverse)}]
                behavior, family = entry["behavior"], None
                group = f"{code}-b{record:02d}"
                flags = ["semantic_review_required"]
                if code in {"O", "E"}:
                    flags.append("narrow_behavioral_proxy")
            for scenario, text in enumerate(texts, 1):
                require(isinstance(text, str) and text.strip(), f"{filename}:{record}:{scenario}: empty text")
                source_label = entry.get("label") if code == "C" else None
                items.append({
                    "id": f"bb-{code}-{record:03d}-{scenario:02d}",
                    "trait_code": code, "trait": trait, "behavioral_proxy": proxy,
                    "group_id": group, "heuristic_family": family,
                    "source": {"file": filename, "record_1based": record,
                               "scenario_1based": scenario if code != "N" else None,
                               "self_control_label": source_label},
                    "behavior": behavior, "scenario": text,
                    "response_type": "forced_choice" if code == "N" else "yes_no",
                    "options": options, "reverse_scored": reverse,
                    "context_labels": [], "context_review_status": "not_annotated",
                    "review_status": "pending_independent_semantic_review", "review_flags": flags,
                })
    return items, hashes


def assign_splits(items, seed):
    """Whole-group split; stratify C by polarity and conservatively hold out one N family."""
    rng = random.Random(seed)
    assignments = {}
    for code in ["O", "C", "E", "A"]:
        subset = [x for x in items if x["trait_code"] == code]
        groups = sorted({x["group_id"] for x in subset})
        if code == "C":
            # Three test groups: 2 direct + 1 reverse; development retains both.
            for reverse, n_test in [(False, 2), (True, 1)]:
                stratum = sorted({x["group_id"] for x in subset if x["reverse_scored"] == reverse})
                rng.shuffle(stratum)
                for i, group in enumerate(stratum):
                    assignments[group] = "test" if i < n_test else "development"
        else:
            rng.shuffle(groups)
            for i, group in enumerate(groups):
                assignments[group] = "test" if i < 3 else "development"
    # Fixed in advance, not selected using model outcomes. Randomizing two families
    # cannot fix family confounding, so make the exact held-out family explicit.
    assignments["N-availability"] = "development"
    assignments["N-representativeness"] = "test"
    for item in items:
        item["split"] = assignments[item["group_id"]]
    return assignments


def audit(items):
    require(len({x["id"] for x in items}) == len(items), "duplicate IDs")
    groups, texts, prompts = defaultdict(set), defaultdict(list), defaultdict(list)
    for item in items:
        groups[item["group_id"]].add(item["split"])
        texts[normalized(item["scenario"])].append(item["id"])
        signature = (normalized(item["behavior"] or ""), normalized(item["scenario"]),
                     tuple(normalized(o["text"]) for o in item["options"]))
        prompts[signature].append(item)
    require(all(len(s) == 1 for s in groups.values()), "group leakage")
    exact = [ids for ids in texts.values() if len(ids) > 1]
    require(not any(len({x["split"] for x in cluster}) > 1 for cluster in prompts.values()),
            "duplicate prompt leakage")
    # Heuristic lexical screen, not a semantic leakage guarantee. Inspect flagged pairs.
    near = []
    for i, a in enumerate(items):
        for b in items[i+1:]:
            if a["split"] == b["split"] or a["trait_code"] != b["trait_code"]:
                continue
            left, right = normalized(a["scenario"]), normalized(b["scenario"])
            if min(len(left), len(right)) / max(len(left), len(right)) < .9:
                continue
            similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
            if similarity >= .9:
                near.append({"development_id": a["id"] if a["split"] == "development" else b["id"],
                             "test_id": b["id"] if b["split"] == "test" else a["id"],
                             "similarity": round(similarity, 4),
                             "status": "requires_semantic_review"})
    return {"group_overlap": [], "exact_scenario_duplicate_clusters": exact,
            "cross_split_near_duplicate_candidates": near,
            "near_duplicate_threshold": .9,
            "semantic_leakage_guaranteed_absent": False}


def build(seed=20261027, root=ROOT):
    items, hashes = load_items(root)
    assignments = assign_splits(items, seed)
    checks = audit(items)
    counts = {}
    for code, trait, _ in SOURCES.values():
        subset = [x for x in items if x["trait_code"] == code]
        counts[trait] = {s: {"items": sum(x["split"] == s for x in subset),
                            "groups": len({x["group_id"] for x in subset if x["split"] == s})}
                         for s in ["development", "test"]}
    manifest = {"schema_version": 1, "seed": seed, "source_sha256": hashes,
                "split_unit": "source behavior; heuristic questions grouped into two manually identified families",
                "group_assignments": dict(sorted(assignments.items())), "counts": counts,
                "excluded_items": [], "all_traits_primary_ready": False,
                "status": "structurally_validated_semantic_review_pending",
                "steering_examples": "Not supplied. Extract only from development or separately sourced data audited against test groups.",
                "test_policy": "Freeze before any model-based selection; do not tune on test responses."}
    report = {"total_items": len(items), "total_groups": len(assignments),
              "split_counts": dict(Counter(x["split"] for x in items)),
              "counts_by_trait": counts, "checks": checks,
              "self_control_polarity": {s: dict(Counter("reverse" if x["reverse_scored"] else "direct"
                  for x in items if x["trait_code"] == "C" and x["split"] == s))
                  for s in ["development", "test"]},
              "heuristic_choice_position": {family: dict(Counter(next(o["key"] for o in x["options"] if o["endorsement"] == 1)
                  for x in items if x["heuristic_family"] == family))
                  for family in ["availability", "representativeness"]},
              "semantic_review_pending": len(items), "context_annotation_pending": len(items)}
    return items, manifest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    items, manifest, report = build(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"all_items.json": items, "development.json": [x for x in items if x["split"] == "development"],
               "test.json": [x for x in items if x["split"] == "test"],
               "split_manifest.json": manifest, "validation_report.json": report}
    for name, value in outputs.items():
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "items": len(items),
                      "splits": report["split_counts"], "near_duplicate_candidates":
                      len(report["checks"]["cross_split_near_duplicate_candidates"]),
                      "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
