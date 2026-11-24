"""Cross-context arithmetic for item-level BiPO results."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bipo_context as context  # noqa: E402


def record(item, level, score, trait="conscientiousness", valid="True"):
    return {
        "item_id": item,
        "trait": trait,
        "target_score": level,
        "score": score,
        "valid": valid,
    }


LABELS = {"a": ["work_school"], "b": ["work_school"], "c": ["home_personal"]}


class BipoContextTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("a", 1, 0),
            record("a", 5, 1),
            record("b", 1, 0),
            record("b", 5, 0),
            record("c", 1, 1),
            record("c", 5, 1),
            record("a", 1, "", valid="False"),
            record("a", 3, 1, trait="openness"),
            record("b", 5, 1, trait="unknown"),
            record("unlabeled", 5, 1),
        ]

    def summary(self, min_items=2):
        return {
            row["context"]: row
            for row in context.analyze(self.records, LABELS, min_items)[1]
        }

    def test_cells_use_only_valid_labeled_rows_for_matching_traits(self):
        work = self.summary()["work_school"]
        self.assertEqual((work["n_items"], work["e1"], work["e5"]), (2, 0.0, 0.5))
        self.assertAlmostEqual(work["gap"], 0.5)
        self.assertAlmostEqual(work["slope"], 0.125)

    def test_small_cells_are_flagged_without_dropping_results(self):
        rows = self.summary()
        self.assertTrue(rows["work_school"]["reportable"])
        self.assertFalse(rows["home_personal"]["reportable"])
        self.assertEqual(rows["home_personal"]["gap"], 0)

    def test_item_bootstrap_brackets_the_estimate(self):
        items = context.item_scores(self.records, LABELS)[("conscientiousness", "work_school")]
        low, high = context.bootstrap(items)["gap"]
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)

    def test_long_rows_include_level_and_dose_metrics(self):
        rows = context.analyze(self.records, LABELS, min_items=2)[0]
        metrics = {(row["context"], row["metric"], row["target_score"]) for row in rows}
        self.assertIn(("work_school", "endorsement_rate", 3), metrics)
        self.assertIn(("work_school", "dose_slope", ""), metrics)
        work_rows = [row for row in rows if row["context"] == "work_school"]
        self.assertEqual(
            {row["trait"] for row in work_rows},
            {"Conscientiousness", "Openness"},
        )

    def test_real_labels_cover_all_test_items(self):
        labels = context.load_labels()
        self.assertEqual(len(labels), 510)
        self.assertTrue(
            all(value and set(value) <= set(context.CONTEXTS) for value in labels.values())
        )
        items = json.loads(
            (ROOT / "data/behaviorbench/adapted/data/all_items_labeled.json").read_text()
        )
        self.assertEqual(
            sum(bool(labels[item["id"]]) for item in items if item["split"] == "test"),
            170,
        )


if __name__ == "__main__":
    unittest.main()
