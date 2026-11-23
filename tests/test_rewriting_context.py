"""Cross-context arithmetic on hand-built records with known answers."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rewriting_context as context  # noqa: E402


def record(item, level, score, condition="free", trait="C", split="test"):
    return {"item_id": item, "item_trait": trait, "condition": condition, "level": level,
            "score": score, "kind": "yes_no", "split": split, "seed": 0}


LABELS = {"a": ["work_school"], "b": ["work_school"], "c": ["home_personal"]}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.records = [record("a", 1, 0), record("a", 5, 1), record("b", 1, 0), record("b", 5, 0),
                        record("c", 1, 1), record("c", 5, 1),
                        record("a", 1, None), record("a", 3, 1, condition="locked"),
                        record("b", 5, 1, split="development"), record("z", 5, 1)]

    def summary(self, splits=("test",), min_items=2):
        return {r["context"]: r for r in context.analyze(self.records, LABELS, "free", splits, min_items)[1]}

    def test_cells_use_only_valid_labeled_records_of_the_condition_and_split(self):
        work = self.summary()["work_school"]
        self.assertEqual((work["n_items"], work["e1"], work["e5"]), (2, 0.0, 0.5))
        self.assertAlmostEqual(work["gap"], 0.5)
        self.assertAlmostEqual(work["slope"], 0.125)

    def test_small_cells_are_flagged_not_dropped(self):
        rows = self.summary()
        self.assertTrue(rows["work_school"]["reportable"])
        self.assertFalse(rows["home_personal"]["reportable"])
        self.assertEqual(rows["home_personal"]["gap"], 0)

    def test_splits_can_be_combined(self):
        work = self.summary(splits=("development", "test"))["work_school"]
        self.assertAlmostEqual(work["e5"], 2 / 3)

    def test_item_bootstrap_brackets_the_estimate(self):
        items = context.item_scores(self.records, LABELS, "free", {"test"})[("C", "work_school")]
        low, high = context.bootstrap(items)["gap"]
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)

    def test_long_rows_have_levels_gap_and_slope(self):
        rows = context.analyze(self.records, LABELS, "free", ("test",), 2)[0]
        metrics = {(r["context"], r["metric"], r["target_score"]) for r in rows}
        self.assertIn(("work_school", "endorsement_rate", 3), metrics)
        self.assertIn(("work_school", "dose_slope", ""), metrics)
        self.assertEqual({r["trait"] for r in rows}, {"Conscientiousness"})

    def test_real_labels_cover_every_item(self):
        labels = context.load_labels()
        self.assertEqual(len(labels), 510)
        self.assertTrue(all(set(v) <= set(context.CONTEXTS) and v for v in labels.values()))


if __name__ == "__main__":
    unittest.main()
