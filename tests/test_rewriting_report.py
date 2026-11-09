"""Report arithmetic on hand-built records with known answers. Requires numpy."""
import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rewriting_report as report  # noqa: E402


def record(item, condition, level, score, kind="yes_no", trait="A", extracted=None):
    return {"item_id": item, "seed": 0, "condition": condition, "level": level, "score": score, "kind": kind,
            "item_trait": trait, "target_trait": trait if level else None, "extracted": extracted, "text": "t"}


def captured(function, *args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        function(*args)
    return output.getvalue()


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.drafts = [record("a", "draft", None, 1), record("b", "draft", None, 1),
                       record("c", "draft", None, 0), record("d", "draft", None, 1)]
        self.low = [record("a", "free", 1, 0), record("b", "free", 1, 1),
                    record("c", "free", 1, 0), record("d", "free", 1, None)]

    def test_rate_excludes_invalid(self):
        self.assertAlmostEqual(report.rate(self.low), 100 / 3)
        self.assertEqual(report.invalid_rate(self.low), 25)
        self.assertIsNone(report.rate([record("a", "free", 1, None)]))

    def test_flips_use_valid_pairs_by_direction(self):
        baseline = {(r["item_id"], r["seed"]): r["score"] for r in self.drafts}
        result = report.flips(self.low + [record("c", "free", 1, 1)], baseline)
        self.assertEqual(result, {"up": 25, "down": 25})

    def test_summary_pis_is_positive_for_lower_rate_at_low_target(self):
        line = next(x for x in captured(report.behavior_summary, self.drafts + self.low).splitlines()
                    if " free " in x)
        # Baseline 75%, rate 33.3%: PIS = -1 * (33.3 - 75) = 41.7.
        self.assertEqual(line.split()[3:6], ["4", "33.3", "41.7"])

    def test_summary_leaves_pis_undefined_at_neutral(self):
        rows = self.drafts + [record("a", "free", 3, 1)]
        line = next(x for x in captured(report.behavior_summary, rows).splitlines() if " free " in x)
        self.assertEqual(line.split()[5], "-")

    def test_bfi_error_uses_only_target_trait_items(self):
        rows = [record("x", "free", 5, 4, "bfi", "E"), record("y", "free", 5, 2, "bfi", "E"),
                dict(record("z", "free", 5, 1, "bfi", "O"), target_trait="E")]
        line = captured(report.bfi_summary, rows).splitlines()[-1]
        self.assertEqual(line.split()[3:5], ["3.0", "2.0"])

    def test_label_round_trip_hides_extraction_and_measures_agreement(self):
        rows = [record("a", "free", 1, 1, extracted="Yes."), record("b", "paraphrase", None, 0, extracted="No")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            captured(report.write_label_sample, rows, path, 2)
            with path.open(newline="", encoding="utf-8") as handle:
                sheet = list(csv.DictReader(handle))
            self.assertNotIn("extracted", sheet[0])
            for row in sheet:
                row["human_label"] = "yes"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, sheet[0].keys())
                writer.writeheader()
                writer.writerows(sheet)
            output = captured(report.agreement, rows, path)
            self.assertIn("50.0% over 2 labels", output)
            self.assertIn("A: 50.0% over 2", output)

    def test_results_csv_uses_fractions_and_matches_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            captured(report.write_results_csv, self.drafts + self.low, path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = {row["condition"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows["draft"]["target_score"], "")
        free = rows["free"]
        self.assertEqual((free["trait"], free["n"], free["valid_n"]), ("Agreeableness", "4", "3"))
        self.assertAlmostEqual(float(free["endorsement_rate"]), 1 / 3)
        self.assertAlmostEqual(float(free["pis"]), 0.41667, places=4)
        self.assertAlmostEqual(float(free["invalid_rate"]), 0.25)

    def test_label_sheet_shows_question_and_accepts_choice_numbers(self):
        import json
        item = json.loads((ROOT / "data/behaviorbench/adapted/data/development.json").read_text())[0]
        choice = dict(record(item["id"], "free", 1, 1, kind="forced_choice", extracted="hardwood"),
                      displayed_options=["vinyl", "hardwood"])
        self.assertIn(item["scenario"], report._question(choice, report._items()))
        self.assertEqual(report._normalize("2", choice), "hardwood")
        self.assertEqual(report._normalize("Yes.", record("a", "free", 1, 1)), "yes")


if __name__ == "__main__":
    unittest.main()
