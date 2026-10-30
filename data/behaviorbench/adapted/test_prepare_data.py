"""Offline checks for adaptation, leakage boundaries, and scoring."""
import copy
import unittest

from prepare_data import audit, build, score_response


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.manifest, cls.report = build()

    def test_inventory_and_partitions(self):
        self.assertEqual(len(self.items), 510)
        self.assertEqual(self.report["split_counts"], {"development": 340, "test": 170})
        self.assertEqual(len(self.manifest["group_assignments"]), 43)
        for group in self.manifest["group_assignments"]:
            self.assertEqual(len({x["split"] for x in self.items if x["group_id"] == group}), 1)

    def test_determinism(self):
        self.assertEqual(build(), (self.items, self.manifest, self.report))

    def test_polarity_and_invalid_responses(self):
        for item in self.items:
            if item["trait_code"] == "C":
                self.assertEqual(score_response(item, "Yes."), int(not item["reverse_scored"]))
                self.assertEqual(score_response(item, "No"), int(item["reverse_scored"]))
            self.assertIsNone(score_response(item, "Yes or no; it depends"))
            self.assertIsNone(score_response(item, ""))
            for option in item["options"]:
                self.assertEqual(score_response(item, option["key"]), option["endorsement"])
                self.assertEqual(score_response(item, option["text"]), option["endorsement"])

    def test_leakage_detection(self):
        corrupt = copy.deepcopy(self.items)
        copy_item = copy.deepcopy(corrupt[0])
        copy_item["id"] = "injected-leak"
        copy_item["split"] = "test" if copy_item["split"] == "development" else "development"
        corrupt.append(copy_item)
        with self.assertRaisesRegex(ValueError, "group leakage"):
            audit(corrupt)

    def test_both_polarities_retained_and_heuristic_families_separate(self):
        for split in ["development", "test"]:
            self.assertEqual({x["reverse_scored"] for x in self.items
                              if x["trait_code"] == "C" and x["split"] == split}, {True, False})
        self.assertEqual({x["heuristic_family"] for x in self.items if x["trait_code"] == "N"
                          and x["split"] == "test"}, {"representativeness"})


if __name__ == "__main__":
    unittest.main()
