"""Offline tests for multi-turn rewriting, using a scripted fake model."""
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(path))

import rewriting_temporal as temporal  # noqa: E402
from direct_prompting_temporal import sample_ids  # noqa: E402
from get_batches import batches_ending_with_id  # noqa: E402

SEED = 20261026


class FakeModel:
    """Drafts 'DRAFT n', rewrites 'REWRITE n', extracts 'Yes'; keeps every conversation it saw."""

    def __init__(self):
        self.calls, self.count = [], itertools.count()

    def generate(self, conversations, *, temperature, max_new_tokens, seed):
        outputs = []
        for messages in conversations:
            self.calls.append(messages)
            user = messages[-1]["content"]
            if "Statement:" in user:
                outputs.append("Yes")
            elif "Original statement" in user:
                outputs.append(f"REWRITE {next(self.count)}")
            else:
                outputs.append(f"DRAFT {next(self.count)}")
        return outputs


def run(model, levels=(1, 5), max_turns=4, probe_turns=(2, 4)):
    ids, by_id = sample_ids(["C"], 1, None, SEED)
    records = temporal.run(model, ids, by_id, seed=SEED, levels=levels, batch_size=3,
                           max_turns=max_turns, probe_turns=list(probe_turns))
    return ids[0], by_id, records


def kind(messages):
    user = messages[-1]["content"]
    return "extract" if "Statement:" in user else "rewrite" if "Original statement" in user else "draft"


class TemporalTests(unittest.TestCase):
    def test_same_targets_as_direct_prompting(self):
        self.assertEqual(sample_ids(["C"], 3, None, SEED)[0], sample_ids(["C"], 3, None, SEED)[0])

    def test_records_one_per_probe_and_level(self):
        _, _, records = run(FakeModel())
        self.assertEqual(len(records), 2 * 2)
        self.assertEqual({(r["level"], r["turn"]) for r in records}, {(1, 1), (1, 3), (5, 1), (5, 3)})
        self.assertTrue(all(r["condition"] == "free" and r["score"] is not None for r in records))
        self.assertTrue(all(r["text"].startswith("REWRITE") and r["draft"].startswith("DRAFT") for r in records))

    def test_history_order_matches_get_batches_and_holds_rewrites(self):
        model = FakeModel()
        target, by_id, _ = run(model, levels=(5,))
        order = list(itertools.islice(batches_ending_with_id(target, seed=SEED), 4))[-1][:-1]
        last_probe = [m for m in model.calls if kind(m) == "draft" and by_id[target]["scenario"] in m[-1]["content"]][-1]
        history = last_probe[1:-1]
        users = [m["content"] for m in history if m["role"] == "user"]
        self.assertEqual(len(users), 3)
        for item, user in zip(order, users):
            self.assertIn(item["scenario"], user)
        assistants = [m["content"] for m in history if m["role"] == "assistant"]
        self.assertTrue(all(text.startswith("REWRITE") for text in assistants))

    def test_probe_never_enters_history(self):
        model = FakeModel()
        target, by_id, _ = run(model, levels=(1,))
        scenario = by_id[target]["scenario"]
        for messages in model.calls:
            if kind(messages) != "extract":
                self.assertFalse(any(scenario in m["content"] for m in messages[1:-1]))

    def test_rewriter_sees_history_and_target(self):
        model = FakeModel()
        run(model, levels=(5,))
        rewrites = [m for m in model.calls if kind(m) == "rewrite"]
        self.assertGreater(max(len(m) for m in rewrites), 2)
        self.assertTrue(all("very high" in m[-1]["content"] for m in rewrites))

    def test_extraction_is_blind(self):
        model = FakeModel()
        run(model)
        for messages in model.calls:
            if kind(messages) == "extract":
                self.assertEqual(len(messages), 2)
                text = json.dumps(messages)
                for word in ["score", "Conscientiousness", "very high", "personality", "DRAFT"]:
                    self.assertNotIn(word, text)

    def test_empty_rewrite_falls_back_to_draft_in_history_but_is_invalid(self):
        class Empty(FakeModel):
            def generate(self, conversations, **kwargs):
                texts = super().generate(conversations, **kwargs)
                return ["" if kind(c) == "rewrite" else t for c, t in zip(conversations, texts)]
        model = Empty()
        _, _, records = run(model, levels=(1,))
        self.assertTrue(all(r["score"] is None for r in records))
        assistants = [m["content"] for c in model.calls for m in c if m["role"] == "assistant"]
        self.assertTrue(assistants and all(a.startswith("DRAFT") for a in assistants))

    def test_run_experiment_writes_records(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "t.jsonl"
            total = temporal.run_experiment(FakeModel(), out, traits=["E"], levels=[1], ids_per_trait=2,
                                            max_turns=3, probe_turns=[3])
            lines = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(total, len(lines))
        self.assertEqual(len(lines), 2)
        self.assertEqual({r["method"] for r in lines}, {"rewriting"})


if __name__ == "__main__":
    unittest.main()
