"""Offline tests for multi-turn BIPO steering, using a scripted fake model and a fake hook."""
import itertools
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(path))

import bipo_temporal as temporal  # noqa: E402
from direct_prompting_temporal import sample_ids  # noqa: E402
from get_batches import batches_ending_with_id  # noqa: E402
from steering.hooks import steering_hook  # noqa: E402

SEED = 20261026
VECTORS = {t: torch.ones(4) for t in "OCEAN"}


class FakeModel:
    """Answers 'ANSWER n' (tagged with the active coefficient), extracts 'Yes'; logs every call."""

    def __init__(self):
        self.calls, self.count, self.coeff = [], itertools.count(), None

    def generate(self, conversations, *, temperature, max_new_tokens, seed):
        outputs = []
        for messages in conversations:
            self.calls.append((messages, self.coeff))
            if "Statement:" in messages[-1]["content"]:
                outputs.append("Yes")
            else:
                outputs.append(f"ANSWER {next(self.count)} c={self.coeff}")
        return outputs


@contextmanager
def fake_steer(generator, vector, coeff, layer):
    generator.coeff = coeff
    try:
        yield
    finally:
        generator.coeff = None


def run(model, levels=(1, 5), max_turns=4, probe_turns=(2, 4)):
    ids, by_id = sample_ids(["C"], 1, None, SEED)
    records = temporal.run(model, ids, by_id, seed=SEED, vectors=VECTORS, levels=levels, batch_size=3,
                           max_turns=max_turns, probe_turns=list(probe_turns), steer_fn=fake_steer)
    return ids[0], by_id, records


def is_extract(messages):
    return "Statement:" in messages[-1]["content"]


class TemporalTests(unittest.TestCase):
    def test_records_one_per_probe_and_level(self):
        _, _, records = run(FakeModel())
        self.assertEqual(len(records), 2 * 2)
        self.assertEqual({(r["level"], r["turn"]) for r in records}, {(1, 1), (1, 3), (5, 1), (5, 3)})
        self.assertTrue(all(r["condition"] == "bipo" and r["score"] is not None for r in records))
        self.assertEqual({r["level"]: r["coeff"] for r in records}, {1: -4.0, 5: 4.0})

    def test_every_answer_is_steered_with_its_level_and_extraction_is_not(self):
        model = FakeModel()
        run(model)
        for messages, coeff in model.calls:
            if is_extract(messages):
                self.assertIsNone(coeff)
            else:
                self.assertIn(coeff, (-4.0, 4.0))
        for messages, coeff in model.calls:  # history answers carry their own conversation's coefficient
            assistants = [m["content"] for m in messages if m["role"] == "assistant"]
            if not is_extract(messages):
                self.assertTrue(all(a.endswith(f"c={coeff}") for a in assistants))

    def test_history_order_matches_get_batches_and_no_persona(self):
        model = FakeModel()
        target, by_id, _ = run(model, levels=(5,))
        order = list(itertools.islice(batches_ending_with_id(target, seed=SEED), 4))[-1][:-1]
        last_probe = [m for m, _ in model.calls
                      if not is_extract(m) and by_id[target]["scenario"] in m[-1]["content"]][-1]
        self.assertEqual(last_probe[0]["content"], temporal.prompts.SPEAKER)
        users = [m["content"] for m in last_probe[1:-1] if m["role"] == "user"]
        self.assertEqual(len(users), 3)
        for item, user in zip(order, users):
            self.assertIn(item["scenario"], user)

    def test_probe_never_enters_history(self):
        model = FakeModel()
        target, by_id, _ = run(model, levels=(1,))
        scenario = by_id[target]["scenario"]
        for messages, _ in model.calls:
            if not is_extract(messages):
                self.assertFalse(any(scenario in m["content"] for m in messages[1:-1]))

    def test_extraction_is_blind(self):
        model = FakeModel()
        run(model)
        for messages, _ in model.calls:
            if is_extract(messages):
                self.assertEqual(len(messages), 2)
                text = json.dumps(messages)
                for word in ["score", "Conscientiousness", "very high", "personality"]:
                    self.assertNotIn(word, text)

    def test_mixed_levels_are_batched_separately(self):
        sizes = []

        class Sizes(FakeModel):
            def generate(self, conversations, **kwargs):
                if not is_extract(conversations[0]):
                    sizes.append({self.coeff})
                return super().generate(conversations, **kwargs)

        ids, by_id = sample_ids(["C", "E"], 2, None, SEED)
        temporal.run(Sizes(), ids, by_id, seed=SEED, vectors=VECTORS, levels=(1, 3, 5), batch_size=16,
                     max_turns=2, probe_turns=[2], steer_fn=fake_steer)
        self.assertTrue(sizes and all(len(s) == 1 for s in sizes))

    def test_real_hook_shifts_layer_output(self):
        layer = torch.nn.Identity()
        layer.register_forward_hook(steering_hook(2.0, torch.ones(3)))
        self.assertTrue(torch.equal(layer(torch.zeros(1, 2, 3)), torch.full((1, 2, 3), 2.0)))

    def test_run_experiment_writes_records(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "t.jsonl"
            total = temporal.run_experiment(FakeModel(), out, traits=["E"], levels=[1], ids_per_trait=2,
                                            max_turns=3, probe_turns=[3], vectors=VECTORS, steer_fn=fake_steer)
            lines = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(total, len(lines))
        self.assertEqual(len(lines), 2)
        self.assertEqual({r["method"] for r in lines}, {"bipo"})

    def test_vectors_load(self):
        vectors = temporal.load_vectors()
        self.assertEqual(set(vectors), set("OCEAN"))
        self.assertTrue(all(v.ndim == 1 for v in vectors.values()))


if __name__ == "__main__":
    unittest.main()
