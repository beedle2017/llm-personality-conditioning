"""Offline tests for the rewriting pipeline, using a scripted fake model."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewriting import prompts  # noqa: E402
from rewriting.pipeline import (ADAPTED_DIR, CONDITIONS, behavior_context, bfi_context,  # noqa: E402
                                clean_statement, plan_records, preservation, run, score_extraction)
from rewriting.traits import LEVELS, TRAITS, describe_target  # noqa: E402

ITEMS = json.loads((ADAPTED_DIR / "data" / "development.json").read_text(encoding="utf-8"))
YES_NO = next(x for x in ITEMS if x["trait_code"] == "A")
REVERSE = next(x for x in ITEMS if x["reverse_scored"] is True)
CHOICE = next(x for x in ITEMS if x["response_type"] == "forced_choice")


class FakeModel:
    """Writes 'I would do it' unless a high target is present; extraction reads the text."""

    def __init__(self, extract=None):
        self.calls, self.extract = [], extract

    def generate(self, conversations, *, temperature, max_new_tokens, seed):
        outputs = []
        for messages in conversations:
            user = messages[-1]["content"]
            self.calls.append((user, temperature))
            if "Statement:" in user:
                outputs.append(self.extract or ("No" if "refuse" in user else "Yes"))
            elif "(very low)" in user:
                outputs.append("I would refuse.")
            else:
                outputs.append("I would do it.")
        return outputs


class PromptTests(unittest.TestCase):
    def test_every_level_describes_target(self):
        for code in TRAITS:
            for level in LEVELS:
                self.assertIn(f"score: {level} on a scale", describe_target(code, level))
        with self.assertRaises(ValueError):
            describe_target("O", 6)

    def test_trait_text_does_not_name_proxies(self):
        text = json.dumps(TRAITS).lower()
        for word in ["risk", "video", "social media", "altruis", "heuristic", "self-control"]:
            self.assertNotIn(word, text)

    def test_openness_does_not_invite_questioning_the_draft(self):
        self.assertNotIn("question", describe_target("O", 5))

    def test_no_prompt_reveals_scoring_key(self):
        for item in (YES_NO, REVERSE, CHOICE):
            context = behavior_context(item, 0)
            built = [prompts.draft_messages(context), prompts.paraphrase_messages(context, "d"),
                     prompts.rewrite_messages(context, "d", item["trait_code"], 5, "free"),
                     prompts.no_draft_messages(context, item["trait_code"], 5),
                     prompts.extract_messages(context, "d")]
            for messages in built:
                self.assertNotIn("endorsement", json.dumps(messages))

    def test_extraction_is_blind_to_target(self):
        context = behavior_context(YES_NO, 0)
        text = json.dumps(prompts.extract_messages(context, "I would do it."))
        for word in ["score", "Agreeableness", "very high", "personality"]:
            self.assertNotIn(word, text)

    def test_yes_no_extraction_names_behavior(self):
        context = behavior_context(YES_NO, 0)
        ask = prompts.extract_messages(context, "I would do it.")[-1]["content"]
        self.assertIn(f"do this: '{YES_NO['behavior']}'", ask)

    def test_hesitant_instruction_only_on_hesitant_items(self):
        choices = [x for x in ITEMS if x["response_type"] == "forced_choice"]
        hesitant = next(x for x in choices if "hesitant" in x["scenario"])
        ordinary = next(x for x in choices if "hesitant" not in x["scenario"])
        for item, expected in ((hesitant, True), (ordinary, False)):
            context = behavior_context(item, 0)
            ask = prompts.extract_messages(context, "I would pick one.")[-1]["content"]
            self.assertEqual("hesitant" in ask, expected)
            for option in context.displayed_options:
                self.assertIn(option, ask)

    def test_no_draft_omits_draft(self):
        context = behavior_context(YES_NO, 0)
        self.assertNotIn("Original statement", json.dumps(prompts.no_draft_messages(context, "A", 5)))


class ScoringTests(unittest.TestCase):
    def test_yes_no_respects_reverse_scoring(self):
        context = behavior_context(REVERSE, 0)
        self.assertEqual(score_extraction(context, "Yes"), 0)
        self.assertEqual(score_extraction(context, "No."), 1)
        self.assertIsNone(score_extraction(context, "Unclear"))

    def test_forced_choice_scores_text_under_any_order(self):
        for seed in range(6):
            context = behavior_context(CHOICE, seed)
            for option in CHOICE["options"]:
                self.assertEqual(score_extraction(context, option["text"]), option["endorsement"])

    def test_forced_choice_rejects_stored_keys(self):
        context = behavior_context(CHOICE, 0)
        self.assertIsNone(score_extraction(context, "A"))
        self.assertIsNone(score_extraction(context, "b."))

    def test_forced_choice_order_varies_and_is_repeatable(self):
        orders = {behavior_context(CHOICE, seed).displayed_options for seed in range(12)}
        self.assertEqual(len(orders), 2)
        self.assertEqual(behavior_context(CHOICE, 3), behavior_context(CHOICE, 3))

    def test_bfi_reverse_and_invalid(self):
        direct, reverse = bfi_context(1, "Is talkative", "E", False), bfi_context(6, "Is reserved", "E", True)
        self.assertEqual(score_extraction(direct, "4"), 4)
        self.assertEqual(score_extraction(reverse, "4."), 2)
        self.assertIsNone(score_extraction(direct, "On a scale of 1 to 5, 4"))
        self.assertIsNone(score_extraction(direct, "Unclear"))
        self.assertIn("someone who is talkative.", direct.prompt)


class HelperTests(unittest.TestCase):
    def test_clean_statement(self):
        self.assertEqual(clean_statement('Here is the rewrite:\n"I would go."'), "I would go.")
        self.assertEqual(clean_statement("I would go."), "I would go.")

    def test_preservation(self):
        result = preservation("I paid 30 dollars", "I paid 30 dollars")
        self.assertEqual((result["length_ratio"], result["word_jaccard"], result["numbers_kept"]), (1.0, 1.0, 1.0))
        self.assertEqual(preservation("I paid 30", "I paid")["numbers_kept"], 0.0)
        self.assertIsNone(preservation("no digits", "none")["numbers_kept"])

    def test_plan_counts(self):
        records = plan_records(behavior_context(YES_NO, 0), 0, ["A"])
        self.assertEqual(len(records), 2 + 3 * len(LEVELS))


class PipelineTests(unittest.TestCase):
    def run_one(self, model, item=YES_NO):
        return run(model, [(behavior_context(item, 0), [item["trait_code"]])], seed=0, batch_size=4)

    def test_conditions_and_scores(self):
        records = self.run_one(FakeModel())
        self.assertEqual({r["condition"] for r in records}, set(CONDITIONS))
        low = next(r for r in records if r["condition"] == "free" and r["level"] == 1)
        high = next(r for r in records if r["condition"] == "free" and r["level"] == 5)
        self.assertEqual((low["text"], low["score"]), ("I would refuse.", 0))
        self.assertEqual((high["text"], high["score"]), ("I would do it.", 1))

    def test_draft_generated_once_and_extraction_is_greedy(self):
        model = FakeModel()
        self.run_one(model)
        self.assertEqual(sum(prompts.DRAFT_TASK in u and "Write as" not in u for u, _ in model.calls), 1)
        self.assertTrue(all(t == 0.0 for u, t in model.calls if "Statement:" in u))

    def test_refusal_is_invalid_and_never_uses_draft(self):
        class Refuser(FakeModel):
            def generate(self, conversations, **kwargs):
                texts = super().generate(conversations, **kwargs)
                return ["As an AI, I cannot." if "Original statement" in c[-1]["content"] else t
                        for c, t in zip(conversations, texts)]
        records = self.run_one(Refuser())
        free = [r for r in records if r["condition"] == "free"]
        self.assertTrue(all(r["refusal"] and r["score"] is None and r["extracted"] is None for r in free))

    def test_in_character_refusal_is_kept(self):
        class Blunt(FakeModel):
            def generate(self, conversations, **kwargs):
                texts = super().generate(conversations, **kwargs)
                return ["I can't help you move, sorry." if "Original statement" in c[-1]["content"] else t
                        for c, t in zip(conversations, texts)]
        records = self.run_one(Blunt())
        self.assertFalse(any(r["refusal"] for r in records))

    def test_empty_rewrite_is_invalid(self):
        class Empty(FakeModel):
            def generate(self, conversations, **kwargs):
                texts = super().generate(conversations, **kwargs)
                return ["" if "Original statement" in c[-1]["content"] else t
                        for c, t in zip(conversations, texts)]
        free = [r for r in self.run_one(Empty()) if r["condition"] == "free"]
        self.assertTrue(all(r["score"] is None for r in free))

    def test_unclear_extraction_is_invalid(self):
        records = self.run_one(FakeModel(extract="Unclear"))
        self.assertTrue(all(r["score"] is None for r in records))

    def test_bfi_runs_each_target_trait(self):
        context = bfi_context(1, "Is talkative", "E", False)
        records = run(FakeModel(extract="4"), [(context, ["E", "O"])], seed=0)
        self.assertEqual(len([r for r in records if r["condition"] == "free"]), 2 * len(LEVELS))
        self.assertTrue(all(r["score"] == 4 for r in records))


class ExperimentTests(unittest.TestCase):
    """`run_experiment` reuses one loaded generator across runs."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_rewriting import run_experiment
        cls.run_experiment = staticmethod(run_experiment)

    def test_same_generator_serves_several_runs(self):
        import tempfile
        model = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.jsonl", "b.jsonl"):
                out = Path(tmp) / name
                total = self.run_experiment(model, out, traits=["A"], limit=2, levels=[1, 5])
                lines = out.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), total)
                self.assertEqual(len(lines), 2 * (2 + 3 * 2))
                self.assertEqual(json.loads(lines[0])["split"], "development")

    def test_guards(self):
        with self.assertRaises(ValueError):
            self.run_experiment(FakeModel(), "unused.jsonl", split="test")
        with self.assertRaises(ValueError):
            self.run_experiment(FakeModel(), "unused.jsonl", conditions=["free"])


if __name__ == "__main__":
    unittest.main()
