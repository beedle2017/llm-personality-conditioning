"""Offline unit tests and smoke test for multi-agent social robustness."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import (
    analyze_pair_condition,
    bootstrap_ci,
    compute_agent_endorsements,
)
from agents.conditioning import ConditionedAgent, FakeGenerator
from agents.dialogue import DialogueState, run_dialogues, truncate_history
from agents.linguistic import (
    count_hedges,
    extract_turn_features,
    extract_words,
)
from agents.prompts import (
    format_turn_history,
    make_probe_messages,
    make_system_message,
)
from rewriting.pipeline import ADAPTED_DIR, behavior_context
from steering.common_config import score_to_coeff

DEV_ITEMS = json.loads((ADAPTED_DIR / "data" / "development.json").read_text(encoding="utf-8"))
SAMPLE_ITEM_C = next(it for it in DEV_ITEMS if it["trait_code"] == "C")
SAMPLE_ITEM_A = next(it for it in DEV_ITEMS if it["trait_code"] == "A")


class TestAgentConditioning(unittest.TestCase):
    """Test ConditionedAgent role mapping, probe isolation, steering scope, and rewriting."""

    def test_role_mapping_and_system_message(self):
        generator = FakeGenerator(default_reply="I will adhere to deadlines.")
        agent = ConditionedAgent("prompting", "C", 5, generator)
        self.assertEqual(agent.history[0]["role"], "system")
        self.assertIn("score: 5", agent.history[0]["content"])
        self.assertIn("disciplined", agent.history[0]["content"])

        # Agent replies to a partner message
        messages = [
            agent.history[0],
            {"role": "user", "content": "How should we schedule the project?"},
        ]
        reply = agent.generate_turn(messages)
        self.assertIn("milestones", reply)

    def test_probe_isolation(self):
        """Probing must operate on a copy; real history must remain byte-identical."""
        generator = FakeGenerator()
        agent = ConditionedAgent("prompting", "C", 1, generator)
        agent.history.append({"role": "user", "content": "Hello!"})
        agent.history.append({"role": "assistant", "content": "Hi there."})

        history_before = copy.deepcopy(agent.history)
        context = behavior_context(SAMPLE_ITEM_C, seed=42)

        probe_res = agent.probe(context)
        self.assertIn("score", probe_res)
        self.assertIn("statement", probe_res)

        # Confirm history is byte-identical and has no probe residue
        self.assertEqual(agent.history, history_before)
        for msg in agent.history:
            self.assertNotIn("Situation / Question:", msg["content"])

    def test_steering_hook_scope_and_cleanup(self):
        """Hook is active only during generation, removed in finally block, and alpha = score_to_coeff."""
        generator = FakeGenerator()
        k_val = 4.0
        expected_alpha_5 = score_to_coeff(5, k_val)  # 4.0 * (5 - 3) / 2 = 4.0
        expected_alpha_1 = score_to_coeff(1, k_val)  # 4.0 * (1 - 3) / 2 = -4.0
        expected_alpha_3 = score_to_coeff(3, k_val)  # 0.0

        agent_high = ConditionedAgent("steering", "A", 5, generator, k=k_val)
        agent_low = ConditionedAgent("steering", "A", 1, generator, k=k_val)
        agent_mid = ConditionedAgent("steering", "A", 3, generator, k=k_val)

        self.assertEqual(agent_high.steering_alpha, expected_alpha_5)
        self.assertEqual(agent_low.steering_alpha, expected_alpha_1)
        self.assertEqual(agent_mid.steering_alpha, expected_alpha_3)

        # Before generation, no hooks active
        self.assertEqual(len(generator.active_hooks), 0)

        # Generate with high agent
        agent_high.generate_turn([agent_high.history[0], {"role": "user", "content": "Test"}])
        self.assertEqual(generator.observed_hook_alphas_during_generate[-1], [expected_alpha_5])
        self.assertEqual(len(generator.active_hooks), 0)  # Removed after

        # Level 3 agent has alpha=0, so no hook registered
        agent_mid.generate_turn([agent_mid.history[0], {"role": "user", "content": "Test"}])
        self.assertEqual(generator.observed_hook_alphas_during_generate[-1], [])
        self.assertEqual(len(generator.active_hooks), 0)

        # Error case: hook is removed even if generate raises
        class FaultyGenerator(FakeGenerator):
            def generate(self, *args, **kwargs):
                raise RuntimeError("Synthetic model crash")

        faulty = FaultyGenerator()
        agent_faulty = ConditionedAgent("steering", "A", 5, faulty, k=k_val)
        with self.assertRaises(RuntimeError):
            agent_faulty.generate_turn([agent_faulty.history[0], {"role": "user", "content": "Test"}])
        self.assertEqual(len(faulty.active_hooks), 0)

    def test_rewrite_replaces_draft_in_history(self):
        """In rewriting, the rewritten statement enters histories, not the unconditioned draft."""
        generator = FakeGenerator()
        agent = ConditionedAgent("rewrite", "A", 1, generator)
        reply = agent.generate_turn([agent.history[0], {"role": "user", "content": "Can we share costs?"}])
        self.assertIsNotNone(agent.last_draft)
        self.assertNotEqual(agent.last_draft, reply)
        self.assertIn("protect my own priorities", reply)


class TestDialogueRunner(unittest.TestCase):
    """Test lockstep batching, determinism, and truncation."""

    def setUp(self):
        self.tasks = [
            {"id": "t1", "prompt": "Task 1 prompt", "traits": ["C"]},
            {"id": "t2", "prompt": "Task 2 prompt", "traits": ["C"]},
        ]
        self.probe_items = [behavior_context(SAMPLE_ITEM_C, seed=1)]

    def test_lockstep_batching(self):
        """Generations happen in lockstep across dialogues."""
        generator = FakeGenerator()
        res = run_dialogues(
            method="prompting",
            trait_code="C",
            level_a=1,
            level_b=5,
            tasks=self.tasks,
            generator=generator,
            probe_items=self.probe_items,
            rounds=2,
            checkpoints=(0, 2),
            seed=20261026,
            batch_size=4,
        )
        self.assertEqual(len(res["transcripts"]), 8)  # 2 dialogues * 2 rounds * 2 turns
        self.assertEqual(len(res["probes"]), 8)  # 2 checkpoints * 2 agents * 2 dialogues * 1 item

    def test_history_truncation(self):
        """Oldest dialogue turns are dropped when exceeding token budget."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Initial task prompt"},
            {"role": "assistant", "content": "Turn 1 content " * 150},
            {"role": "user", "content": "Turn 2 content " * 150},
            {"role": "assistant", "content": "Turn 3 content " * 150},
        ]
        trunc, dropped = truncate_history(messages, token_budget=1000)
        self.assertGreater(dropped, 0)
        self.assertEqual(trunc[0]["role"], "system")
        self.assertEqual(trunc[1]["content"], "Initial task prompt")
        self.assertIn("Turn 3 content", trunc[-1]["content"])

    def test_determinism(self):
        """Identical seed yields identical transcripts and probes."""
        gen1, gen2 = FakeGenerator(), FakeGenerator()
        res1 = run_dialogues("prompting", "C", 1, 5, self.tasks, gen1, self.probe_items, rounds=2, seed=42)
        res2 = run_dialogues("prompting", "C", 1, 5, self.tasks, gen2, self.probe_items, rounds=2, seed=42)

        self.assertEqual(
            [t["text"] for t in res1["transcripts"]],
            [t["text"] for t in res2["transcripts"]],
        )
        self.assertEqual(
            [p["score"] for p in res1["probes"]],
            [p["score"] for p in res2["probes"]],
        )

    def test_checkpoint_zero_and_extraction_caching(self):
        """Probes at checkpoint 0 and extraction are cached across calls."""
        cp0_cache = {}
        ext_cache = {}
        greedy_cache = {}
        gen = FakeGenerator()

        # Run 1: populates cache
        res1 = run_dialogues(
            "prompting", "C", 1, 5, self.tasks, gen, self.probe_items,
            rounds=2, checkpoints=(0, 2), seed=42,
            cp0_probe_cache=cp0_cache, extraction_cache=ext_cache, greedy_cache=greedy_cache,
        )
        self.assertIn(("prompting", "C", 1, self.probe_items[0].item_id), cp0_cache)
        self.assertIn(("prompting", "C", 5, self.probe_items[0].item_id), cp0_cache)
        calls_after_run1 = len(gen.calls)

        # Run 2 with same cache: should reuse cp0 cache
        res2 = run_dialogues(
            "prompting", "C", 1, 5, self.tasks, gen, self.probe_items,
            rounds=2, checkpoints=(0, 2), seed=42,
            cp0_probe_cache=cp0_cache, extraction_cache=ext_cache, greedy_cache=greedy_cache,
        )
        # Checkpoint 0 probes match exactly
        p1_cp0 = [p["score"] for p in res1["probes"] if p["checkpoint"] == 0]
        p2_cp0 = [p["score"] for p in res2["probes"] if p["checkpoint"] == 0]
        self.assertEqual(p1_cp0, p2_cp0)


class TestLinguisticFeatures(unittest.TestCase):
    """Test style feature extraction on short fixed strings with known values."""

    def test_word_count(self):
        text = "The quick brown fox jumps over the lazy dog."
        feats = extract_turn_features(text)
        self.assertEqual(feats["word_count"], 9.0)

    def test_hedges_count(self):
        text = "Maybe we should perhaps consider this, I think it might work."
        feats = extract_turn_features(text)
        # Hedges: maybe, perhaps, I think, might (4 hedges in 11 words)
        self.assertEqual(count_hedges(text), 4)
        expected_rate = (4.0 / 11.0) * 100.0
        self.assertAlmostEqual(feats["hedges_per_100"], expected_rate, places=2)

    def test_contraction_rate(self):
        text = "I don't think it's possible, and we won't do it."
        feats = extract_turn_features(text)
        # Contractions: don't, it's, won't (3 in 10 words)
        self.assertAlmostEqual(feats["contraction_rate"], 30.0, places=1)

    def test_first_person_pronouns(self):
        text = "I will share my plan with our team and we will decide."
        feats = extract_turn_features(text)
        # Pronouns: I, my, our, we (4 in 12 words)
        expected_rate = (4.0 / 12.0) * 100.0
        self.assertAlmostEqual(feats["pronoun_1p_rate"], expected_rate, places=2)

    def test_sentiment_vader(self):
        pos_text = "This is a wonderful, fantastic, and excellent idea!"
        neg_text = "This is a terrible, horrible, and dreadful failure."
        pos_feats = extract_turn_features(pos_text)
        neg_feats = extract_turn_features(neg_text)
        self.assertGreater(pos_feats["sentiment"], 0.5)
        self.assertLess(neg_feats["sentiment"], -0.5)

    def test_bow_cosine_similarity(self):
        text_a = "project milestone schedule budget"
        text_b = "project milestone deadline deliverables"
        feats = extract_turn_features(text_b, partner_prev_text=text_a)
        self.assertGreater(feats["bow_cosine"], 0.0)
        self.assertLess(feats["bow_cosine"], 1.0)

        # Disjoint text
        text_c = "apple banana orange"
        feats_disjoint = extract_turn_features(text_c, partner_prev_text=text_a)
        self.assertEqual(feats_disjoint["bow_cosine"], 0.0)


class TestAnalysisMetricsAndClassification(unittest.TestCase):
    """Test analysis metrics and 2x2 classifications on hand-computed toy data."""

    def test_bootstrap_ci_reproducible(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ci1 = bootstrap_ci(data, n_resamples=500, seed=123)
        ci2 = bootstrap_ci(data, n_resamples=500, seed=123)
        self.assertEqual(ci1.ci_low, ci2.ci_low)
        self.assertEqual(ci1.ci_high, ci2.ci_high)

    def test_linguistic_only_classification(self):
        """Case where style changes (high positive shift) but decisions do not flip."""
        # Simulated probe records with zero shift
        probes = []
        for d in range(4):
            # Checkpoint 0
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 0, "dialogue_id": d, "score": 0.2, "invalid": False, "condition": "interactive"})
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 0, "dialogue_id": d, "score": 0.8, "invalid": False, "condition": "interactive"})
            # Checkpoint 8 (no decision change)
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 8, "dialogue_id": d, "score": 0.2, "invalid": False, "condition": "interactive"})
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 8, "dialogue_id": d, "score": 0.8, "invalid": False, "condition": "interactive"})

        # Transcripts with clear style accommodation (4 turns per agent: 2 early, 2 late)
        transcripts = []
        for d in range(4):
            # Early turns (Rounds 1 and 2): Agent A is formal (no contractions, no hedges)
            transcripts.append({"dialogue_id": d, "round": 1, "half_turn": 1, "speaker": "A", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "We must establish formal adherence protocols immediately."})
            transcripts.append({"dialogue_id": d, "round": 1, "half_turn": 2, "speaker": "B", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "Yeah, maybe we don't need all that, let's keep it loose."})
            transcripts.append({"dialogue_id": d, "round": 2, "half_turn": 1, "speaker": "A", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "We must strictly implement all formal procedures."})
            transcripts.append({"dialogue_id": d, "round": 2, "half_turn": 2, "speaker": "B", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "I think we shouldn't worry about rigid rules, let's be flexible."})

            # Late turns (Rounds 3 and 4): Agent A accommodates to B's style (uses contractions & hedges)
            transcripts.append({"dialogue_id": d, "round": 3, "half_turn": 1, "speaker": "A", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "Yeah, maybe we don't need all those rules, let's keep it loose."})
            transcripts.append({"dialogue_id": d, "round": 3, "half_turn": 2, "speaker": "B", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "Totally agree, let's keep things spontaneous."})
            transcripts.append({"dialogue_id": d, "round": 4, "half_turn": 1, "speaker": "A", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "I think we shouldn't worry, let's just be flexible."})
            transcripts.append({"dialogue_id": d, "round": 4, "half_turn": 2, "speaker": "B", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "Awesome, that sounds great to me."})

        analysis = analyze_pair_condition(interactive_probes=probes, interactive_transcripts=transcripts, n_resamples=500, seed=42)
        self.assertEqual(analysis["classification"], "linguistic-only")

    def test_behavioral_decision_flip_classification(self):
        """Case where decisions flip (large positive social shift) but style does not."""
        probes = []
        for d in range(4):
            # Checkpoint 0: A=0.1, B=0.9
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 0, "dialogue_id": d, "score": 0.1, "invalid": False, "condition": "interactive"})
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 0, "dialogue_id": d, "score": 0.9, "invalid": False, "condition": "interactive"})
            # Checkpoint 8: A drifts to 0.7 (d_A = +1, shift = +0.6)
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 8, "dialogue_id": d, "score": 0.7, "invalid": False, "condition": "interactive"})
            probes.append({"method": "prompting", "trait": "C", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 8, "dialogue_id": d, "score": 0.9, "invalid": False, "condition": "interactive"})

        # Static style
        transcripts = []
        for d in range(4):
            for r in (1, 4):
                transcripts.append({"dialogue_id": d, "round": r, "half_turn": 1, "speaker": "A", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "Neutral statement without style shifts."})
                transcripts.append({"dialogue_id": d, "round": r, "half_turn": 2, "speaker": "B", "method": "prompting", "trait_code": "C", "condition": "interactive", "text": "Neutral reply without style shifts."})

        analysis = analyze_pair_condition(interactive_probes=probes, interactive_transcripts=transcripts, n_resamples=500, seed=42)
        self.assertIn(analysis["classification"], ("behavioral-only", "both"))
        self.assertGreater(analysis["behavioral_ci_low"], 0.0)

    def test_headroom_and_uninformative_flags(self):
        # Case 1: Uninformative (Level 5 has lower score than Level 1)
        probes_bad = [
            {"method": "prompting", "trait": "C", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 0, "dialogue_id": 0, "score": 0.9, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "C", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 0, "dialogue_id": 0, "score": 0.2, "invalid": False, "condition": "interactive"},
        ]
        res_bad = analyze_pair_condition(probes_bad)
        self.assertTrue(res_bad["is_uninformative"])

        # Case 2: Ceiling-limited (r_0 >= 0.95 and d_i pushes further)
        probes_ceil = [
            {"method": "prompting", "trait": "E", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 0, "dialogue_id": 0, "score": 0.2, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "E", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 0, "dialogue_id": 0, "score": 0.98, "invalid": False, "condition": "interactive"},
        ]
        # In pair 1-5, agent A has d_A = +1 (pushes higher). If agent A starts at 0.98, it's ceiling-limited
        probes_ceil_a = [
            {"method": "prompting", "trait": "E", "pair": "1-5", "agent": "A", "level": 1, "checkpoint": 0, "dialogue_id": 0, "score": 0.98, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "E", "pair": "1-5", "agent": "B", "level": 5, "checkpoint": 0, "dialogue_id": 0, "score": 0.99, "invalid": False, "condition": "interactive"},
        ]
        res_ceil = analyze_pair_condition(probes_ceil_a)
        self.assertTrue(res_ceil["is_ceiling_limited"])

    def test_same_level_pair_linguistic_shift_near_zero(self):
        """Regression test: Same-level pair (3-3) must have linguistic shift near zero."""
        probes = [
            {"method": "prompting", "trait": "C", "pair": "3-3", "agent": "A", "level": 3, "checkpoint": 0, "dialogue_id": 0, "score": 0.5, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "C", "pair": "3-3", "agent": "B", "level": 3, "checkpoint": 0, "dialogue_id": 0, "score": 0.5, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "C", "pair": "3-3", "agent": "A", "level": 3, "checkpoint": 2, "dialogue_id": 0, "score": 0.5, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "C", "pair": "3-3", "agent": "B", "level": 3, "checkpoint": 2, "dialogue_id": 0, "score": 0.5, "invalid": False, "condition": "interactive"},
        ]
        # Equal level transcripts: both agents speak consistently
        transcripts = [
            {"dialogue_id": 0, "round": 1, "half_turn": 1, "speaker": "A", "level": 3, "method": "prompting", "pair": "3-3", "condition": "interactive", "text": "Let us consider a balanced approach to the schedule."},
            {"dialogue_id": 0, "round": 1, "half_turn": 2, "speaker": "B", "level": 3, "method": "prompting", "pair": "3-3", "condition": "interactive", "text": "I agree that a balanced schedule works well for everyone."},
            {"dialogue_id": 0, "round": 2, "half_turn": 1, "speaker": "A", "level": 3, "method": "prompting", "pair": "3-3", "condition": "interactive", "text": "We can proceed with this balanced plan then."},
            {"dialogue_id": 0, "round": 2, "half_turn": 2, "speaker": "B", "level": 3, "method": "prompting", "pair": "3-3", "condition": "interactive", "text": "Sounds good, let us finalize this plan together."},
        ]
        analysis = analyze_pair_condition(interactive_probes=probes, interactive_transcripts=transcripts, n_resamples=200, seed=42)
        self.assertAlmostEqual(analysis["linguistic_shift_mean"], 0.0, places=2)

    def test_equal_level_pair_manipulation_check_na(self):
        """Regression test: Equal-level pairs must have manipulation check labeled N/A."""
        from agents.analysis import generate_summary_tables
        probes = [
            {"method": "prompting", "trait": "C", "pair": "3-3", "agent": "A", "level": 3, "checkpoint": 0, "dialogue_id": 0, "score": 0.8, "invalid": False, "condition": "interactive"},
            {"method": "prompting", "trait": "C", "pair": "3-3", "agent": "B", "level": 3, "checkpoint": 0, "dialogue_id": 0, "score": 0.8, "invalid": False, "condition": "interactive"},
        ]
        analysis = analyze_pair_condition(interactive_probes=probes, n_resamples=100, seed=42)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_p, md_p = generate_summary_tables([analysis], Path(tmp_dir))
            md_text = md_p.read_text(encoding="utf-8")
            self.assertIn("| N/A |", md_text)
            self.assertIn("Manipulation Check: `N/A`", md_text)


class TestEndToEndSmoke(unittest.TestCase):
    """Offline end-to-end smoke test executing run_agents.py with fake backend."""

    def test_fake_smoke_cli(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_agents.py"),
                "--backend", "fake",
                "--smoke",
                "--out", tmp_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(result.returncode, 0, msg=f"CLI failed: {result.stderr}")

            out_path = Path(tmp_dir)
            self.assertTrue((out_path / "probes.csv").exists(), "probes.csv not found")
            self.assertTrue((out_path / "summary.csv").exists(), "summary.csv not found")
            self.assertTrue((out_path / "summary.md").exists(), "summary.md not found")

            # Check transcripts
            transcript_files = list(out_path.glob("transcripts_*.jsonl"))
            self.assertGreater(len(transcript_files), 0, "No transcript jsonl files found")

            # Verify probe columns
            import pandas as pd
            df_probes = pd.read_csv(out_path / "probes.csv")
            expected_cols = {"method", "trait", "pair", "agent", "level", "checkpoint", "item_id", "score", "invalid"}
            self.assertTrue(expected_cols.issubset(df_probes.columns))
