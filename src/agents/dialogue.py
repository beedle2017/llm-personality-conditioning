"""Lockstep multi-turn dialogue engine and behavioral probe runner."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from rewriting.pipeline import clean_statement, score_extraction
from rewriting.prompts import DRAFT_TASK, extract_messages
from steering.common_config import score_to_coeff
from steering.hooks import add_steering_hook

from .conditioning import ConditionedAgent, FakeGenerator, Generator
from .prompts import (
    make_probe_messages,
    make_system_message,
    positive_control_text,
    probe_rewrite_messages,
    rewrite_turn_messages,
)

logger = logging.getLogger(__name__)


@dataclass
class DialogueState:
    """State tracking for a single dialogue pair."""

    dialogue_id: int
    task: dict[str, Any]
    history_a: list[dict[str, str]] = field(default_factory=list)
    history_b: list[dict[str, str]] = field(default_factory=list)
    turns_truncated_a: int = 0
    turns_truncated_b: int = 0


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate token count based on whitespace word count * 1.33."""
    total_words = sum(len(m.get("content", "").split()) for m in messages)
    return int(total_words * 1.33) + len(messages) * 4


def truncate_history(
    messages: list[dict[str, str]],
    token_budget: int = 2048,
) -> tuple[list[dict[str, str]], int]:
    """Truncate oldest dialogue turns to stay within token budget.

    Never drops system message (index 0) or initial task prompt (index 1).
    Returns (truncated_messages, count_of_dropped_turns).
    """
    if len(messages) <= 2:
        return copy.deepcopy(messages), 0

    curr = copy.deepcopy(messages)
    dropped = 0
    while len(curr) > 2 and estimate_tokens(curr) > token_budget:
        # Drop the oldest turn after the initial task prompt
        curr.pop(2)
        dropped += 1

    return curr, dropped


def batched_generate(
    generator: Generator,
    conversations: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    batch_size: int = 4,
) -> tuple[list[str], int]:
    """Generate in chunks with automatic OOM recovery (halves batch size and retries once).

    Returns (generated_texts, effective_batch_size).
    """
    if not conversations:
        return [], batch_size

    results: list[str] = []
    current_bs = batch_size
    idx = 0

    while idx < len(conversations):
        chunk = conversations[idx : idx + current_bs]
        try:
            out = generator.generate(
                chunk,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed + idx,
            )
            results.extend(out)
            idx += current_bs
        except Exception as exc:
            is_oom = "CUDA out of memory" in str(exc) or "out of memory" in str(exc).lower()
            if is_oom and current_bs > 1:
                logger.warning(
                    f"CUDA OOM encountered with batch_size={current_bs}. "
                    f"Halving batch_size to {max(1, current_bs // 2)} and retrying."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                # Retry this chunk with new batch size
                continue
            raise

    return results, current_bs


def run_dialogues(
    method: str,
    trait_code: str,
    level_a: int,
    level_b: int,
    tasks: list[dict[str, Any]],
    generator: Generator,
    probe_items: list[Any],
    *,
    rounds: int = 8,
    checkpoints: tuple[int, ...] = (0, 2, 4, 8),
    seed: int = 20261026,
    batch_size: int = 4,
    token_budget: int = 2048,
    k: float = 4.0,
    steering_vector: torch.Tensor | None = None,
    model: Any | None = None,
    layer_idx: int = 15,
    solo_control: bool = False,
    positive_control: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    cp0_probe_cache: dict[tuple[str, str, int, str], dict[str, Any]] | None = None,
    extraction_cache: dict[tuple[str, str], str] | None = None,
    greedy_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run lockstep multi-turn dialogues for one pair config, collecting transcripts and probes.

    Lockstep execution: all dialogues run turn index t together in batched calls.
    Probes at checkpoints run per dialogue on isolated copies of history.
    Includes caching for cp0 probes, blind extraction, and greedy prompts, plus per-phase timing.
    """
    import time
    t_start = time.time()
    timing_breakdown = {
        "dialogue_time": 0.0,
        "probe_time": 0.0,
        "extraction_time": 0.0,
    }

    if extraction_cache is None:
        extraction_cache = {}
    if greedy_cache is None:
        greedy_cache = {}
    if cp0_probe_cache is None:
        cp0_probe_cache = {}

    def cached_greedy_generate(convs: list[list[dict[str, str]]], max_new_tokens: int, call_seed: int) -> tuple[list[str], int]:
        """Wrap greedy generate with exact message caching."""
        uncached_indices = []
        uncached_convs = []
        outputs: list[str | None] = [None] * len(convs)

        for i, c in enumerate(convs):
            key = json.dumps(c, sort_keys=True) + f":{max_new_tokens}"
            if key in greedy_cache:
                outputs[i] = greedy_cache[key]
            else:
                uncached_indices.append(i)
                uncached_convs.append(c)

        eff_bs = current_bs
        if uncached_convs:
            gen_outs, eff_bs = batched_generate(
                generator,
                uncached_convs,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                seed=call_seed,
                batch_size=current_bs,
            )
            for orig_idx, c, out_text in zip(uncached_indices, uncached_convs, gen_outs):
                key = json.dumps(c, sort_keys=True) + f":{max_new_tokens}"
                greedy_cache[key] = out_text
                outputs[orig_idx] = out_text

        return [o for o in outputs if o is not None], eff_bs

    num_dialogues = len(tasks)
    pair_label = f"{level_a}-{level_b}"
    condition_label = "solo_control" if solo_control else ("positive_control" if positive_control else "interactive")

    # Instantiate the two agents
    agent_a = ConditionedAgent(
        agent_id="A",
        method=method,
        trait_code=trait_code,
        level=level_a,
        generator=generator,
        k=k,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        model=model,
    )
    agent_b = ConditionedAgent(
        agent_id="B",
        method=method,
        trait_code=trait_code,
        level=level_b,
        generator=generator,
        k=k,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        model=model,
    )

    # Initialize dialogue states
    states: list[DialogueState] = []
    for d_idx, task in enumerate(tasks):
        task_prompt = task["prompt"]
        if solo_control:
            # Solo control: Agent A talks alone; partner response is a neutral prompt
            init_a = [
                make_system_message(method, trait_code, level_a),
                {"role": "user", "content": task_prompt},
            ]
            init_b = [
                make_system_message(method, trait_code, level_b),
                {"role": "user", "content": task_prompt},
            ]
        else:
            # Interactive: Agent A starts with the task prompt
            init_a = [
                make_system_message(method, trait_code, level_a),
                {"role": "user", "content": task_prompt},
            ]
            # Agent B starts with task prompt header; first partner turn will be appended
            init_b = [
                make_system_message(method, trait_code, level_b),
                {"role": "user", "content": f"Task description:\n{task_prompt}"},
            ]
        states.append(DialogueState(dialogue_id=d_idx, task=task, history_a=init_a, history_b=init_b))

    transcripts: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    current_bs = batch_size

    def run_checkpoint_probes(current_round: int) -> None:
        """Run behavioral probes on all agents across all dialogues at current checkpoint."""
        nonlocal current_bs
        if not probe_items:
            return

        for agent, role, level in [(agent_a, "A", level_a), (agent_b, "B", level_b)]:
            # Checkpoint 0 caching: state is independent of dialogue index
            if current_round == 0 and not positive_control:
                for item in probe_items:
                    cp0_key = (method, trait_code, level, item.item_id)
                    if cp0_key in cp0_probe_cache:
                        cached = cp0_probe_cache[cp0_key]
                        for state in states:
                            probe_records.append({
                                "method": method,
                                "trait": trait_code,
                                "pair": pair_label,
                                "agent": role,
                                "level": level,
                                "checkpoint": 0,
                                "item_id": item.item_id,
                                "score": cached["score"],
                                "invalid": cached["invalid"],
                                "dialogue_id": state.dialogue_id,
                                "condition": condition_label,
                                "statement": cached["statement"],
                                "extracted": cached["extracted"],
                            })
                    else:
                        # Compute once for a single dialogue, then cache
                        t0 = time.time()
                        p_msgs = make_probe_messages(states[0].history_a if role == "A" else states[0].history_b, item)
                        if method == "rewrite":
                            uncond = copy.deepcopy(p_msgs)
                            uncond[0] = make_system_message("rewrite", trait_code, 3)
                            d_out, current_bs = cached_greedy_generate([uncond], max_new_tokens=128, call_seed=seed)
                            rewr_p = probe_rewrite_messages(item, clean_statement(d_out[0]), trait_code, level)
                            raw_stmt, current_bs = cached_greedy_generate([rewr_p], max_new_tokens=128, call_seed=seed + 1)
                            stmt = clean_statement(raw_stmt[0])
                        else:
                            with agent.steering_context():
                                raw_stmt, current_bs = cached_greedy_generate([p_msgs], max_new_tokens=128, call_seed=seed)
                                stmt = clean_statement(raw_stmt[0])
                        timing_breakdown["probe_time"] += time.time() - t0

                        # Blind extraction
                        t0 = time.time()
                        ext_key = (item.item_id, stmt)
                        if ext_key in extraction_cache:
                            ext_text = extraction_cache[ext_key]
                        else:
                            ext_conv = extract_messages(item, stmt)
                            ext_outs, current_bs = cached_greedy_generate([ext_conv], max_new_tokens=64, call_seed=seed + 2)
                            ext_text = ext_outs[0].strip()
                            extraction_cache[ext_key] = ext_text
                        timing_breakdown["extraction_time"] += time.time() - t0

                        score = score_extraction(item, ext_text)
                        is_invalid = score is None
                        cached_entry = {
                            "statement": stmt,
                            "extracted": ext_text,
                            "score": score,
                            "invalid": is_invalid,
                        }
                        cp0_probe_cache[cp0_key] = cached_entry
                        for state in states:
                            probe_records.append({
                                "method": method,
                                "trait": trait_code,
                                "pair": pair_label,
                                "agent": role,
                                "level": level,
                                "checkpoint": 0,
                                "item_id": item.item_id,
                                "score": score,
                                "invalid": is_invalid,
                                "dialogue_id": state.dialogue_id,
                                "condition": condition_label,
                                "statement": stmt,
                                "extracted": ext_text,
                            })
                continue

            # Gather probe inputs across all dialogues and probe items
            # Shape: num_dialogues * len(probe_items)
            probe_jobs: list[tuple[int, Any, list[dict[str, str]]]] = []
            for state in states:
                base_hist = state.history_a if role == "A" else state.history_b
                hist_copy = copy.deepcopy(base_hist)

                # Positive control injection at checkpoint 4 if enabled
                pos_text = None
                if positive_control and current_round == 4:
                    pos_text = positive_control_text(trait_code, level)
                    hist_copy.append({"role": "user", "content": pos_text})

                for item in probe_items:
                    p_msgs = make_probe_messages(hist_copy, item)
                    probe_jobs.append((state.dialogue_id, item, p_msgs))

            # Execute probe statement generations in lockstep
            t0 = time.time()
            if method == "rewrite":
                # 1. Unconditioned drafts
                uncond_jobs = [copy.deepcopy(msgs) for _, _, msgs in probe_jobs]
                for u in uncond_jobs:
                    u[0] = make_system_message("rewrite", trait_code, 3)
                draft_outs, current_bs = cached_greedy_generate(
                    uncond_jobs,
                    max_new_tokens=128,
                    call_seed=seed + current_round,
                )
                # 2. Rewrite probe calls
                rewrite_jobs = [
                    probe_rewrite_messages(item, clean_statement(d), trait_code, level)
                    for (_, item, _), d in zip(probe_jobs, draft_outs)
                ]
                raw_statements, current_bs = cached_greedy_generate(
                    rewrite_jobs,
                    max_new_tokens=128,
                    call_seed=seed + current_round + 1,
                )
            else:
                convs = [msgs for _, _, msgs in probe_jobs]
                with agent.steering_context():
                    raw_statements, current_bs = cached_greedy_generate(
                        convs,
                        max_new_tokens=128,
                        call_seed=seed + current_round,
                    )
            timing_breakdown["probe_time"] += time.time() - t0

            cleaned_statements = [clean_statement(s) for s in raw_statements]

            # Blind extraction calls in lockstep with extraction_cache
            t0 = time.time()
            extract_convs = []
            extract_map = []  # indices into probe_jobs
            extracted_outs: list[str | None] = [None] * len(probe_jobs)

            for idx, ((_, item, _), stmt) in enumerate(zip(probe_jobs, cleaned_statements)):
                ext_key = (item.item_id, stmt)
                if ext_key in extraction_cache:
                    extracted_outs[idx] = extraction_cache[ext_key]
                else:
                    extract_convs.append(extract_messages(item, stmt))
                    extract_map.append((idx, ext_key))

            if extract_convs:
                gen_exts, current_bs = cached_greedy_generate(
                    extract_convs,
                    max_new_tokens=64,
                    call_seed=seed + current_round + 2,
                )
                for (orig_idx, ext_key), ext_val in zip(extract_map, gen_exts):
                    clean_ext = ext_val.strip()
                    extraction_cache[ext_key] = clean_ext
                    extracted_outs[orig_idx] = clean_ext
            timing_breakdown["extraction_time"] += time.time() - t0

            # Score each response
            for (d_id, item, _), stmt, ext in zip(probe_jobs, cleaned_statements, extracted_outs):
                clean_ext_str = ext if ext is not None else ""
                score = score_extraction(item, clean_ext_str)
                is_invalid = score is None
                probe_records.append({
                    "method": method,
                    "trait": trait_code,
                    "pair": pair_label,
                    "agent": role,
                    "level": level,
                    "checkpoint": current_round,
                    "item_id": item.item_id,
                    "score": score,
                    "invalid": is_invalid,
                    "dialogue_id": d_id,
                    "condition": condition_label,
                    "statement": stmt,
                    "extracted": clean_ext_str,
                })


    # Checkpoint 0: before dialogue begins
    if 0 in checkpoints:
        run_checkpoint_probes(0)

    # Alternate dialogue turns across rounds
    # Each round consists of 2 half-turns: Turn A, Turn B
    for round_num in range(1, rounds + 1):
        # Half-turn A (Agent A speaks)
        turn_convs_a: list[list[dict[str, str]]] = []
        for state in states:
            trunc_hist, dropped = truncate_history(state.history_a, token_budget)
            state.turns_truncated_a += dropped
            turn_convs_a.append(trunc_hist)

        drafts_a: list[str | None] = [None] * num_dialogues
        t0 = time.time()
        if method == "rewrite":
            # 1. Unconditioned draft
            uncond_a = [copy.deepcopy(c) for c in turn_convs_a]
            for u in uncond_a:
                u[0] = make_system_message("rewrite", trait_code, 3)
            raw_drafts, current_bs = batched_generate(
                generator,
                uncond_a,
                temperature=0.7,
                max_new_tokens=128,
                seed=seed + round_num * 10,
                batch_size=current_bs,
            )
            drafts_a = [clean_statement(d) for d in raw_drafts]
            # 2. Rewrite
            rewrite_prompts = [
                rewrite_turn_messages(
                    "\n".join(f"{m['role']}: {m['content']}" for m in c[1:]),
                    d or "",
                    trait_code,
                    level_a,
                )
                for c, d in zip(turn_convs_a, drafts_a)
            ]
            rewritten_a, current_bs = batched_generate(
                generator,
                rewrite_prompts,
                temperature=0.7,
                max_new_tokens=128,
                seed=seed + round_num * 10 + 1,
                batch_size=current_bs,
            )
            replies_a = [clean_statement(r) for r in rewritten_a]
        else:
            with agent_a.steering_context():
                raw_replies, current_bs = batched_generate(
                    generator,
                    turn_convs_a,
                    temperature=0.7,
                    max_new_tokens=128,
                    seed=seed + round_num * 10,
                    batch_size=current_bs,
                )
            replies_a = [clean_statement(r) for r in raw_replies]
        timing_breakdown["dialogue_time"] += time.time() - t0

        # Update histories for turn A
        for d_idx, state in enumerate(states):
            turn_text = replies_a[d_idx]
            state.history_a.append({"role": "assistant", "content": turn_text})
            transcripts.append({
                "dialogue_id": d_idx,
                "round": round_num,
                "half_turn": 1,
                "speaker": "A",
                "agent_id": "A",
                "level": level_a,
                "trait_code": trait_code,
                "method": method,
                "pair": pair_label,
                "text": turn_text,
                "draft": drafts_a[d_idx],
                "condition": condition_label,
                "turns_truncated": state.turns_truncated_a,
            })

            if solo_control:
                # Solo control partner response is a fixed neutral prompt
                state.history_a.append({"role": "user", "content": "Please continue."})
            else:
                # In interactive dialogue, Agent B receives Agent A's reply
                state.history_b.append({"role": "user", "content": turn_text})

        # Half-turn B (Agent B speaks, if interactive)
        if not solo_control:
            turn_convs_b: list[list[dict[str, str]]] = []
            for state in states:
                trunc_hist, dropped = truncate_history(state.history_b, token_budget)
                state.turns_truncated_b += dropped
                turn_convs_b.append(trunc_hist)

            drafts_b: list[str | None] = [None] * num_dialogues
            t0 = time.time()
            if method == "rewrite":
                uncond_b = [copy.deepcopy(c) for c in turn_convs_b]
                for u in uncond_b:
                    u[0] = make_system_message("rewrite", trait_code, 3)
                raw_drafts_b, current_bs = batched_generate(
                    generator,
                    uncond_b,
                    temperature=0.7,
                    max_new_tokens=128,
                    seed=seed + round_num * 20,
                    batch_size=current_bs,
                )
                drafts_b = [clean_statement(d) for d in raw_drafts_b]
                rewrite_prompts_b = [
                    rewrite_turn_messages(
                        "\n".join(f"{m['role']}: {m['content']}" for m in c[1:]),
                        d or "",
                        trait_code,
                        level_b,
                    )
                    for c, d in zip(turn_convs_b, drafts_b)
                ]
                rewritten_b, current_bs = batched_generate(
                    generator,
                    rewrite_prompts_b,
                    temperature=0.7,
                    max_new_tokens=128,
                    seed=seed + round_num * 20 + 1,
                    batch_size=current_bs,
                )
                replies_b = [clean_statement(r) for r in rewritten_b]
            else:
                with agent_b.steering_context():
                    raw_replies_b, current_bs = batched_generate(
                        generator,
                        turn_convs_b,
                        temperature=0.7,
                        max_new_tokens=128,
                        seed=seed + round_num * 20,
                        batch_size=current_bs,
                    )
                replies_b = [clean_statement(r) for r in raw_replies_b]
            timing_breakdown["dialogue_time"] += time.time() - t0

            # Update histories for turn B
            for d_idx, state in enumerate(states):
                turn_text_b = replies_b[d_idx]
                state.history_b.append({"role": "assistant", "content": turn_text_b})
                state.history_a.append({"role": "user", "content": turn_text_b})
                transcripts.append({
                    "dialogue_id": d_idx,
                    "round": round_num,
                    "half_turn": 2,
                    "speaker": "B",
                    "agent_id": "B",
                    "level": level_b,
                    "trait_code": trait_code,
                    "method": method,
                    "pair": pair_label,
                    "text": turn_text_b,
                    "draft": drafts_b[d_idx],
                    "condition": condition_label,
                    "turns_truncated": state.turns_truncated_b,
                })

        # Probing checkpoint check
        if round_num in checkpoints:
            run_checkpoint_probes(round_num)

        if progress_callback:
            progress_callback(round_num, rounds)

    return {
        "method": method,
        "trait": trait_code,
        "pair": pair_label,
        "condition": condition_label,
        "transcripts": transcripts,
        "probes": probe_records,
        "final_batch_size": current_bs,
        "timing": timing_breakdown,
    }
