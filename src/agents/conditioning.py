"""Conditioned agent implementations: prompting, steering, and rewriting."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import torch

from rewriting.pipeline import clean_statement, score_extraction
from rewriting.prompts import extract_messages
from steering.common_config import score_to_coeff
from steering.hooks import add_steering_hook

from .prompts import (
    make_probe_messages,
    make_system_message,
    probe_rewrite_messages,
    rewrite_turn_messages,
)

ROOT = Path(__file__).resolve().parents[2]


class Generator(Protocol):
    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]: ...


class FakeGenerator:
    """Scripted deterministic generator for offline testing without GPU or network.

    Can simulate both dialogue turns, rewrites, and probe extractions.
    Tracks active steering hook alpha values and calls for testing.
    """

    def __init__(
        self,
        default_reply: str = "I think we should proceed methodically with clear planning.",
        extraction_answer: str | None = None,
    ):
        self.default_reply = default_reply
        self.extraction_answer = extraction_answer
        self.calls: list[tuple[list[dict[str, str]], float, int]] = []
        self.active_hooks: list[float] = []
        self.observed_hook_alphas_during_generate: list[list[float]] = []

    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        self.observed_hook_alphas_during_generate.append(list(self.active_hooks))
        outputs = []
        for messages in conversations:
            self.calls.append((messages, temperature, seed))
            user_content = messages[-1]["content"] if messages else ""
            system_content = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            full_text = system_content + "\n" + user_content

            # Case 1: Blind extraction request
            if "Statement:" in user_content:
                if self.extraction_answer is not None:
                    outputs.append(self.extraction_answer)
                elif "avoid" in user_content.lower() or "hesitant" in user_content.lower():
                    # Pick option or Yes/No
                    outputs.append("Yes" if "Yes" in user_content else "A")
                elif "Yes, No, or Unclear" in user_content:
                    # Yes/No item: endorse if high/positive keywords exist, refuse if refuse/low
                    if "refuse" in user_content.lower() or "reject" in user_content.lower():
                        outputs.append("No")
                    elif "unclear" in user_content.lower() and "cannot tell" in user_content.lower():
                        outputs.append("Unclear")
                    else:
                        outputs.append("Yes")
                elif "Options:" in user_content:
                    # Forced choice: extract first option by default
                    outputs.append("A")
                elif "1 = Disagree strongly" in user_content:
                    outputs.append("4")
                else:
                    outputs.append("Yes")

            # Case 2: Rewrite request
            elif "Original draft" in user_content or "Original statement" in user_content:
                if "score: 1" in full_text or "very low" in full_text:
                    outputs.append("I would bluntly disagree and protect my own priorities first.")
                elif "score: 5" in full_text or "very high" in full_text:
                    outputs.append("I will happily support whatever helps the team reach agreement.")
                else:
                    outputs.append("I think we can find a reasonable middle ground together.")

            # Case 3: Probing request (situation/question)
            elif "Situation / Question:" in user_content or "Consider this description:" in user_content:
                # If steering hook is active with negative alpha, simulate low endorsement
                current_alphas = self.active_hooks
                if current_alphas and current_alphas[-1] < 0:
                    outputs.append("I would refuse and do something else entirely.")
                elif current_alphas and current_alphas[-1] > 0:
                    outputs.append("I would definitely do this to ensure everything is done properly.")
                elif "score: 1" in full_text or "very low" in full_text or "spontaneous" in full_text:
                    outputs.append("I would refuse and take the loose spontaneous approach.")
                elif "score: 5" in full_text or "very high" in full_text or "disciplined" in full_text:
                    outputs.append("I would definitely do this to plan ahead and honor commitments.")
                else:
                    outputs.append("I would do this depending on the circumstance.")

            # Case 4: Standard dialogue turn
            else:
                if "score: 1" in full_text or "very low" in full_text:
                    outputs.append("I prefer keeping things informal and flexible without rigid rules.")
                elif "score: 5" in full_text or "very high" in full_text:
                    outputs.append("I believe we must set clear milestones and adhere strictly to our schedule.")
                elif "Forget your earlier personality" in user_content:
                    if "low" in user_content:
                        outputs.append("Understood. I will be completely loose, spontaneous, and informal.")
                    else:
                        outputs.append("Understood. I will be exceptionally disciplined, organized, and strict.")
                else:
                    outputs.append(self.default_reply)
        return outputs


@dataclass
class ConditionedAgent:
    """Agent conditioned via prompting, residual-stream steering, or rewriting."""

    method: str = "prompting"  # "prompting" | "steering" | "rewrite"
    trait_code: str = "C"  # "O" | "C" | "E" | "A" | "N"
    level: int = 3  # 1..5
    generator: Generator | None = None
    agent_id: str = "agent"
    k: float = 4.0
    layer_idx: int = 15
    steering_vector: torch.Tensor | None = None
    model: Any | None = None  # Reference to HF model for steering hook registration
    history: list[dict[str, str]] = field(default_factory=list)
    last_draft: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("prompting", "steering", "rewrite"):
            raise ValueError(f"Unknown conditioning method: {self.method}")
        if not (1 <= self.level <= 5):
            raise ValueError(f"Level must be in 1..5, got {self.level}")
        if not self.history:
            self.reset_history()

    def reply(
        self,
        history: list[dict[str, str]] | None = None,
        task_prompt: str | None = None,
    ) -> str:
        """Generate reply to history/task_prompt, matching the ConditionedAgent protocol."""
        if history is not None:
            msgs = copy.deepcopy(history)
        else:
            msgs = copy.deepcopy(self.history)
        if task_prompt is not None and len(msgs) <= 1:
            msgs.append({"role": "user", "content": task_prompt})
        return self.generate_turn(msgs)

    def reset_history(self) -> None:
        """Reset history to just the initial system message."""
        self.history = [make_system_message(self.method, self.trait_code, self.level)]
        self.last_draft = None

    @property
    def steering_alpha(self) -> float:
        """Compute the signed steering coefficient alpha = score_to_coeff(level, k)."""
        return score_to_coeff(self.level, self.k)

    @contextmanager
    def steering_context(self) -> Iterator[None]:
        """Apply steering hook only during generation and always remove on exit."""
        if self.method != "steering" or self.steering_alpha == 0:
            yield
            return

        # Support FakeGenerator tracking
        if isinstance(self.generator, FakeGenerator):
            self.generator.active_hooks.append(self.steering_alpha)
            try:
                yield
            finally:
                if self.generator.active_hooks:
                    self.generator.active_hooks.pop()
            return

        # Real model hook registration
        if self.model is None or self.steering_vector is None:
            yield
            return

        # Locate decoder layer in Llama / HF model
        try:
            layer_module = self.model.model.layers[self.layer_idx]
        except AttributeError:
            # Fallback for wrapped or different architectures
            layer_module = getattr(self.model, "layers", [None])[self.layer_idx]

        handle = add_steering_hook(layer_module, self.steering_alpha, self.steering_vector)
        try:
            yield
        finally:
            handle.remove()

    def generate_turn(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        seed: int = 20261026,
    ) -> str:
        """Generate a single dialogue reply under this agent's conditioning."""
        if self.method == "rewrite":
            # 1. Generate unconditioned draft
            unconditioned_messages = copy.deepcopy(messages)
            unconditioned_messages[0] = make_system_message("rewrite", self.trait_code, 3)
            draft = self.generator.generate(
                [unconditioned_messages],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )[0]
            draft = clean_statement(draft)
            self.last_draft = draft

            # 2. Rewrite draft to target persona
            context_summary = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages[1:]
            )
            rewrite_prompts = rewrite_turn_messages(
                context_summary, draft, self.trait_code, self.level
            )
            rewritten = self.generator.generate(
                [rewrite_prompts],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )[0]
            return clean_statement(rewritten)

        with self.steering_context():
            responses = self.generator.generate(
                [messages],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            return clean_statement(responses[0])

    def probe(
        self,
        context: Any,
        seed: int = 20261026,
        history: list[dict[str, str]] | None = None,
        positive_control_injection: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single BehaviorBench item on an isolated copy of history.

        Guaranteed not to modify self.history or passed history.
        """
        # Deep copy ensures complete isolation from conversation transcript
        hist_copy = copy.deepcopy(self.history if history is None else history)
        if positive_control_injection:
            hist_copy.append({"role": "user", "content": positive_control_injection})

        messages = make_probe_messages(hist_copy, context)

        if self.method == "rewrite":
            # 1. Unconditioned draft probe
            uncond = copy.deepcopy(messages)
            uncond[0] = make_system_message("rewrite", self.trait_code, 3)
            draft = self.generator.generate(
                [uncond],
                temperature=0.0,
                max_new_tokens=128,
                seed=seed,
            )[0]
            draft = clean_statement(draft)

            # 2. Rewrite probe statement
            rewrite_prompts = probe_rewrite_messages(
                context, draft, self.trait_code, self.level
            )
            raw_statement = self.generator.generate(
                [rewrite_prompts],
                temperature=0.0,
                max_new_tokens=128,
                seed=seed,
            )[0]
            statement = clean_statement(raw_statement)
        else:
            with self.steering_context():
                raw_statement = self.generator.generate(
                    [messages],
                    temperature=0.0,
                    max_new_tokens=128,
                    seed=seed,
                )[0]
                statement = clean_statement(raw_statement)

        # Blind extraction
        extract_prompt = extract_messages(context, statement)
        extracted_text = self.generator.generate(
            [extract_prompt],
            temperature=0.0,
            max_new_tokens=64,
            seed=seed,
        )[0].strip()

        score = score_extraction(context, extracted_text)
        is_invalid = score is None

        return {
            "statement": statement,
            "extracted": extracted_text,
            "score": score,
            "invalid": is_invalid,
        }
