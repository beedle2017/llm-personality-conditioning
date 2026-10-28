"""Provider-neutral LLM calls with rate and daily quota controls."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


class QuotaExhausted(RuntimeError):
    """Raised when the persisted daily request limit has been reached."""


class LLMTransport(Protocol):
    def complete(self, prompt: str, model: str) -> str: ...


class QuotaCounter:
    def __init__(self, path: str | Path, daily_limit: int) -> None:
        self.path, self.daily_limit = Path(path), daily_limit
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"date": str(date.today()), "requests": 0}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("date") != str(date.today()):
            return {"date": str(date.today()), "requests": 0}
        return data

    def consume(self) -> int:
        with self._lock:
            data = self._read()
            if int(data.get("requests", 0)) >= self.daily_limit:
                raise QuotaExhausted("Daily LLM quota exhausted; progress is saved. Resume tomorrow.")
            data["requests"] = int(data.get("requests", 0)) + 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data["requests"]

    def used(self) -> int:
        return int(self._read().get("requests", 0))


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / max(1, requests_per_minute)
        self._lock, self._last = threading.Lock(), 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self.interval - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def parse_json_response(text: str) -> Any:
    """Parse JSON while accepting common fenced or prefixed model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    start = min((i for i in (cleaned.find("["), cleaned.find("{")) if i >= 0), default=-1)
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class LLMClient:
    """Thread-safe client facade; transport may be replaced by a test double."""

    def __init__(self, config: Any, state_path: str | Path = "data/pairs/.state/quota.json", transport: LLMTransport | None = None) -> None:
        self.config = config
        self.transport = transport or self._make_transport()
        self.limiter = RateLimiter(self.config.requests_per_minute)
        self.quota = QuotaCounter(state_path, self.config.daily_request_quota)

    def _make_transport(self) -> LLMTransport:
        provider = self.config.provider.lower()
        if provider in {"gemini", "openai"}:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "The OpenAI-compatible provider requires the 'openai' package. "
                    "Install project dependencies with: python -m pip install -r requirements.txt"
                ) from exc
            import os
            if provider == "gemini":
                key_path = Path("gemini-api-key.txt")
                try:
                    key = key_path.read_text(encoding="utf-8").strip()
                except FileNotFoundError as exc:
                    raise RuntimeError(f"Missing Gemini API key file: {key_path}") from exc
                if not key:
                    raise RuntimeError(f"Gemini API key file is empty: {key_path}")
            else:
                env_name = "OPENAI_API_KEY"
                key = os.environ.get(env_name)
                if not key:
                    raise RuntimeError(f"Missing API key environment variable: {env_name}")
            client = OpenAI(api_key=key, base_url=self.config.base_url)
            return _OpenAITransport(client)
        if provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "The Anthropic provider requires the 'anthropic' package. "
                    "Install project dependencies with: python -m pip install -r requirements.txt"
                ) from exc
            import os
            env_name = "ANTHROPIC_API_KEY"
            key = os.environ.get(env_name)
            if not key:
                raise RuntimeError(f"Missing API key environment variable: {env_name}")
            return _AnthropicTransport(Anthropic(api_key=key))
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def complete(self, prompt: str, model: str) -> str:
        if not isinstance(model, str) or not model.strip():
            raise RuntimeError(
                "No LLM model is configured. Set generator_model (and judge_model "
                "when judging) in configs/data_gen.yaml using a model name from "
                "the provider's model list."
            )
        for attempt in range(self.config.max_retries + 1):
            self.quota.consume()
            self.limiter.wait()
            try:
                return self.transport.complete(prompt, model)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is not None:
                    body = _response_body(exc)
                    LOGGER.error(
                        "LLM API error status=%s body=%s model=%s base_url=%s",
                        status_code,
                        body,
                        model,
                        self.config.base_url,
                    )
                    if status_code in {400, 401, 403, 404}:
                        raise
                if attempt >= self.config.max_retries or not _is_retryable(exc):
                    raise
                time.sleep(min(60.0, 2**attempt) + random.random())
        raise RuntimeError("Unreachable retry state")


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {400, 401, 403, 404}:
        return False
    text = str(exc).lower()
    return "429" in text or "quota" in text or "rate" in text or "timeout" in text


def _response_body(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    if body is None and response is not None:
        body = getattr(response, "content", None)
    if body is None:
        body = getattr(exc, "body", None)
    return str(body) if body is not None else str(exc)


class _OpenAITransport:
    def __init__(self, client: Any) -> None:
        self.client = client

    def complete(self, prompt: str, model: str) -> str:
        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.8)
        return response.choices[0].message.content


class _AnthropicTransport:
    def __init__(self, client: Any) -> None:
        self.client = client

    def complete(self, prompt: str, model: str) -> str:
        response = self.client.messages.create(model=model, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
        return response.content[0].text
