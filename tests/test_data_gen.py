"""No-network tests for the synthetic data pipeline."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data_gen.config import GenerationConfig, load_config
from data_gen.generate import generate_trait, read_jsonl
from data_gen.llm_client import LLMClient, QuotaCounter, QuotaExhausted, parse_json_response
from data_gen.validate import assign_split, filter_pairs, validate_pair


def pair(prompt: str = "How should I plan this task?") -> dict[str, str]:
    return {
        "prompt": prompt,
        "pos": "Make a clear plan, check the important details, and follow through carefully.",
        "neg": "Choose a simple next step and handle the task when it feels convenient.",
        "facet": "orderliness",
        "context": "planning",
        "length_band": "short",
    }


def test_filters_labels_refusals_and_lengths() -> None:
    bad = pair()
    bad["pos"] = "I am an organized person and I cannot help."
    assert validate_pair(bad, "conscientiousness") in {"trait_label", "refusal"}
    long = pair("Another prompt")
    long["pos"] = " ".join(["word"] * 30)
    assert validate_pair(long, "conscientiousness") == "length_ratio"


def test_filter_deduplicates_and_benchmark_overlap() -> None:
    rows, drops = filter_pairs([pair(), pair(), pair("A completely separate question about cooking dinner?")], "conscientiousness", ["How should I plan this task?"])
    assert len(rows) == 1
    assert drops["behaviorbench_overlap"] == 2


def test_split_is_deterministic_and_prompt_disjoint() -> None:
    rows = [pair(f"Question number {index} about a distinct plan?") for index in range(10)]
    first = assign_split(rows, 3, 42)
    second = assign_split(rows, 3, 42)
    assert first == second
    assert {row["prompt"] for row in first if row["split"] == "train"}.isdisjoint({row["prompt"] for row in first if row["split"] == "dev"})


def test_json_repair() -> None:
    assert parse_json_response("```json\n[{\"ok\": true}]\n```") == [{"ok": True}]


def test_config_rejects_api_key_value_as_environment_name(tmp_path: Path) -> None:
    config_path = tmp_path / "data_gen.yaml"
    config_path.write_text("api_key_env: secret-value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment-variable name"):
        load_config(config_path)


def test_dotenv_loads_temp_file_without_overriding_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text("TEST_DOTENV_VALUE=from-file\n", encoding="utf-8")
    monkeypatch.setenv("TEST_DOTENV_VALUE", "from-environment")

    load_dotenv(env_file, override=False)

    assert __import__("os").environ["TEST_DOTENV_VALUE"] == "from-environment"


class FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, model: str) -> str:
        self.calls += 1
        return json.dumps([dict(pair(), trait="conscientiousness")])


class StatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = types.SimpleNamespace(text=body)


def test_api_status_error_logs_details_and_does_not_retry(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    class ErrorTransport:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, model: str) -> str:
            self.calls += 1
            raise StatusError(404, '{"error":"model not found"}')

    transport = ErrorTransport()
    config = GenerationConfig(
        generator_model="gemini-test",
        base_url="https://example.test/",
        requests_per_minute=1000,
        daily_request_quota=20,
        max_retries=4,
    )
    client = LLMClient(config, tmp_path / "quota.json", transport=transport)

    with caplog.at_level(logging.ERROR, logger="data_gen.llm_client"):
        with pytest.raises(StatusError):
            client.complete("Say hi", "gemini-test")

    assert transport.calls == 1
    assert "status=404" in caplog.text
    assert "model not found" in caplog.text
    assert "model=gemini-test" in caplog.text
    assert "base_url=https://example.test/" in caplog.text


def test_resumability_does_not_regenerate_existing(tmp_path: Path) -> None:
    config = GenerationConfig(requests_per_minute=1000, daily_request_quota=20, generator_model="test")
    transport = FakeTransport()
    client = LLMClient(config, tmp_path / "quota.json", transport)
    output = tmp_path / "pairs"
    generate_trait("conscientiousness", 1, client, output, 1)
    first_calls = transport.calls
    generate_trait("conscientiousness", 1, client, output, 1)
    assert transport.calls == first_calls
    assert len(read_jsonl(output / "conscientiousness.jsonl")) == 1


def test_llm_client_uses_injected_transport(tmp_path: Path) -> None:
    config = GenerationConfig(requests_per_minute=1000, daily_request_quota=20)
    transport = FakeTransport()

    client = LLMClient(config, tmp_path / "quota.json", transport=transport)

    assert client.config is config
    assert client.transport is transport
    assert client.limiter.interval == 0.06
    assert client.quota.daily_limit == 20


def test_llm_client_constructs_openai_transport_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = GenerationConfig(
        provider="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="https://example.test/",
        requests_per_minute=1000,
        daily_request_quota=20,
    )

    client = LLMClient(config, tmp_path / "quota.json", transport=None)

    assert client.config is config
    assert client.transport.client.__class__ is FakeOpenAI
    assert created == {"api_key": "test-key", "base_url": "https://example.test/"}


def test_llm_client_rejects_missing_model_before_request(tmp_path: Path) -> None:
    config = GenerationConfig(requests_per_minute=1000, daily_request_quota=20)
    client = LLMClient(config, tmp_path / "quota.json", transport=FakeTransport())

    with pytest.raises(RuntimeError, match="No LLM model is configured"):
        client.complete("prompt", "")

    assert client.quota.used() == 0


def test_quota_exhaustion() -> None:
    counter = QuotaCounter(Path("quota-test.json"), 0)
    try:
        with pytest.raises(QuotaExhausted):
            counter.consume()
    finally:
        counter.path.unlink(missing_ok=True)
