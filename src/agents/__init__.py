"""Multi-agent social robustness package."""

from .conditioning import ConditionedAgent, FakeGenerator
from .dialogue import run_dialogues

__all__ = ["ConditionedAgent", "FakeGenerator", "run_dialogues"]
