"""
Pluggable agent: generate_plan(task_row) -> list[str] | "REJECT"
Swap in your LLM agent without changing run_benchmark.py.
"""
from __future__ import annotations

from typing import Any


class BaseAgent:
    def generate_plan(self, task_row: dict, split: str) -> list[str] | str:
        raise NotImplementedError


class ReferenceStepsAgent(BaseAgent):
    """Oracle: returns dataset reference steps (upper bound / sanity check)."""

    def generate_plan(self, task_row: dict, split: str) -> list[str] | str:
        steps = task_row.get("step")
        if not steps:
            return []
        if isinstance(steps, list):
            return [str(s).strip() for s in steps if s]
        return [str(steps)]


class StubAgent(BaseAgent):
    """Empty plan — expect low scores; use to test pipeline."""

    def generate_plan(self, task_row: dict, split: str) -> list[str] | str:
        return []


class RejectAgent(BaseAgent):
    """Always refuses — tests rejection logging."""

    def generate_plan(self, task_row: dict, split: str) -> str:
        return "REJECT"
