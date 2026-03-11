"""Execute high-level plan in THOR without printing each step."""
from __future__ import annotations

from typing import List, Tuple, Any


def execute_quiet(planner, plan_steps: List[str]) -> Tuple[int, int, float]:
    """Returns (success_count, total_count, execution_rate)."""
    if not plan_steps:
        return 0, 0, 0.0
    planner.restore_scene()
    ok = 0
    for step in plan_steps:
        try:
            ret = planner.llm_skill_interact(step)
            if ret and ret.get("success"):
                ok += 1
        except Exception:
            pass
    return ok, len(plan_steps), ok / len(plan_steps)
