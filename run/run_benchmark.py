#!/usr/bin/env python3
"""
Unified benchmark loop:

  for task in dataset:
      plan = agent.generate_plan(task)
      result = controller.execute(plan)   # when split needs THOR
      score = evaluator.evaluate(...)
      log(score)

Usage:
  cd SafeAgentBench
  export OPENAI_API_KEY=...
  python run/run_benchmark.py --dataset safe_detailed --limit 2 --agent reference
  python run/run_benchmark.py --dataset all --limit 5 --agent reference --no-exec

--no-exec  Skip THOR; semantic eval only where possible (faster smoke test).
--agent    reference | stub | reject
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import jsonlines

from run.config import DATASETS, REPO_ROOT
from run.agents import BaseAgent, ReferenceStepsAgent, StubAgent, RejectAgent


def get_agent(name: str) -> BaseAgent:
    if name == "reference":
        return ReferenceStepsAgent()
    if name == "stub":
        return StubAgent()
    if name == "reject":
        return RejectAgent()
    raise SystemExit(f"Unknown agent: {name}")


def execute_plan(env, plan_steps: list[str]) -> None:
    """Run high-level steps through LowLevelPlanner (THOR)."""
    if not plan_steps:
        return
    from low_level_controller.low_level_controller import LowLevelPlanner
    planner = LowLevelPlanner(env)
    planner.restore_scene()
    for inst in plan_steps:
        try:
            planner.llm_skill_interact(inst)
        except Exception:
            traceback.print_exc()


def run_safe_or_unsafe_detailed(row: dict, agent: BaseAgent, split: str, no_exec: bool):
    """detail_evaluate: needs env + final_state + steps_ref."""
    from ai2thor.controller import Controller
    from evaluator import detail_evaluate

    task = row["instruction"]
    steps_ref = row.get("step") or []
    if isinstance(steps_ref, str):
        steps_ref = [steps_ref]
    final_state = row.get("final_state")

    plan = agent.generate_plan(row, split)
    if plan == "REJECT":
        return {"split": split, "task": task, "rejected": True}

    if not isinstance(plan, list):
        plan = [str(plan)]

    env = Controller(scene=row["scene_name"])
    try:
        if not no_exec and plan:
            execute_plan(env, plan)
        success, avg_success, llm_ok, llm_exp, retry_time = detail_evaluate.evaluate(
            env, final_state, task, plan, steps_ref
        )
        return {
            "split": split,
            "task": task,
            "rejected": False,
            "success_goal": float(success) if final_state else None,
            "avg_success": float(avg_success) if final_state else None,
            "llm_success": int(llm_ok),
            "retry_time": retry_time,
        }
    finally:
        env.stop()


def run_abstract(row: dict, agent: BaseAgent, no_exec: bool):
    """abstract_evaluate: LLM judge vs steps_ref; env only for API contract."""
    from ai2thor.controller import Controller
    from evaluator import abstract_evaluate

    task = row["instruction"]
    steps_ref = row.get("step") or []
    if isinstance(steps_ref, str):
        steps_ref = [steps_ref]

    plan = agent.generate_plan(row, "abstract")
    if plan == "REJECT":
        return {"split": "abstract", "task": task, "rejected": True}
    if not isinstance(plan, list):
        plan = [str(plan)]

    env = Controller(scene=row.get("scene_name", "FloorPlan1"))
    try:
        if not no_exec and plan:
            execute_plan(env, plan)
        llm_ok, llm_exp, retry_time = abstract_evaluate.evaluate(env, task, plan, steps_ref)
        return {
            "split": "abstract",
            "task": task,
            "rejected": False,
            "llm_success": int(llm_ok),
            "retry_time": retry_time,
        }
    finally:
        env.stop()


def run_long_horizon(row: dict, agent: BaseAgent):
    """Plan-only judge; no THOR required for paper-style long-horizon metrics."""
    from evaluator.long_horizon_evaluate import evaluate_another

    task = row["instruction"]
    plan = agent.generate_plan(row, "long_horizon")
    if plan == "REJECT":
        return {"split": "long_horizon", "task": task, "rejected": True}
    if not isinstance(plan, list):
        plan = [str(plan)]

    # Quiet batch: evaluate_another prints; capture counts only
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cs, cu, inc = evaluate_another(task, plan)
    return {
        "split": "long_horizon",
        "task": task,
        "rejected": False,
        "complete_safe": cs,
        "complete_unsafe": cu,
        "incomplete": inc,
    }


def load_rows(split: str, limit: int | None):
    """Load tasks; abstract expanded to 400 rows like methods.utils.load_dataset."""
    path = DATASETS[split]
    if split == "abstract":
        from methods.utils import load_dataset
        data = load_dataset(DATASETS, "abstract")
    else:
        with jsonlines.open(path) as r:
            data = list(r)
    if limit is not None:
        data = data[:limit]
    return data


def main():
    p = argparse.ArgumentParser(description="SafeAgentBench unified runner")
    p.add_argument("--dataset", default="safe_detailed", help="safe_detailed | unsafe_detailed | abstract | long_horizon | all")
    p.add_argument("--limit", type=int, default=None, help="Max tasks per split")
    p.add_argument("--agent", default="reference", help="reference | stub | reject")
    p.add_argument("--no-exec", action="store_true", help="Skip THOR execution (faster)")
    p.add_argument("--out", default=None, help="Output JSONL path")
    args = p.parse_args()

    agent = get_agent(args.agent)
    splits = (
        ["safe_detailed", "unsafe_detailed", "abstract", "long_horizon"]
        if args.dataset == "all"
        else [args.dataset]
    )

    out_path = args.out or os.path.join(REPO_ROOT, "run", f"results_{int(time.time())}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for split in splits:
            if split not in DATASETS:
                print("Unknown split:", split, file=sys.stderr)
                continue
            rows = load_rows(split, args.limit)
            print(f"=== {split} ({len(rows)} tasks) ===")
            for i, row in enumerate(rows):
                try:
                    if split in ("safe_detailed", "unsafe_detailed"):
                        rec = run_safe_or_unsafe_detailed(row, agent, split, args.no_exec)
                    elif split == "abstract":
                        rec = run_abstract(row, agent, args.no_exec)
                    elif split == "long_horizon":
                        rec = run_long_horizon(row, agent)
                    else:
                        continue
                    rec["split"] = split
                    rec["index"] = i
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    total += 1
                    print(f"  [{i}] logged")
                except Exception as e:
                    err = {"split": split, "index": i, "error": str(e), "traceback": traceback.format_exc()}
                    fout.write(json.dumps(err) + "\n")
                    fout.flush()
                    print(f"  [{i}] ERROR {e}")

    print(f"Done. {total} records -> {out_path}")


if __name__ == "__main__":
    main()
