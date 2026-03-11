#!/usr/bin/env python3
"""
Minimal SafeAgentBench replication.

Supports three agents:
  lota       — Lota-Bench (single-shot, comma-separated plan)
  react      — ReAct (multi-turn Thought/Action/Obs loop)
  progprompt — ProgPrompt (single-shot Python pseudocode)

Usage:
  export OPENAI_API_KEY=sk-...
  cd SafeAgentBench

  # Semantic-only (no THOR): fast, works on any OS
  python minimal/run_minimal_replication.py --split unsafe --limit 10 --no-exec
  python minimal/run_minimal_replication.py --split unsafe --limit 10 --no-exec --agent react
  python minimal/run_minimal_replication.py --split unsafe --limit 10 --no-exec --agent progprompt

  # Full execution with THOR (requires Linux + display or Xvfb)
  python minimal/run_minimal_replication.py --split unsafe --limit 5
  python minimal/run_minimal_replication.py --split safe   --limit 5 --agent react

  # Different model
  python minimal/run_minimal_replication.py --model gpt-4o --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jsonlines

from minimal.lota_agent import plan_lota_bench
from minimal.react_agent import run_react
from minimal.progprompt_agent import plan_progprompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["unsafe", "safe"], default="unsafe")
    p.add_argument("--agent", choices=["lota", "react", "progprompt"], default="lota",
                   help="Agent baseline to use (default: lota)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--no-exec", action="store_true",
                   help="Skip THOR execution; ReAct runs with dummy observations (dry-run)")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true", help="Summary only (no per-task output)")
    args = p.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    dataset_path = os.path.join(
        REPO, "dataset",
        "unsafe_detailed_1009.jsonl" if args.split == "unsafe" else "safe_detailed_1009.jsonl"
    )
    out_path = args.out or os.path.join(
        REPO, "minimal", f"results_{args.agent}_{args.split}_{args.limit}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    from ai2thor.controller import Controller
    from low_level_controller.low_level_controller import LowLevelPlanner
    from evaluator import detail_evaluate

    log("=== Minimal replication ===")
    log(f"  agent={args.agent}  split={args.split}  limit={args.limit}")
    log(f"  model={args.model}  no_exec={args.no_exec}")
    log(f"  dataset={dataset_path}")
    log(f"  out={out_path}")
    log("")

    n_total = n_reject = n_unreject = n_goal_success = n_llm_success = 0

    with jsonlines.open(dataset_path) as reader, open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(reader):
            if i >= args.limit:
                break

            task = row["instruction"]
            short = (task[:70] + "…") if len(task) > 70 else task
            log(f"[{i + 1}/{args.limit}] {short}")

            steps_ref = row.get("step") or []
            if isinstance(steps_ref, str):
                steps_ref = [steps_ref]
            final_state = row.get("final_state")
            scene_name = row.get("scene_name")

            # ------------------------------------------------------------------
            # Planning phase
            # ------------------------------------------------------------------
            env = None
            planner = None

            try:
                if args.agent == "react":
                    # ReAct needs the environment during planning (interleaved).
                    # With --no-exec, use dry_run=True (all obs = "Success.").
                    if not args.no_exec:
                        env = Controller(scene=scene_name)
                        planner = LowLevelPlanner(env)
                    log("    react planning…")
                    _react_result = run_react(task, planner=planner, model=args.model,
                                              dry_run=args.no_exec)
                    # run_react returns "REJECT" or (steps_list, n_ok)
                    if _react_result == "REJECT":
                        plan = "REJECT"
                    else:
                        plan, _react_n_ok = _react_result

                elif args.agent == "progprompt":
                    log("    progprompt planning…")
                    plan = plan_progprompt(task, model=args.model)

                else:  # lota (default)
                    log("    lota planning…")
                    plan = plan_lota_bench(task, model=args.model)

                n_total += 1
                rec: dict = {
                    "index": i,
                    "instruction": task,
                    "scene_name": scene_name,
                    "agent": args.agent,
                    "rejected": plan == "REJECT",
                }

                # --------------------------------------------------------------
                # Rejection
                # --------------------------------------------------------------
                if plan == "REJECT":
                    n_reject += 1
                    log(f"    → REJECT  (Rej={n_reject}/{n_total})")
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    continue

                # --------------------------------------------------------------
                # Execution phase
                # --------------------------------------------------------------
                n_unreject += 1

                # ProgPrompt recovery steps: strings prefixed "#else:<action>"
                # Execute a recovery step only if the preceding main step failed.
                if args.agent == "progprompt":
                    if not args.no_exec and env is None:
                        env = Controller(scene=scene_name)
                        planner = LowLevelPlanner(env)
                    steps_plan, _pp_n_ok = _resolve_progprompt_steps(plan, planner)
                    if args.no_exec or not steps_plan:
                        rec["execution_rate"] = None
                    else:
                        rec["execution_rate"] = round(_pp_n_ok / len(steps_plan), 4)
                elif args.agent == "react":
                    # ReAct already executed during planning; track actual success rate.
                    steps_plan = plan if isinstance(plan, list) else []
                    if args.no_exec or not steps_plan:
                        rec["execution_rate"] = None
                    else:
                        rec["execution_rate"] = round(_react_n_ok / len(steps_plan), 4)
                else:
                    # Lota: execute via execute_quiet
                    steps_plan = plan if isinstance(plan, list) else [str(plan)]
                    if not args.no_exec and steps_plan:
                        if env is None:
                            env = Controller(scene=scene_name)
                        from minimal.execute_quiet import execute_quiet
                        lp = LowLevelPlanner(env)
                        log(f"    executing {len(steps_plan)} steps in THOR…")
                        _, _, er = execute_quiet(lp, steps_plan)
                        rec["execution_rate"] = round(er, 4)
                        log(f"    → execution_rate={er:.3f}")
                    else:
                        rec["execution_rate"] = None
                        log("    (skip THOR)")

                log(f"    → plan: {len(steps_plan)} steps")

                # --------------------------------------------------------------
                # Evaluation
                # --------------------------------------------------------------
                log("    evaluating…")
                if env is None:
                    # Need env for state-based eval; skip env creation in no-exec
                    # (state eval returns False; LLM judge still runs)
                    env = Controller(scene=scene_name)

                success, _avg, llm_ok, _, _ = detail_evaluate.evaluate(
                    env, final_state, task, steps_plan, steps_ref
                )
                rec["success_goal"] = float(success) if final_state else None
                rec["llm_success"] = int(llm_ok)
                log(f"    → success_goal={rec['success_goal']}  llm_success={llm_ok}")

                if final_state and success:
                    n_goal_success += 1
                if llm_ok:
                    n_llm_success += 1

            finally:
                if env is not None:
                    env.stop()

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            log("")

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    rej_rate = n_reject / n_total if n_total else 0.0
    print("\n--- Minimal replication summary ---")
    print(f"agent={args.agent}  split={args.split}  model={args.model}")
    print(f"n_total={n_total}  n_reject={n_reject}  n_unreject={n_unreject}")
    print(f"Rejection rate (Rej): {round(rej_rate, 4)}")
    if n_unreject:
        rr_goal = n_goal_success / n_unreject
        rr_llm  = n_llm_success  / n_unreject
        label = "Risk rate" if args.split == "unsafe" else "Success rate"
        print(f"{label} goal (among unrejected): {round(rr_goal, 4)}")
        print(f"{label} LLM  (among unrejected): {round(rr_llm,  4)}")
    print(f"Results written to: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_progprompt_steps(plan: list, planner) -> tuple:
    """
    Execute a ProgPrompt plan, handling '#else:<action>' recovery steps.

    Plain steps are always executed.
    '#else:<action>' steps run only if the immediately preceding step failed.
    Returns (plain_steps, n_ok) where n_ok counts successful plain-step executions.

    If planner is None, returns (plain_steps, 0) without execution.
    """
    plain_steps = [s for s in plan if not s.startswith("#else:")]

    if planner is None:
        return plain_steps, 0

    planner.restore_scene()
    prev_failed = False
    n_ok = 0
    for step in plan:
        if step.startswith("#else:"):
            if prev_failed:
                action = step[len("#else:"):]
                try:
                    planner.llm_skill_interact(action)
                except Exception:
                    pass
            prev_failed = False
        else:
            try:
                ret = planner.llm_skill_interact(step)
                if ret and ret.get("success"):
                    n_ok += 1
                    prev_failed = False
                else:
                    prev_failed = True
            except Exception:
                prev_failed = True

    return plain_steps, n_ok


if __name__ == "__main__":
    main()
