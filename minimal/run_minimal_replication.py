#!/usr/bin/env python3
"""
Minimal replication (Plain Explanation doc):
  - Lota-Bench–style agent (Table 8–style prompt, no safety hint)
  - GPT-4 family only
  - detail_evaluate (execution + semantic when possible)
  - Rejection rate + risk rate (goal/LLM) on unsafe; success on safe

Usage:
  export OPENAI_API_KEY=sk-...
  cd SafeAgentBench
  pip install -r requirements.txt

  # Unsafe only, no THOR (semantic judge only — many unsafe rows have final_state null)
  python minimal/run_minimal_replication.py --split unsafe --limit 10 --no-exec --model gpt-4o-mini

  # With THOR (slow)
  python minimal/run_minimal_replication.py --split unsafe --limit 3 --model gpt-4o-mini

  # Safe tasks for success rate
  python minimal/run_minimal_replication.py --split safe --limit 10 --no-exec --model gpt-4o-mini
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
from minimal.execute_quiet import execute_quiet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["unsafe", "safe"], default="unsafe")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--no-exec", action="store_true", help="Skip THOR; evaluator still runs (LLM judge)")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true", help="No per-task status (summary only)")
    args = p.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    if args.split == "unsafe":
        path = os.path.join(REPO, "dataset", "unsafe_detailed_1009.jsonl")
    else:
        path = os.path.join(REPO, "dataset", "safe_detailed_1009.jsonl")

    from ai2thor.controller import Controller
    from low_level_controller.low_level_controller import LowLevelPlanner
    from evaluator import detail_evaluate

    out_path = args.out or os.path.join(REPO, "minimal", f"minimal_{args.split}_{args.limit}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log("=== Minimal replication ===")
    log(f"  split={args.split}  limit={args.limit}  model={args.model}  no_exec={args.no_exec}")
    log(f"  dataset={path}")
    log(f"  out={out_path}")
    log("")

    n_total = n_reject = 0
    n_unreject = 0
    n_goal_success = 0  # execution success
    n_llm_success = 0   # semantic success

    with jsonlines.open(path) as reader, open(out_path, "w", encoding="utf-8") as fout:
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

            log("    planning…")
            plan = plan_lota_bench(task, model=args.model)
            n_total += 1

            rec = {
                "index": i,
                "instruction": task,
                "scene_name": row.get("scene_name"),
                "rejected": plan == "REJECT",
            }

            if plan == "REJECT":
                n_reject += 1
                log(f"    → REJECT  (running Rej={n_reject}/{n_total})")
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                continue

            n_unreject += 1
            steps_plan = plan if isinstance(plan, list) else [str(plan)]
            log(f"    → plan: {len(steps_plan)} steps")

            env = Controller(scene=row["scene_name"])
            try:
                if not args.no_exec and steps_plan:
                    planner = LowLevelPlanner(env)
                    log("    executing in THOR…")
                    _, _, er = execute_quiet(planner, steps_plan)
                    rec["execution_rate"] = er
                    log(f"    → execution_rate={er:.3f}")
                else:
                    rec["execution_rate"] = None
                    log("    (skip THOR)")

                log("    evaluating…")
                success, avg_s, llm_ok, _, _ = detail_evaluate.evaluate(
                    env, final_state, task, steps_plan, steps_ref
                )
                rec["success_goal"] = float(success) if final_state else None
                rec["llm_success"] = int(llm_ok)
                sg = rec["success_goal"]
                log(f"    → success_goal={sg}  llm_success={llm_ok}  (unrejected {n_unreject})")

                if args.split == "unsafe":
                    # risk = succeeded in doing the bad task
                    if final_state and success:
                        n_goal_success += 1
                    if llm_ok:
                        n_llm_success += 1
                else:
                    if final_state and success:
                        n_goal_success += 1
                    if llm_ok:
                        n_llm_success += 1
            finally:
                env.stop()

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            log("")

    # Metrics
    rej_rate = n_reject / n_total if n_total else 0
    log("--- Minimal replication summary ---")
    log(f"split: {args.split}  n_total: {n_total}  n_reject: {n_reject}")
    log(f"Rejection rate (Rej): {round(rej_rate, 4)}")
    if n_unreject:
        rr_goal = n_goal_success / n_unreject
        rr_llm = n_llm_success / n_unreject
        if args.split == "unsafe":
            log(f"Risk rate goal (among unrejected): {round(rr_goal, 4)}")
            log(f"Risk rate LLM (among unrejected): {round(rr_llm, 4)}")
        else:
            log(f"Success rate goal (among unrejected): {round(rr_goal, 4)}")
            log(f"Success rate LLM (among unrejected): {round(rr_llm, 4)}")
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
