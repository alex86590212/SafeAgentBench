#!/usr/bin/env python3
"""Evaluate the NSG gate against SafeAgentBench ground-truth reference plans.

Loads safe_detailed, unsafe_detailed, and abstract dataset splits.
Converts each task's reference steps to TypedPrimitive objects via the adapter,
runs the NSG gate, and records gate decisions vs. ground-truth safety labels.

long_horizon is skipped — it has no reference step lists.

Usage:
    python run_nsg_experiment.py [--no-asp] [--out results/nsg/run.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from methods.nsg_gate.adapter import plan_to_primitives
from methods.nsg_gate.safety_gate import GateStatus, evaluate_plan

_RULES_PATH = _REPO / "methods" / "nsg_gate" / "rules" / "rsafety_household_v0.yaml"
_DATASET = _REPO / "dataset"


def _load_split(path: Path, true_label: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            steps = obj.get("step")
            if not steps or not isinstance(steps, list):
                continue
            instruction = obj.get("instruction", "")
            # abstract split: instruction is a list of 4 variants — use first (concrete)
            if isinstance(instruction, list):
                instruction = instruction[0]
            rows.append({
                "split": path.stem,
                "instruction": instruction,
                "risk_category": obj.get("risk_category", ""),
                "steps": steps,
                "true_label": true_label,
            })
    return rows


def _run_gate(rows: list[dict], use_asp: bool) -> list[dict]:
    results = []
    for row in rows:
        primitives = plan_to_primitives(row["steps"])
        gate = evaluate_plan(primitives, rules_path=_RULES_PATH, use_asp=use_asp)
        predicted = "unsafe" if gate.status in (GateStatus.blocked, GateStatus.needs_human) else "safe"
        results.append({
            **row,
            "gate_status": gate.status.value,
            "violations": gate.violations,
            "confirmation_requests": gate.confirmation_requests,
            "parse_flagged": gate.parse_flagged,
            "asp_used": gate.asp_used,
            "predicted_label": predicted,
        })
    return results


def _print_summary(results: list[dict]) -> None:
    unsafe = [r for r in results if r["true_label"] == "unsafe"]
    safe = [r for r in results if r["true_label"] == "safe"]

    unsafe_blocked = [r for r in unsafe if r["predicted_label"] == "unsafe"]
    safe_blocked = [r for r in safe if r["predicted_label"] == "unsafe"]

    rej_rate = len(unsafe_blocked) / len(unsafe) * 100 if unsafe else 0
    fp_rate = len(safe_blocked) / len(safe) * 100 if safe else 0

    print("\n=== NSG Gate — SafeAgentBench Results ===")
    print(f"Unsafe tasks:  {len(unsafe):>4}  |  Caught:  {len(unsafe_blocked):>4}  |  Rejection rate:    {rej_rate:.1f}%")
    print(f"Safe tasks:    {len(safe):>4}  |  Blocked: {len(safe_blocked):>4}  |  False-positive rate: {fp_rate:.1f}%")

    # Per-category breakdown (unsafe only)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in unsafe:
        by_cat[r["risk_category"] or "Unknown"].append(r)

    print("\nPer-category breakdown (unsafe tasks):")
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        caught = sum(1 for r in cat_rows if r["predicted_label"] == "unsafe")
        pct = caught / len(cat_rows) * 100
        print(f"  {cat:<50} {caught:>3}/{len(cat_rows):<3}  {pct:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSG gate evaluation on SafeAgentBench")
    parser.add_argument("--no-asp", action="store_true", help="Disable ASP backend (YAML rules only)")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "nsg" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.jsonl",
    )
    args = parser.parse_args()

    use_asp = not args.no_asp

    if not _RULES_PATH.is_file():
        print(f"Rules file not found: {_RULES_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset splits...")
    rows: list[dict] = []
    rows += _load_split(_DATASET / "safe_detailed_1009.jsonl", "safe")
    rows += _load_split(_DATASET / "unsafe_detailed_1009.jsonl", "unsafe")
    rows += _load_split(_DATASET / "abstract_1009.jsonl", "unsafe")
    print(f"  {len(rows)} tasks loaded (long_horizon skipped — no reference steps)")

    print(f"Running gate (ASP={'enabled' if use_asp else 'disabled'})...")
    results = _run_gate(rows, use_asp=use_asp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results written to {args.out}")

    _print_summary(results)


if __name__ == "__main__":
    main()
