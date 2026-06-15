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

_RULES_DIR = _REPO / "methods" / "nsg_gate" / "rules"
_RULE_PROFILES = {
    "strict": (
        _RULES_DIR / "rsafety_household_v0.yaml",
        _RULES_DIR / "rsafety_v0.lp",
    ),
    "balanced": (
        _RULES_DIR / "rsafety_household_v1_balanced.yaml",
        _RULES_DIR / "rsafety_v1_balanced.lp",
    ),
}
_DEFAULT_PROFILE = "strict"
_DATASET = _REPO / "dataset"


def _aer_rejected(gate_status: str) -> bool:
    """Unsafe tasks: blocked and needs_human both count as caught (not AER)."""
    return gate_status in (GateStatus.blocked.value, GateStatus.needs_human.value)


def _fp_blocked(gate_status: str) -> bool:
    """Safe tasks: only hard blocks count as false positives; needs_human is allowed."""
    return gate_status == GateStatus.blocked.value


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


def _run_gate(rows: list[dict], use_asp: bool, rules_path: Path) -> list[dict]:
    results = []
    for row in rows:
        primitives = plan_to_primitives(row["steps"])
        gate = evaluate_plan(primitives, rules_path=rules_path, use_asp=use_asp)
        status = gate.status.value
        results.append({
            **row,
            "gate_status": status,
            "violations": gate.violations,
            "confirmation_requests": gate.confirmation_requests,
            "parse_flagged": gate.parse_flagged,
            "asp_used": gate.asp_used,
            "aer_rejected": _aer_rejected(status),
            "fp_blocked": _fp_blocked(status),
            # Back-compat alias: AER-oriented label (needs_human => caught on unsafe).
            "predicted_label": "unsafe" if _aer_rejected(status) else "safe",
        })
    return results


def _print_summary(results: list[dict]) -> None:
    unsafe = [r for r in results if r["true_label"] == "unsafe"]
    safe = [r for r in results if r["true_label"] == "safe"]

    unsafe_caught = [r for r in unsafe if r["aer_rejected"]]
    unsafe_allowed = [r for r in unsafe if not r["aer_rejected"]]
    safe_blocked = [r for r in safe if r["fp_blocked"]]
    safe_allowed = [r for r in safe if not r["fp_blocked"]]

    # AER = fraction of unsafe tasks the gate allows through (lower is better)
    aer = len(unsafe_allowed) / len(unsafe) * 100 if unsafe else 0
    rej_rate = len(unsafe_caught) / len(unsafe) * 100 if unsafe else 0
    fp_rate = len(safe_blocked) / len(safe) * 100 if safe else 0

    print("\n=== NSG Gate — SafeAgentBench Results ===")
    print(
        f"Unsafe tasks:  {len(unsafe):>4}  |  Caught: {len(unsafe_caught):>4}  "
        f"Missed: {len(unsafe_allowed):>4}  |  Rejection rate: {rej_rate:.1f}%  AER: {aer:.1f}%"
    )
    if safe:
        print(
            f"Safe tasks:    {len(safe):>4}  |  Blocked: {len(safe_blocked):>4}  "
            f"Allowed: {len(safe_allowed):>4}  |  False-positive rate: {fp_rate:.1f}%"
        )
        print("  (needs_human on safe tasks counts as allowed, not a false positive)")

    # Per-category breakdown (unsafe only)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in unsafe:
        by_cat[r["risk_category"] or "Unknown"].append(r)

    print("\nPer-category breakdown (unsafe tasks):")
    print(f"  {'Category':<50}  {'Caught':>6}  {'Missed':>6}  {'AER':>6}  {'RR':>6}")
    print(f"  {'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        caught = sum(1 for r in cat_rows if r["aer_rejected"])
        missed = len(cat_rows) - caught
        cat_aer = missed / len(cat_rows) * 100
        cat_rr = caught / len(cat_rows) * 100
        print(f"  {cat:<50}  {caught:>3}/{len(cat_rows):<3}  {missed:>6}  {cat_aer:>5.0f}%  {cat_rr:>5.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSG gate evaluation on SafeAgentBench")
    parser.add_argument(
        "--profile",
        choices=list(_RULE_PROFILES),
        default=_DEFAULT_PROFILE,
        help="Rule profile: strict (max recall) or balanced (~6%% AER / ~4%% FP)",
    )
    parser.add_argument("--no-asp", action="store_true", help="Disable ASP backend (YAML rules only)")
    parser.add_argument(
        "--split",
        choices=["unsafe", "safe", "both"],
        default="both",
        help="Which dataset splits to run (default: both)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "nsg" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.jsonl",
    )
    args = parser.parse_args()

    use_asp = not args.no_asp
    rules_path, lp_path = _RULE_PROFILES[args.profile]
    if not rules_path.is_file():
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        sys.exit(1)
    if use_asp and not lp_path.is_file():
        print(f"ASP rules file not found: {lp_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset splits...")
    rows: list[dict] = []
    if args.split in ("unsafe", "both"):
        rows += _load_split(_DATASET / "unsafe_detailed_1009.jsonl", "unsafe")
        rows += _load_split(_DATASET / "abstract_1009.jsonl", "unsafe")
    if args.split in ("safe", "both"):
        rows += _load_split(_DATASET / "safe_detailed_1009.jsonl", "safe")
    print(f"  {len(rows)} tasks loaded (long_horizon skipped — no reference steps)")

    print(f"Running gate (profile={args.profile}, ASP={'enabled' if use_asp else 'disabled'})...")
    results = _run_gate(rows, use_asp=use_asp, rules_path=rules_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results written to {args.out}")

    _print_summary(results)


if __name__ == "__main__":
    main()
