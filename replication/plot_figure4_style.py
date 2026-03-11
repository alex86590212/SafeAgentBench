#!/usr/bin/env python3
"""
Stacked bar chart in the style of SafeAgentBench Figure 4:
  Rejection | Deliberate Failure | Planning Failure | Success/Risk

Feed per-baseline fractions (0–1) that sum to ~1 for hazardous breakdown.
No changes to base evaluator code.
"""
from __future__ import annotations

import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)


def plot_stacked(baselines: list[str], rej: list[float], df: list[float], pf: list[float], risk: list[float], out_path: str):
    x = np.arange(len(baselines))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, rej, width, label="Rejection", color="#4a90d9")
    ax.bar(x, df, width, bottom=rej, label="Deliberate Failure", color="#7b68ee")
    ax.bar(x, pf, width, bottom=np.array(rej) + np.array(df), label="Planning Failure", color="#f5a623")
    ax.bar(x, risk, width, bottom=np.array(rej) + np.array(df) + np.array(pf), label="Success/Risk", color="#d0021b")
    ax.set_ylabel("Fraction")
    ax.set_title("Hazardous tasks — performance breakdown (paper Figure 4 style)")
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=25, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Wrote", out_path)


def main():
    # Example: Table 2 style — replace with your logged aggregates
    example = {
        "baselines": ["Lota-Bench", "LLM-Planner", "ReAct"],
        "rejection": [0.00, 0.00, 0.10],
        "deliberate_failure": [0.05, 0.08, 0.12],
        "planning_failure": [0.55, 0.42, 0.38],
        "success_risk": [0.40, 0.50, 0.40],
    }
    json_path = os.path.join(os.path.dirname(__file__), "figure4_data.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            example = json.load(f)
    out = os.path.join(os.path.dirname(__file__), "figure4_replication.png")
    plot_stacked(
        example["baselines"],
        example["rejection"],
        example["deliberate_failure"],
        example["planning_failure"],
        example["success_risk"],
        out,
    )


if __name__ == "__main__":
    main()
