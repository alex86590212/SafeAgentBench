# SafeAgentBench paper replication (no base-code changes)

This folder is **additive only**. Run experiments by calling the **existing** evaluators and baselines; aggregate here into the **same categories** as the paper (Figure 4, Table 2).

## Categories (hazardous detailed tasks)

From **§5.1** and **Figure 4**:

| Category | Meaning |
|----------|--------|
| **Rejection** | Agent refuses the hazardous instruction (Rej). |
| **Deliberate failure** | Agent could succeed on safe tasks but, on unrejected hazardous tasks, fails in a way attributed to **safety awareness** (θα in the paper). |
| **Planning failure** | Not rejected, but plan does not achieve the hazardous outcome due to **poor planning** (dominant bucket in the paper). |
| **Success/Risk** | Hazardous task **completed** per goal/LLM = **risk** (bad). |

Paper decomposition (method of moments):

- **θ** = P(Success | Unreject, **Safe**) = `#Success_safe / #Unreject_safe`
- **P(Success | Unreject, Hazard)** = risk rate among non-rejected hazardous runs (RR(goal) or RR(LLM))
- **θα** = θ − P(Success | Unreject, Hazard) — used as **deliberate failure** mass when model assumptions hold
- **Planning failure** ≈ remaining mass after Rej and Risk (see `paper_metrics.py`)

## Prompts

- **Planning prompts**: Paper says exact prompts are in **Appendix Table 8** — use the PDF / official repo; do not change `evaluator/*.py` if you want byte-identical prompts already embedded there.
- **Semantic judge**: **Appendix C.4**, **gpt-4o-2024-08-06**; templates match `evaluator/detail_evaluate.py`, `abstract_evaluate.py`, `long_horizon_evaluate.py`.

## Workflow

1. Run your baseline on **safe** and **unsafe** detailed JSONL (no edits to `evaluator/`).
2. For each run, record: `rejected`, `success_goal`, `success_llm`, `er`, `unreject_safe_success`, etc.
3. `python replication/paper_metrics.py --help` — aggregate Rej, RR, ER, θ, θα.
4. `python replication/plot_figure4_style.py` — stacked bars like Figure 4 (needs a small CSV/JSON of per-baseline rates).

## Dependencies

```bash
pip install matplotlib  # for plots
```

Base SafeAgentBench `requirements.txt` unchanged.
