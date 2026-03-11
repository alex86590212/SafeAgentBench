# Paper prompts (reference only)

- **Table 8** (appendix): exact **planning** prompts for each baseline — copy from the PDF; do not edit `evaluator/*.py` if you need identical judge prompts (those are already in code and match appendix Tables 9–11).
- **Semantic evaluator**: **gpt-4o-2024-08-06**; text matches `evaluator/detail_evaluate.py` (`compute_SR_llm`), `evaluator/abstract_evaluate.py`, `evaluator/long_horizon_evaluate.py`.

To run an experiment **without modifying base code**:

1. Implement baseline planner **externally** or in a **new file** under `replication/` that **imports** `evaluator.detail_evaluate.evaluate` etc. as functions.
2. Pass the same `task`, `steps_plan`, `steps_ref`, `final_state` the paper uses.
3. Log `rejected`, execution success, LLM success — then aggregate with `replication/paper_metrics.py` and plot with `replication/plot_figure4_style.py`.
