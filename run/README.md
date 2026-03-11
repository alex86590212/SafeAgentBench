# run_benchmark.py

Implements the loop you described:

```text
for task in dataset:
    plan = agent.generate_plan(task)
    result = controller.execute(plan)   # unless --no-exec
    score = evaluator.evaluate(...)
    log(score)
```

## Dataset sizes

| Split            | Raw tasks | Notes                                      |
|----------------|-----------|--------------------------------------------|
| safe_detailed    | 300       | THOR + detail_evaluate                     |
| unsafe_detailed  | 300       | THOR + detail_evaluate                     |
| abstract         | 100       | Expands to **400** rows (L1–L4) like paper |
| long_horizon     | 50        | Plan-only judge (no THOR required)         |

## Usage

From repo root:

```bash
export OPENAI_API_KEY=sk-...
cd SafeAgentBench

# Smoke test (2 tasks, oracle agent, no THOR)
python run/run_benchmark.py --dataset safe_detailed --limit 2 --agent reference --no-exec

# All splits, 5 tasks each
python run/run_benchmark.py --dataset all --limit 5 --agent reference --no-exec

# Full run with execution (slow; needs AI2-THOR)
python run/run_benchmark.py --dataset safe_detailed --agent reference
```

## Agents

| `--agent`   | Behavior                          |
|------------|------------------------------------|
| `reference`| Uses dataset `step` as plan (oracle)|
| `stub`     | Empty plan                         |
| `reject`   | Always REJECT                      |

Plug your LLM: subclass `BaseAgent` in `agents.py` and add to `get_agent()`.

## Output

JSONL path printed at end; default `run/results_<timestamp>.jsonl`. Feed into `replication/plot_figure4_style.py` after aggregating.
