# Minimal replication

Implements the **Plain Explanation** "minimum viable replication":

1. **Lota-Bench–style agent** — Table 8–style prompt, no safety hint, `temperature=0`.
2. **GPT only** — pass `--model gpt-4o` or `gpt-4o-mini`.
3. **detail_evaluate** — execution (when `final_state` exists) + semantic judge.
4. **Metrics** — Rej; on unsafe, RR(goal) and RR(LLM) among unrejected.

## Agents

| `--agent` | Style | Notes |
|-----------|-------|-------|
| `lota` (default) | Single-shot, comma-separated plan | Table 8-style prompt |
| `react` | Multi-turn Thought/Action/Obs loop | Interleaves planning & execution |
| `progprompt` | Single-shot Python pseudocode | `#else:<action>` recovery steps |

## Run

```bash
export OPENAI_API_KEY=sk-...
cd SafeAgentBench
pip install -r requirements.txt

# Fast: no THOR, first 15 unsafe tasks (prints per-task status)
python minimal/run_minimal_replication.py --split unsafe --limit 15 --no-exec --model gpt-4o-mini

# Verbose: also prints plan steps, reference steps, and LLM judge rationale
python minimal/run_minimal_replication.py --split unsafe --limit 15 --no-exec --verbose

# Quiet: summary only
python minimal/run_minimal_replication.py --split unsafe --limit 15 --no-exec --quiet

# Different agent
python minimal/run_minimal_replication.py --split unsafe --limit 15 --no-exec --agent react

# With THOR (first 3 only)
python minimal/run_minimal_replication.py --split unsafe --limit 3 --model gpt-4o-mini
```

## Output JSONL fields

Every non-rejected record in the output JSONL includes:

| Field | Description |
|-------|-------------|
| `plan_steps` | Actual steps the agent planned |
| `ref_steps` | Reference steps from the dataset |
| `llm_judge` | Full LLM judge explanation (why success/fail) |
| `llm_success` | 1 = judge passed, 0 = judge failed |
| `success_goal` | State-based success (null when `final_state` missing) |
| `execution_rate` | Fraction of steps that executed successfully in THOR |
| `rejected` | True if agent refused the task |

Rejected records include `ref_steps` so you can see what was refused.

## Diagnosing failures

Run with `--verbose` to see per-task:
- The agent's plan vs. the reference plan
- The LLM judge's full rationale when it rules **fail**

For batch analysis, load the output JSONL and filter on `llm_success == 0`
to inspect `plan_steps` vs `ref_steps` and `llm_judge` for every failure.

Many **unsafe** rows have `final_state: null` — then only **LLM** risk rate is meaningful unless you execute and infer from state manually.

## Next steps (not implemented here)

- Full 300+300 loop and aggregation into Figure 4 buckets via `replication/`.
