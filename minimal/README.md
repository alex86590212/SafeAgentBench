# Minimal replication

Implements the **Plain Explanation** “minimum viable replication”:

1. **Lota-Bench–style agent** — Table 8–style prompt, no safety hint, `temperature=0`.
2. **GPT only** — pass `--model gpt-4o` or `gpt-4o-mini`.
3. **detail_evaluate** — execution (when `final_state` exists) + semantic judge.
4. **Metrics** — Rej; on unsafe, RR(goal) and RR(LLM) among unrejected.

## Run

```bash
export OPENAI_API_KEY=sk-...
cd SafeAgentBench
pip install -r requirements.txt

# Fast: no THOR, first 15 unsafe tasks (prints per-task status)
python minimal/run_minimal_replication.py --split unsafe --limit 15 --no-exec --model gpt-4o-mini

# Quiet: summary only
python minimal/run_minimal_replication.py --split unsafe --limit 15 --no-exec --quiet

# With THOR (first 3 only)
python minimal/run_minimal_replication.py --split unsafe --limit 3 --model gpt-4o-mini
```

Many **unsafe** rows have `final_state: null` — then only **LLM** risk rate is meaningful unless you execute and infer from state manually.

## Next steps (not implemented here)

- **ReAct** agent (Thought/Action + higher refusal).
- **ProgPrompt** (assert/else plans) — needs format alignment.
- Full 300+300 loop and aggregation into Figure 4 buckets via `replication/`.
