# CySent Demo Flow (5-7 minutes)

## 1. Setup (30s)
- Start backend API and frontend.
- Open dashboard and show health endpoint is reachable.

## 2. PPO Baseline (90s)
- Keep `ppo_agent` selected.
- Run a short live simulation.
- Highlight risk trend, incidents, and active action stream.

## 3. HF Assisted Defense (90s)
- Switch to `hf_llm_agent` or keep hybrid mode if configured.
- Show source badge changing to HF LLM Defender on decision turns.
- Explain fallback to PPO if HF is unavailable or slow.

## 4. Benchmark and Evidence (120s)
- Run benchmark command.
- Open generated files:
  - `backend/train/artifacts/benchmark/benchmark_summary.json`
  - `backend/train/artifacts/benchmark/benchmark_table.csv`
- Call export endpoints for reviewers.

## 5. Replay Export (45s)
- Use replay endpoint after an episode.
- Export replay JSON for audit traceability.

## 6. Close (30s)
- Recap: adaptive RED, PPO default, optional HF boost, reproducible benchmark artifacts.
