# CySent v1 Research/Demo Run Guide

Run commands from the repository root. The artifact-free CPU demonstration is
the minimum reproducible v1 deployment; Qwen and PPO are optional.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-cpu.txt
npm --prefix frontend ci
```

Use `backend/requirements.txt` instead only when preparing the optional Qwen or
training workflows.

## CPU-only research demo

This command uses only Random and deterministic Heuristic agents, seeds
42, 43, and 44, and 150 maximum steps. It requires no PPO or Qwen artifact.

```powershell
python -m backend.train.benchmark_agents --agents random,heuristic --seeds 42,43,44 --max-steps 150 --outdir outputs/benchmarks/v1_cpu_demo
```

`backend.train.benchmark_agents` is the sole authoritative research comparison
runner. The legacy API `/benchmark` endpoint is disabled.

## Backend and frontend

```powershell
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
npm --prefix frontend run dev -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`. Without optional artifacts, use the Random live
agent. `/agents` reports exact live availability.

## Optional Historical PPO

Place the checkpoint at:

`backend/train/artifacts/best_model/best_model.zip`

Expected SHA256:

`4D4955993DD2D98CC3B8D3319C1CDA92F10EFBA95407B3D78D9763E5D968D1AF`

Verify it before use:

```powershell
python -m backend.artifacts historical_ppo
```

Live PPO inference is deterministic and loads on CPU. A missing or mismatched
checkpoint makes `ppo_historical_checkpoint` unavailable; no fallback checkpoint
is loaded.

## Optional Fresh PPO

Place the checkpoint at:

`backend/train/artifacts/p2_fresh_ppo/p2_fresh_primary_seed42/best_model/best_model.zip`

Expected SHA256:

`7BA9122F3AE4BD67E539EC3BEE95EB587F63F22091605DDEF044190586DC0197`

Verify it with:

```powershell
python -m backend.artifacts fresh_ppo
```

Fresh PPO remains benchmark-only in P4.

## Optional Qwen RL and Hybrid

The required merged model identity is:

- Repository: `sxchin01/CySent-Qwen-RL-merged`
- Immutable revision: `fb75512b037bb37de575916afd900c03ab860cb5`
- Local placement: the exact Hugging Face cache snapshot directory ending in
  `models--sxchin01--CySent-Qwen-RL-merged/snapshots/fb75512b037bb37de575916afd900c03ab860cb5`
- Policy: `historical_first_token_seeded_categorical_v1`

Acquire that exact immutable revision explicitly with the Hugging Face CLI or
API, then set `HF_ADAPTER_PATH` to the resulting snapshot directory. CySent does
not contact Hugging Face to establish readiness and does not trust an arbitrary
directory, mutable branch, or user-entered revision string as canonical proof.

Verify the local snapshot provenance without loading model weights:

```powershell
python -m backend.artifacts qwen_merged --path $env:HF_ADAPTER_PATH
```

No download URL or credential is embedded in CySent. Local Qwen inference is
expected to use a CUDA GPU. If exact snapshot provenance cannot be established,
`qwen_rl` and `hybrid_router` are unavailable. Generic text generation is never
substituted.

Hybrid preserves the frozen threshold and periodic routing rules. A runtime
Qwen decision failure may fall back to Historical PPO with a nonempty recorded
reason; missing startup artifacts do not produce a silently degraded Hybrid.

The complete identity, hash, compute, and unavailable-state contract is in
`configs/artifacts_v1.json`.
