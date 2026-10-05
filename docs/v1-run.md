# CySent v1 Run Guide

CySent v1 is a reproducible research and demonstration system. The artifact-free live path runs Random and manual actions on CPU; the artifact-free benchmark path also includes Heuristic. PPO and Qwen paths require separately supplied model artifacts.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Windows PowerShell commands are shown first; equivalent POSIX commands follow where they differ.

## Fresh-Clone CPU Setup

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-cpu.txt
cd frontend
npm install
cd ..
```

On Linux or macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

The CPU setup does not require a Hugging Face token, a PPO checkpoint, or a Qwen artifact for the Random/manual demo or Random/Heuristic benchmarks.

## Start the Backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. Use `GET /agents` to inspect which canonical policies are currently available and why unavailable policies cannot run.

## Start the Frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000`. The frontend uses `http://127.0.0.1:8000` by default unless `NEXT_PUBLIC_API_URL` is set.

## CPU Demonstration

1. Start the backend and frontend.
2. Select `Random` from the policy controls.
3. Reset the episode, then step or run it.
4. Inspect requested and executed actions, substitutions, reward, risk, uptime, cost, and episode state.
5. Pause to inspect history or submit a manual action.

Random and Heuristic must remain usable when every model-backed policy is unavailable. The interface reports unavailable PPO, Qwen, and Hybrid paths rather than silently substituting another identity.

## CPU Benchmark

The frozen three-case matrix can be rerun for artifact-free policies:

```powershell
.\.venv\Scripts\python.exe -m backend.train.benchmark_agents --agents random,heuristic --seeds 42,43,44 --max-steps 150 --outdir outputs/benchmarks/v1_cpu_demo
```

Choose a new output directory for each run. Do not overwrite the frozen evidence under `outputs/benchmarks/p1_baseline_v2` or `outputs/benchmarks/p2_fresh_ppo_primary`.

## PPO Artifacts

PPO checkpoints are intentionally not bundled as ordinary source dependencies. The frozen identities are:

| Identity | Expected path | SHA-256 |
| --- | --- | --- |
| Historical PPO | `backend/train/artifacts/best_model/best_model.zip` | `4D4955993DD2D98CC3B8D3319C1CDA92F10EFBA95407B3D78D9763E5D968D1AF` |
| Fresh PPO best | `backend/train/artifacts/p2_fresh_ppo/p2_fresh_primary_seed42/best_model/best_model.zip` | `7BA9122F3AE4BD67E539EC3BEE95EB587F63F22091605DDEF044190586DC0197` |

After placing an artifact at its expected path, verify it before execution:

```powershell
.\.venv\Scripts\python.exe -m backend.artifacts historical_ppo
.\.venv\Scripts\python.exe -m backend.artifacts fresh_ppo
```

With both verified checkpoints, the controlled four-agent benchmark command is:

```powershell
.\.venv\Scripts\python.exe -m backend.train.benchmark_agents --agents random,heuristic,historical_ppo,fresh_ppo --seeds 42,43,44 --max-steps 150 --outdir outputs/benchmarks/local_four_agent_run
```

This creates new evidence; it does not replace the frozen P1/P2 results.

## Qwen GPU Evaluation

The canonical Qwen artifact is:

- Repository: `sxchin01/CySent-Qwen-RL-merged`
- Immutable revision: `fb75512b037bb37de575916afd900c03ab860cb5`
- Policy: `historical_first_token_seeded_categorical_v1`

Qwen execution requires a compatible GPU environment and a verified Hugging Face cache snapshot or verified local artifact metadata. Set `HF_ADAPTER_PATH` to that resolved local path; do not use an unverified directory merely because it contains model files.

```powershell
.\.venv\Scripts\python.exe -m backend.artifacts qwen_merged --path $env:HF_ADAPTER_PATH
```

The historical evaluation workflow is documented in `notebooks/cysent_qwen_evaluation.ipynb`. It preserves the historical prompt, first-token scoring, action-token collision, FP32 categorical normalization, and agent-local seeded sampling contract. Real Qwen smoke testing succeeded, but a controlled tracked Qwen benchmark result is not part of the frozen v1 evidence. Real-model Hybrid smoke remains incomplete because the hosted GPU session terminated without a captured Python exception.

The original SFT/RL notebooks are historical training records, not the canonical v1 execution path.

## Tests and Builds

Backend suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Frontend checks:

```powershell
cd frontend
npm run typecheck
npm run build
```

## Reproducibility Classification

| Path | Fresh clone | External artifact | GPU | v1 status |
| --- | --- | --- | --- | --- |
| Random / Heuristic / manual | Yes | No | No | Reproducible CPU demo |
| Historical / Fresh PPO | Code only | Required | No | Reproducible after hash verification |
| Qwen RL | Code only | Required | Yes for intended evaluation | Real smoke verified; controlled benchmark incomplete |
| Hybrid Router | Code only | PPO and Qwen required | Yes for Qwen branch | Routing tested; real-model smoke incomplete |

## Deployment Notes

The repository contains Docker and Hugging Face Spaces packaging, but those paths are not the frozen v1 reproducibility contract and have not been validated as part of P6. Use the local CPU workflow above for the supported demo path.
