# CySent

Autonomous Cyber Defense Command Center

CySent is a cybersecurity simulation and decision platform where:
1. RED models adaptive attackers using threat profiles, chains, and pressure signals.
2. BLUE defends with a stable PPO policy by default.
3. BLUE can optionally use Hugging Face LLM decision support with PPO fallback.

## Recent Updates (Apr 2026)

1. HF local LoRA loading path hardened for Qwen adapters:
	- Tries configured `HF_MODEL_ID` + `PeftModel` first.
	- Falls back to adapter-declared auto-PEFT loading when needed.
	- Emits clear adapter/base mismatch errors instead of silent failure.
2. Backend startup now surfaces HF initialization failures in router logs while preserving PPO fallback behavior.
3. Frontend Cytoscape graph rendering is guarded against invalid edges by only drawing edges whose source and target nodes exist in the current asset set.
4. Frontend API client now returns actionable connectivity and timeout errors (instead of generic "failed to fetch") when backend is down.

## Problem Statement

SOC teams face alert overload and inconsistent response quality while attack paths evolve faster than manual triage.
CySent provides a measurable cyber defense simulation where autonomous BLUE actions are benchmarked against stable and naive baselines.

## Why Cyber Defense Matters

1. Multi-step attacks can escalate from foothold to breach in minutes.
2. Defensive decisions must be reproducible, explainable, and stress-tested.
3. Benchmark-backed autonomy reduces guesswork in high-pressure incident response.

## Why CySent Is Novel / OpenEnv-Worthy

1. PPO-first control path with optional HF reasoning and strict PPO fallback safety.
2. End-to-end judge evidence pipeline: replay export, benchmark export, and artifacts.
3. OpenEnv adapter contract with operational `reset`, `step`, `state`, `close` methods.
4. Hybrid mode supports practical LLM usage under cost and latency constraints.

## Core Features

1. Real-time defense simulation with incident timeline and risk telemetry.
2. PPO-first decision path for stable baseline behavior.
3. Optional HF-assisted defense path with hybrid credit-saving mode.
4. Benchmark and evaluation pipeline with JSON and CSV artifacts.
5. Replay and benchmark export endpoints for submission evidence.
6. OpenEnv-compliant adapter with reset, step, state, and close methods.

## Architecture

1. Backend API: FastAPI runtime and orchestration.
2. Environment: Gymnasium-compatible cyber defense environment.
3. Threat Engine: adaptive attacker strategy and pressure dynamics.
4. Agents:
	- `ppo_agent` (default)
	- `hf_llm_agent` (optional)
5. Frontend: Next.js command dashboard and live visualization.

```text
   [Next.js Frontend]
	   |
	   v
 [FastAPI Runtime: backend/api/main.py]
	   |
	   v
       [AgentRouter] ----> [PPO Agent]
	   |
	   +-------------> [HF Agent (Base + LoRA)]
	   |
	   v
   [CySentSecurityEnv + Threat Engine]
	   |
	   v
 [Replay + Benchmark + Training Artifacts]
```

Primary paths:
1. Environment and threat logic: `backend/env/`
2. Agents and routing: `backend/agents/`
3. API service: `backend/api/main.py`
4. Training and benchmark: `backend/train/`
5. Frontend app: `frontend/`

## Quick Start

### 1) Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
npm --prefix frontend install
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
npm --prefix frontend install
```

### 2) Configure env

Create local `.env` from `.env.example` and set values for your machine.

Important variables:
1. `HF_TOKEN`
2. `HF_MODEL_ID`
3. `HF_ENDPOINT_URL`
4. `HF_TIMEOUT`
5. `AGENT_MODE` (`hybrid`, `full_llm`, `ppo_only`)
6. `HYBRID_THRESHOLD`

Local adapter run setup:
1. `HF_MODEL_ID=Qwen/Qwen2.5-3B-Instruct`
2. `HF_ADAPTER_PATH=outputs/cysent_unsloth_adapter`
3. `HF_ENDPOINT_URL=` (empty for local adapter mode)

### 3) Run backend + frontend

One-command scripts:
1. Windows: `scripts/run_dev.ps1` or `scripts/run_dev.cmd`
2. Linux/macOS: `scripts/run_dev.sh`

Manual run:

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
npm --prefix frontend run dev -- --hostname 127.0.0.1 --port 3000
```

Open:
1. Frontend: `http://127.0.0.1:3000`
2. API docs: `http://127.0.0.1:8000/docs`

## OpenEnv Compliance

CySent is OpenEnv compliant through `CySentOpenEnvAdapter` and `openenv.yaml`.

Required methods:
1. `reset()`
2. `step(action)`
3. `state()`
4. `close()`

Spec path:
1. `openenv.yaml`

Run a quick local check:

```bash
python -c "from backend.env.security_env import CySentOpenEnvAdapter; env=CySentOpenEnvAdapter(); obs,info=env.reset(); out=env.step(0); st=env.state(); env.close(); print(type(obs).__name__, type(info).__name__, len(out), type(st).__name__)"
```

## PPO vs HF Modes

Supported BLUE sources:
1. `ppo_agent` (default)
2. `hf_llm_agent`

Routing behavior:
1. PPO remains default and production-safe baseline.
2. HF can run full-time or hybrid mode (high risk or every N turns).
3. If HF fails or times out, router falls back to PPO.

## Training, Evaluation, Benchmark

Train PPO:

```bash
python -m backend.train.train_ppo --timesteps 100000 --model-path backend/train/artifacts/cysent_ppo
```

Evaluate PPO:

```bash
python -m backend.train.evaluate --episodes 50 --max-steps 150 --model-path backend/train/artifacts/best_model/best_model.zip --output backend/train/artifacts/eval_summary.json
```

Benchmark leaderboard:

```bash
python -m backend.train.benchmark --episodes 50 --seeds 20 --stress default --agents ppo hf_llm random
```

Artifacts:
1. `backend/train/artifacts/benchmark/benchmark_summary.json`
2. `backend/train/artifacts/benchmark/benchmark_table.csv`
3. `backend/train/artifacts/benchmark/benchmark_table.json`

## Results Snapshot (Judge View)

Use this table for final submission. Replace `TBD` with values from benchmark artifacts.

| Agent | Mean Reward | Mean Risk | Breach Rate | Evidence |
|---|---:|---:|---:|---|
| PPO (`ppo`) | TBD | TBD | TBD | `benchmark_summary.json` |
| HF (`hf_llm_agent`) | TBD | TBD | TBD | `benchmark_summary.json` |
| Random (`random`) | TBD | TBD | TBD | `benchmark_summary.json` |

If metrics are not finalized, include command + artifact timestamps as evidence placeholders.

## Training Evidence

PPO evidence placeholders:
1. `assets/screenshots/ppo-reward-curve.png`
2. `assets/screenshots/ppo-loss-curve.png`

Colab Unsloth fine-tune summary:
1. Notebook path: `notebooks/CySent_Unsloth_Train.ipynb`
2. Base model, dataset rows, and adapter output path summary
3. Local adapter smoke-test output (`scripts/test_hf_agent.py`)

## Train CySent Model in Colab with Unsloth

Quickstart:
1. Build dataset locally:

```bash
python scripts/build_cysent_dataset.py --output datasets/cysent_action_dataset.jsonl --rows 1200
```

2. Open notebook and run cells in order:

1. `notebooks/CySent_Unsloth_Train.ipynb`

3. Save adapter and test local inference:

```bash
python scripts/test_hf_agent.py --base-model Qwen/Qwen2.5-3B-Instruct --adapter outputs/cysent_unsloth_adapter --state "risk=0.72, attack=phishing_email, target=auth server, compromised=1, credential_exposure=0.81"
```

Use a trained adapter at runtime:
1. Set `HF_MODEL_ID` to the Qwen base model or your merged HF Hub repo id.
2. Set `HF_ADAPTER_PATH` to a local LoRA folder, a merged local model folder, or a HF Hub adapter repo id.
3. Keep `AGENT_MODE=hybrid` and select `hf_llm_agent`; PPO remains the fallback if the adapter fails or times out.

Today (Colab adapter local):
1. `HF_MODEL_ID=Qwen/Qwen2.5-3B-Instruct`
2. `HF_ADAPTER_PATH=outputs/cysent_unsloth_adapter`
3. `HF_ENDPOINT_URL=` (leave empty)
4. `HF_TIMEOUT=10.0`

Tomorrow (HF credits cloud):
1. `HF_TOKEN=<your_hf_token>`
2. `HF_MODEL_ID=<hf_model_repo_or_inference_target>`
3. Optional `HF_ENDPOINT_URL=<dedicated_endpoint_url>`
4. Clear `HF_ADAPTER_PATH=` to force hosted path

Runtime precedence:
1. Hosted HF is used when `HF_ENDPOINT_URL` is set, or when `HF_TOKEN` is set and `HF_ADAPTER_PATH` is empty.
2. Local adapter path is used when `HF_ADAPTER_PATH` is set and hosted config is not selected.
3. PPO remains default and fallback if HF init or HF inference fails.

Notes:
1. Dataset rows are generated in CySent instruction format: `instruction`, `input`, `output`.
2. Output action labels are constrained to current CySent valid action names.
3. PPO and backend runtime paths remain unchanged.

## Colab Training Completed

If the Colab notebook finishes, the adapter is saved at `outputs/cysent_unsloth_adapter`.

Use it at runtime by setting:
1. `HF_MODEL_ID` to the base model you trained against.
2. `HF_ADAPTER_PATH=outputs/cysent_unsloth_adapter` for a local adapter, or a merged local folder / HF repo id.
3. `AGENT_MODE=hybrid` to keep PPO as the fallback.

Benchmark the three comparison modes:

```bash
python -m backend.train.benchmark --episodes 50 --seeds 20 --stress default --agents ppo,hf_llm_agent,random
```

Artifacts land under `backend/train/artifacts/benchmark/`.

Screenshot placeholder path: `assets/screenshots/`

## Frontend Screenshots (Submission Placeholders)

Recommended placeholders:
1. `assets/screenshots/01-dashboard-overview.png`
2. `assets/screenshots/02-live-incident-feed.png`
3. `assets/screenshots/03-agent-selection-hf-vs-ppo.png`
4. `assets/screenshots/04-network-graph-active-defense.png`
5. `assets/screenshots/05-results-or-export-view.png`

## Export Endpoints

1. Replay JSON download: `GET /replay/{episode_id}/export`
2. Benchmark JSON: `GET /benchmark/export?format=json`
3. Benchmark CSV: `GET /benchmark/export?format=csv`

## Demo Flow

Judge quick flow (3 steps):
1. Start stack and open dashboard; pick scenario + attacker + agent mode.
2. Click Start and observe live graph, incidents, and action rationale updates.
3. Export replay/benchmark evidence and compare PPO vs HF vs Random.

Detailed guides:
1. `docs/demo-flow.md`
2. `docs/pitch-2min.md`

Recommended screenshot folder:
1. `assets/screenshots/`

## Deployment

Local Docker deployment:

```bash
docker compose up --build
```

Included files:
1. `Dockerfile.backend`
2. `Dockerfile.frontend`
3. `docker-compose.yml`

Hugging Face Spaces (Docker):
1. `hf_spaces/Dockerfile`
2. `hf_spaces/start.sh`
3. `hf_spaces/README.md`

## Branding

Product name: CySent

Tagline: Autonomous Cyber Defense Command Center

Visual language: deep slate/cyan command interface with incident-first telemetry.

## Future Work

1. Add richer scenario packs and attacker behavior libraries.
2. Improve adaptive hybrid routing for lower LLM cost and latency.
3. Expand benchmark suite with robustness and cross-seed confidence reporting.