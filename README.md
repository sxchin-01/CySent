# CySent

Autonomous Cyber Defense Command Center

CySent is a cybersecurity simulation and decision platform where:
1. RED models adaptive attackers using threat profiles, chains, and pressure signals.
2. BLUE defends with a stable PPO policy by default.
3. BLUE can optionally use Hugging Face LLM decision support with PPO fallback.

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

Notes:
1. Dataset rows are generated in CySent instruction format: `instruction`, `input`, `output`.
2. Output action labels are constrained to current CySent valid action names.
3. PPO and backend runtime paths remain unchanged.

## Export Endpoints

1. Replay JSON download: `GET /replay/{episode_id}/export`
2. Benchmark JSON: `GET /benchmark/export?format=json`
3. Benchmark CSV: `GET /benchmark/export?format=csv`

## Demo Flow

Use these guides:
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