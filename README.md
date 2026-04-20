# CySent

AI Security Operations Commander

CySent is a reinforcement learning cybersecurity simulation platform where:

- RED is a dynamic threat engine (rules, probabilities, attack chains)
- BLUE is an RL policy agent (PPO) that learns enterprise defense strategy

## Project Layout

- `backend/env`: Gymnasium-compatible environment, risk, reward, and threat logic
- `backend/train`: PPO training and evaluation scripts
- `backend/api`: FastAPI service exposing environment and training APIs
- `frontend`: Next.js dashboard with live network graph and telemetry
- `notebooks`: notebooks for experimentation (optional)

## Tech Stack

### Backend

- Python 3.10+
- OpenEnv (optional runtime registration)
- Gymnasium
- Stable-Baselines3 (PPO)
- PyTorch
- FastAPI

### Frontend

- Next.js App Router
- React + TypeScript
- TailwindCSS
- Framer Motion
- Recharts
- Cytoscape.js
- shadcn-style UI primitives

## Backend Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Run API:

```bash
python -m uvicorn backend.api.main:app --reload
```

Train PPO:

```bash
.venv/Scripts/python.exe -m backend.train.train_ppo --timesteps 100000 --model-path backend/train/artifacts/cysent_ppo
```

Train tuned PPO with 4 envs and tracking:

```bash
.venv/Scripts/python.exe -m backend.train.train_ppo --timesteps 200000 --model-path backend/train/artifacts/cysent_ppo_tuned --n-envs 4
```

Windows stable launcher (always uses root `.venv`):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_ppo.ps1
```

Open TensorBoard:

```bash
.venv/Scripts/tensorboard.exe --logdir backend/train/artifacts/runs --port 6006
```

Windows TensorBoard launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tensorboard.ps1 -Port 6006
```

Evaluate trained vs random policy:

```bash
.venv/Scripts/python.exe -m backend.train.evaluate --model-path backend/train/artifacts/cysent_ppo.zip --episodes 50
```

Run benchmark mode (random vs baseline vs tuned vs cloud-ready slot):

```bash
.venv/Scripts/python.exe -m backend.train.benchmark --episodes 50 --baseline-model backend/train/artifacts/cysent_ppo.zip --tuned-model backend/train/artifacts/best_model/best_model.zip
```

Benchmark outputs JSON and charts into `backend/train/artifacts/`.

## API Endpoints

- `GET /state`
- `POST /step`
- `GET /metrics`
- `GET /training-status`
- `GET /replay/{episode_id}`
- `POST /train`
- `POST /benchmark`

## Frontend Quick Start

```bash
cd frontend
npm install
set NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Open `http://localhost:3000`.

## Action Space

- `0` do_nothing
- `1` patch_hr_systems
- `2` patch_web_server
- `3` patch_auth_server
- `4` rotate_credentials
- `5` isolate_suspicious_host
- `6` increase_monitoring
- `7` restore_backup
- `8` deploy_honeypot
- `9` phishing_training
- `10` investigate_top_alert
- `11` segment_finance_database

## Notes on OpenEnv

The environment is implemented as a Gymnasium environment and includes optional OpenEnv registration through `maybe_register_openenv_env()`. If OpenEnv is installed and provides a `register` API, the environment registers as `CySentSecurity-v0`.