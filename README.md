# CySent

Autonomous cyber defense simulator with live environment decision-making.

CySent models adversarial attacks (RED) and autonomous defense actions (BLUE) in a Gymnasium-style environment, with a Next.js command dashboard and a FastAPI backend.

## Project Overview

CySent is designed for submission/demo workflows where you need:
1. A reproducible cyber defense environment with measurable risk and reward.
2. A stable PPO baseline policy.
# CySent

Autonomous cyber defense simulator with live environment decision-making.

CySent models adversarial attacks (RED) and autonomous defense actions (BLUE) in a Gymnasium-style environment, with a Next.js command dashboard and a FastAPI backend.

## Project Overview

CySent is designed for submission/demo workflows where you need:
1. A reproducible cyber defense environment with measurable risk and reward.
2. A stable PPO baseline policy.
3. An optional Hugging Face LLM agent path using a merged RL-trained Qwen model.
4. Exportable replay/benchmark evidence.

## Stack

1. Frontend: Next.js + TypeScript + Tailwind + Cytoscape.
2. Backend: FastAPI + Python.
3. Environment: custom Gymnasium-compatible security environment.
4. Agents:
	1. PPO (`ppo_agent`) default baseline.
	2. HF LLM (`hf_llm_agent`) using merged model repo when configured.

## Agent Paths

1. PPO Agent
	1. Deterministic baseline for stable defense behavior.
	2. Remains default-safe runtime path.

2. Hugging Face LLM Agent
	1. Uses merged model target when `HF_MERGED_MODEL_ID` is set.
	2. Current merged model repo: `sxchin01/CySent-Qwen-RL-merged`.
	3. Intended for RL-tuned Qwen behavior in the same live environment loop.

## Training Pipeline

1. PPO live environment training
	1. Script: `backend/train/train_ppo.py`.
	2. Produces PPO baseline artifacts in `backend/train/artifacts/`.

2. Qwen SFT warm start (optional)
	1. Notebook: `notebooks/CySent_Unsloth_Train.ipynb`.
	2. Produces adapter warm-start artifacts (optional stage).

3. Qwen live RL fine-tuning against environment
	1. Notebook: `notebooks/CySent_Qwen_LiveRL.ipynb`.
	2. Script path used by jobs: `scripts/train_on_hf.py`.
	3. Merge/export helper: `merge_upload.py`.

## Hugging Face Repositories

1. Space repo (app code):
	1. `https://huggingface.co/spaces/sxchin01/CySent`

2. Model repo (merged RL model):
	1. `https://huggingface.co/sxchin01/CySent-Qwen-RL-merged`

## Run Locally

### 1) Install dependencies

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

### 2) Configure `.env`

Required variables for current merged-model setup:

```dotenv
HF_TOKEN=your_hf_token
HF_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
HF_ADAPTER_PATH=sxchin01/CySent-Qwen-RL
HF_MERGED_MODEL_ID=sxchin01/CySent-Qwen-RL-merged
HF_ENDPOINT_URL=
HF_TIMEOUT=45.0

DEFAULT_AGENT=ppo_agent
AGENT_MODE=hybrid
HYBRID_THRESHOLD=10

API_HOST=127.0.0.1
API_PORT=8000
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Note:
1. `.env` is git-ignored.
2. Never commit real tokens.

### 3) Start backend and frontend

Backend:

```powershell
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
npm --prefix frontend run dev
```

Open:
1. Frontend: `http://127.0.0.1:3000`
2. API docs: `http://127.0.0.1:8000/docs`

## Switch Agents in Frontend

1. Use the agent selector in the dashboard (PPO vs HF LLM).
2. For explicit HF testing, select `hf_llm_agent` then reset/start a run.
3. Backend receives this selection as `action_source` through `/reset`.

## Screenshots / Assets

Current screenshot assets in repo:
1. `assets/screenshots/architecture.png`
2. `assets/screenshots/dashboard.png`
3. `assets/screenshots/colab_training_1.png`
4. `assets/screenshots/hf training.jpeg`

## Known Limitations

1. HF Space git remote history can diverge from local because of binary/LFS constraints; API sync may be needed for code-only updates.
2. HF Jobs scheduling can occasionally stall; retry or run merge from a GPU notebook when needed.
3. Provider/endpoint behavior may vary by account routing; merged model repo usage is the most stable path for this project.
4. On Windows, running multiple uvicorn instances causes `WinError 10048` on port `8000`; keep one backend instance only.

## Submission Links (Placeholders)

1. Demo video: `TBD`
2. GitHub repo: `https://github.com/sxchin-01/CySent`
3. HF Space: `https://huggingface.co/spaces/sxchin01/CySent`
4. HF merged model: `https://huggingface.co/sxchin01/CySent-Qwen-RL-merged`
5. Benchmark evidence: `TBD`

## Repo Hygiene Notes

1. `.env` is ignored by `.gitignore`.
2. Secrets should be passed via local env, HF secrets, or CI secrets only.
3. Use `git status` before pushing to confirm no accidental secret/config file is staged.