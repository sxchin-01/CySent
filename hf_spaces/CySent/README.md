---
title: CySent
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# CySent — Autonomous Cyber Defense Command Center

**Live cyber defense decision-making powered by PPO and fine-tuned LLMs.**

## Blog

- Read the full project article: [BLOG.md](./BLOG.md)

## What Is CySent?

CySent is a cybersecurity simulation and decision platform where autonomous defense agents reason about attack-defense dynamics and recommend immediate actions.

### Agents
| Agent | Mode | Use Case |
|-------|------|----------|
| PPO | Default | Stable, production-ready RL baseline |
| Qwen LoRA (SFT) | Optional | Fine-tuned LLM trained on static dataset |
| Qwen LoRA (RL) | Optional | Live RL-trained LLM (REINFORCE + CySent env) |
| Hybrid | Mixed | Risk-based switching between PPO and LLM |

## How to Use

1. **Describe the Security State** — risk level, attack type, compromise status
2. **Get Recommendation** — CySent returns an action with confidence and method
3. **Apply Action** — execute the recommended defense measure

## Configuration

Set these in **Space Secrets** (Settings > Repo Secrets):

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `HF_ADAPTER_PATH` | `sxchin01/CySent-adapter` | LoRA adapter (HF Hub or local) |
| `HF_TOKEN` | — | HF token (for private models) |

## Links

| Resource | URL |
|----------|-----|
| GitHub (full project) | [sxchin-01/CySent](https://github.com/sxchin-01/CySent) |
| Project Blog | [BLOG.md](./BLOG.md) |
| Adapter (Qwen LoRA) | [sxchin01/CySent-adapter](https://huggingface.co/sxchin01/CySent-adapter) |
| Colab: Qwen SFT | [CySent_Unsloth_Train.ipynb](https://github.com/sxchin-01/CySent/blob/main/notebooks/CySent_Unsloth_Train.ipynb) |
| Colab: Qwen Live RL | [CySent_Qwen_LiveRL.ipynb](https://github.com/sxchin-01/CySent/blob/main/notebooks/CySent_Qwen_LiveRL.ipynb) |
## Running Locally

```bash
pip install -r requirements.txt
python app.py
# Visit http://127.0.0.1:7860
```

---

**CySent** — Autonomous Defense at Human Speed.
