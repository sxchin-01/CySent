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

## What Is CySent?

CySent is a cybersecurity simulation and decision platform where autonomous defense agents reason about attack-defense dynamics and recommend immediate actions.

### Key Capabilities
- **PPO Baseline:** Production-safe, stable decision-making policy trained with reinforcement learning
- **HF LLM Reasoning:** Optional fine-tuned Qwen adapter for contextual action selection
- **Hybrid Mode:** Combines PPO and LLM with risk-based switching
- **OpenEnv Compliance:** Standard environment interface (reset, step, state, close)

## Why It Matters

1. **Consistency:** Defenders need repeatable action quality under incident pressure.
2. **Explainability:** Decision systems should reason transparently, not as black boxes.
3. **Reproducibility:** Benchmark-backed autonomy supports trust and auditing.

## Features

### Defense Actions
- Patch systems
- Isolate compromised hosts
- Enable/reset credentials
- Alert SOC
- Investigate incidents
- And 7+ more in standard action set

### Agents
| Agent | Mode | Use Case |
|-------|------|----------|
| PPO | Default | Stable, production-ready baseline |
| HF LLM | Optional | Contextual reasoning on high-risk events |
| Hybrid | Mixed | Switch between PPO/HF based on risk score |

## How to Use

1. **Describe the Security State**  
   Provide a brief state: risk level, attack type, compromise status, etc.
   
2. **Get Recommendation**  
   CySent recommends an action with confidence and method.
   
3. **Apply Action**  
   Execute the recommended defense measure.

## Benchmark Results (Placeholder)

| Agent | Avg Reward | Avg Risk | Detection Rate |
|---|---:|---:|---:|
| PPO | TBD | TBD | TBD |
| HF LLM | TBD | TBD | TBD |
| Random Baseline | TBD | TBD | TBD |

## Screenshots

- [Architecture Diagram](../assets/screenshots/architecture.png)
- [Colab Training Flow](../assets/screenshots/colab_training.png)

## Configuration

Set environment variables in Space secrets:
- `HF_MODEL_ID` (default: `Qwen/Qwen2.5-3B-Instruct`)
- `HF_ADAPTER_PATH` (local LoRA folder or HF Hub repo)
- `HF_TOKEN` (for private models/adapters)

## Links

- **GitHub:** https://github.com/sxchin-01/CySent
- **Colab Training:** `notebooks/CySent_Unsloth_Train.ipynb`
- **Paper/Docs:** See GitHub repository

## Limitations

- Inference uses simplified heuristics when HF model is unavailable.
- Adapter training uses curated cybersecurity action datasets (not real incidents).
- This Space demonstrates the model; production deployment requires full backend/frontend stack.

## Running Locally

```bash
# Install minimal deps
pip install -r requirements.txt

# Run Gradio app
python app.py

# Visit http://127.0.0.1:7860
```

## Citation

If you use CySent in research or production, cite:

```bibtex
@software{cysent2026,
  title={CySent: Autonomous Cyber Defense Command Center},
  author={...},
  year={2026},
  url={https://github.com/sxchin-01/CySent}
}
```

---

**CySent** — Autonomous Defense at Human Speed.
