# Hugging Face Spaces Packaging

This directory contains optional deployment packaging. It is not the canonical CySent v1 reproducibility path and was not validated during the P6 documentation pass.

The current container path is intended to serve the FastAPI backend and Next.js frontend. Random, Heuristic, and manual execution are the only artifact-free paths. Historical PPO, Fresh PPO, Qwen RL, and Hybrid require their exact external artifacts and provenance checks; packaging alone does not make those policies available.

For the supported local CPU research/demo workflow, use [`docs/v1-run.md`](../docs/v1-run.md). Do not infer deployment readiness from the presence of Docker or Spaces files.

The legacy `hf_spaces/CySent` Gradio material is retained as historical project context. It uses older naming and claims and is not the authoritative v1 API, interface, or policy contract. See the banner in `hf_spaces/CySent/BLOG.md` before citing that material.
