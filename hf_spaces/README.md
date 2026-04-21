# Hugging Face Spaces (Docker) Launch

This folder contains a Docker-based launch option for CySent on Hugging Face Spaces.

## Expected Space setup
- SDK: Docker
- App port: 7860

## Files
- `Dockerfile`: Builds backend API and frontend app in one container.
- `start.sh`: Starts backend on 8000 and serves frontend on 7860.

## Environment variables
Set these in Space secrets/variables:
- `HF_TOKEN`
- `HF_MODEL_ID`
- `HF_ENDPOINT_URL`
- `HF_TIMEOUT`
- `AGENT_MODE`
- `HYBRID_THRESHOLD`

The frontend will call backend via `http://127.0.0.1:8000` inside the same container.
