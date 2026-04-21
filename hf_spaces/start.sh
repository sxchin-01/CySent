#!/usr/bin/env bash
set -euo pipefail

python3 -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 &

exec npm --prefix /app/frontend run start -- --hostname 0.0.0.0 --port 7860
