#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"

python3 -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

NEXT_PUBLIC_API_URL="http://127.0.0.1:8000" npm --prefix frontend run dev -- --hostname 127.0.0.1 --port 3000
