from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.agents import AgentRouter
from backend.agents.router import VALID_AGENT_NAMES
from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv
from backend.train.benchmark import build_benchmark
from backend.train.evaluate import evaluate
from backend.train.train_ppo import train

app = FastAPI(title="CySent API", version="1.0.0")


def _load_local_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries from .env into process env when missing."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and os.getenv(key) is None:
            os.environ[key] = value


_load_local_env_file()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StepRequest(BaseModel):
    action: int = Field(ge=0, le=11)


class ResetRequest(BaseModel):
    seed: int = 42
    scenario: str = "legacy"
    difficulty: str = "medium"
    attacker: str = "legacy_default"
    strategy_mode: str = "balanced"
    action_source: str = "ppo_agent"
    intelligence_enabled: bool = True


class TrainRequest(BaseModel):
    timesteps: int = Field(default=100_000, ge=1_000, le=5_000_000)
    seed: int = 42
    max_steps: int = Field(default=150, ge=25, le=1000)


class ReplayResponse(BaseModel):
    episode_id: str
    events: List[Dict[str, Any]]


class BenchmarkRequest(BaseModel):
    episodes: int = Field(default=50, ge=10, le=500)
    max_steps: int = Field(default=150, ge=25, le=1000)
    seed: int = 42
    agents: List[str] = Field(default_factory=lambda: ["ppo", "hf_llm_agent", "random"])
    seeds: int = Field(default=20, ge=1, le=200)
    stress: str = "default"
    baseline_model: str = "backend/train/artifacts/cysent_ppo.zip"
    tuned_model: str = "backend/train/artifacts/best_model/best_model.zip"
    cloud_model: Optional[str] = None
    output: str = "backend/train/artifacts/benchmark/benchmark_summary.json"


class CySentRuntime:
    def __init__(self) -> None:
        self.env = CySentSecurityEnv(max_steps=150, seed=42)
        self.agent_router = AgentRouter()
        # Re-entrant lock prevents deadlocks when locked code paths call snapshot_state().
        self.state_lock = threading.RLock()
        self.last_info: Dict[str, Any] = self.env.reset()[1]
        self.current_episode_id = str(uuid.uuid4())
        self.replays: Dict[str, List[Dict[str, Any]]] = {}
        self.training: Dict[str, Any] = {
            "running": False,
            "last_run": None,
            "last_error": None,
            "evaluation": None,
            "benchmark": None,
        }

    def snapshot_state(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "episode_id": self.current_episode_id,
                "step": self.last_info.get("step", 0),
                "network_risk": self.last_info.get("network_risk", 0.0),
                "risk_breakdown": self.last_info.get("risk_breakdown", {}),
                "assets": self.last_info.get("assets", []),
                "last_action": self.last_info.get("last_action", "reset"),
                "red_log": self.last_info.get("red_log", {}),
                "profile": self.last_info.get("profile", {}),
                "intelligence": self.last_info.get("intelligence", {}),
                "events": self.last_info.get("events", []),
                "narrative": self.last_info.get("narrative", ""),
                "termination_reason": self.last_info.get("termination_reason", "active"),
            }


class RuntimeManager:
    """Manages per-session CySentRuntime instances with TTL-based eviction."""

    def __init__(self, max_sessions: int = 64, ttl_seconds: float = 3600) -> None:
        self._sessions: Dict[str, CySentRuntime] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    def get(self, session_id: str = "default") -> CySentRuntime:
        with self._lock:
            now = time.monotonic()
            self._evict_stale(now)
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    self._evict_oldest()
                self._sessions[session_id] = CySentRuntime()
            self._last_access[session_id] = now
            return self._sessions[session_id]

    def _evict_stale(self, now: float) -> None:
        expired = [
            k for k, t in self._last_access.items()
            if now - t > self._ttl and k != "default"
        ]
        for k in expired:
            self._sessions.pop(k, None)
            self._last_access.pop(k, None)

    def _evict_oldest(self) -> None:
        candidates = {
            k: t for k, t in self._last_access.items() if k != "default"
        }
        if not candidates:
            return
        oldest = min(candidates, key=candidates.get)
        self._sessions.pop(oldest, None)
        self._last_access.pop(oldest, None)


_manager = RuntimeManager()
_manager.get()


def _get_runtime(request: Request) -> CySentRuntime:
    session_id = request.headers.get("x-session-id", "default")
    return _manager.get(session_id)


def _default_runtime() -> CySentRuntime:
    return _manager.get()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
def get_state(request: Request) -> Dict[str, Any]:
    return _get_runtime(request).snapshot_state()


@app.post("/reset")
def reset(req: ResetRequest, request: Request) -> Dict[str, Any]:
    rt = _get_runtime(request)
    with rt.state_lock:
        if req.action_source in VALID_AGENT_NAMES:
            rt.agent_router.switch_agent(req.action_source)

        _, info = rt.env.reset(
            seed=req.seed,
            options={
                "scenario": req.scenario,
                "difficulty": req.difficulty,
                "attacker": req.attacker,
                "strategy_mode": req.strategy_mode,
                "action_source": req.action_source,
                "intelligence_enabled": req.intelligence_enabled,
            },
        )
        rt.current_episode_id = str(uuid.uuid4())
        rt.last_info = info

        return {
            "episode_id": rt.current_episode_id,
            "step": info.get("step", 0),
            "network_risk": info.get("network_risk", 0.0),
            "risk_breakdown": info.get("risk_breakdown", {}),
            "assets": info.get("assets", []),
            "last_action": info.get("last_action", "reset"),
            "red_log": info.get("red_log", {}),
            "profile": info.get("profile", {}),
            "intelligence": info.get("intelligence", {}),
            "events": info.get("events", []),
            "narrative": info.get("narrative", ""),
            "termination_reason": info.get("termination_reason", "active"),
        }


@app.post("/step")
def step(request: Request) -> Dict[str, Any]:
    rt = _get_runtime(request)
    with rt.state_lock:
        obs = rt.env._get_observation()
        state = rt.snapshot_state()

        action = rt.agent_router.predict_action(obs, state)

        _, reward, terminated, truncated, info = rt.env.step(action)
        info["action_name"] = ACTION_NAMES[action]
        info["reward"] = float(reward)
        info["selected_action"] = action
        info["active_agent"] = rt.agent_router.get_active_agent_name()

        if terminated or truncated:
            rt.replays[rt.current_episode_id] = list(rt.env.replay)
            rt.current_episode_id = str(uuid.uuid4())
            rt.env.reset()

        rt.last_info = info
        return {
            "episode_id": rt.current_episode_id,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "action_name": info["action_name"],
            "selected_action": info.get("selected_action"),
            "active_agent": info.get("active_agent"),
            "network_risk": info["network_risk"],
            "risk_breakdown": info.get("risk_breakdown", {}),
            "assets": info["assets"],
            "red_log": info["red_log"],
            "profile": info.get("profile", {}),
            "intelligence": info.get("intelligence", {}),
            "events": info.get("events", []),
            "narrative": info.get("narrative", ""),
            "metrics": info["metrics"],
            "termination_reason": info.get("termination_reason", "active"),
        }


@app.get("/metrics")
def get_metrics(request: Request) -> Dict[str, Any]:
    rt = _get_runtime(request)
    with rt.state_lock:
        return rt.last_info.get("metrics", {})


@app.get("/training-status")
def training_status() -> Dict[str, Any]:
    return _default_runtime().training


@app.get("/replay/{episode_id}", response_model=ReplayResponse)
def get_replay(episode_id: str, request: Request) -> ReplayResponse:
    rt = _get_runtime(request)
    if episode_id not in rt.replays:
        raise HTTPException(status_code=404, detail="episode replay not found")
    return ReplayResponse(episode_id=episode_id, events=rt.replays[episode_id])


@app.get("/replay/{episode_id}/export")
def export_replay(episode_id: str, request: Request) -> JSONResponse:
    rt = _get_runtime(request)
    if episode_id not in rt.replays:
        raise HTTPException(status_code=404, detail="episode replay not found")
    payload = {"episode_id": episode_id, "events": rt.replays[episode_id]}
    headers = {"Content-Disposition": f'attachment; filename="replay_{episode_id}.json"'}
    return JSONResponse(content=payload, headers=headers)


@app.post("/train")
def start_training(req: TrainRequest) -> Dict[str, Any]:
    rt = _default_runtime()
    if rt.training["running"]:
        raise HTTPException(status_code=409, detail="training already in progress")

    rt.training["running"] = True
    rt.training["last_error"] = None

    def _job() -> None:
        try:
            model_base = "backend/train/artifacts/cysent_ppo"
            result = train(
                total_timesteps=req.timesteps,
                model_path=model_base,
                seed=req.seed,
                max_steps=req.max_steps,
            )
            eval_summary = evaluate(
                model_path=model_base + ".zip",
                episodes=25,
                max_steps=req.max_steps,
                seed=req.seed,
            )

            artifact = Path("backend/train/artifacts")
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "api_latest_eval.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

            rt.training["last_run"] = result
            rt.training["evaluation"] = eval_summary
        except Exception as exc:  # pragma: no cover
            rt.training["last_error"] = str(exc)
        finally:
            rt.training["running"] = False

    thread = threading.Thread(target=_job, daemon=True)
    thread.start()

    return {"message": "training started", "timesteps": req.timesteps}


@app.post("/benchmark")
def run_benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
    result = build_benchmark(
        episodes=req.episodes,
        max_steps=req.max_steps,
        seed=req.seed,
        agents=req.agents,
        seeds=req.seeds,
        stress=req.stress,
        baseline_model=req.baseline_model,
        tuned_model=req.tuned_model,
        cloud_model=req.cloud_model,
        output=req.output,
    )
    _default_runtime().training["benchmark"] = result
    return result


@app.get("/benchmark/export")
def export_benchmark(format: str = "json") -> FileResponse:
    fmt = format.lower()
    if fmt not in {"json", "csv"}:
        raise HTTPException(status_code=400, detail="format must be one of: json, csv")

    filename = "benchmark_table.json" if fmt == "json" else "benchmark_table.csv"
    path = Path("backend/train/artifacts/benchmark") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"benchmark export not found: {filename}")

    media_type = "application/json" if fmt == "json" else "text/csv"
    return FileResponse(path=path, media_type=media_type, filename=filename)
