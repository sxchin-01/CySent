from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agents import AgentRouter
from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv
from backend.train.benchmark import build_benchmark
from backend.train.evaluate import evaluate
from backend.train.train_ppo import train

app = FastAPI(title="CySent API", version="1.0.0")

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
    action_source: str = "ppo_ai"
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
    agents: List[str] = Field(default_factory=lambda: ["ppo", "hf_llm", "random"])
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


runtime = CySentRuntime()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return runtime.snapshot_state()


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    with runtime.state_lock:
        # Switch agent based on action_source
        if req.action_source in ["ppo_agent", "hf_llm_agent"]:
            runtime.agent_router.switch_agent(req.action_source)

        _, info = runtime.env.reset(
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
        runtime.current_episode_id = str(uuid.uuid4())
        runtime.last_info = info

        return {
            "episode_id": runtime.current_episode_id,
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
def step() -> Dict[str, Any]:
    with runtime.state_lock:
        # Get current observation and state for agent decision
        obs = runtime.env._get_observation()
        state = runtime.snapshot_state()

        # Use agent router to select action
        action = runtime.agent_router.predict_action(obs, state)

        _, reward, terminated, truncated, info = runtime.env.step(action)
        info["action_name"] = ACTION_NAMES[action]
        info["reward"] = float(reward)
        info["selected_action"] = action
        info["active_agent"] = runtime.agent_router.get_active_agent_name()

        if terminated or truncated:
            runtime.replays[runtime.current_episode_id] = list(runtime.env.replay)
            runtime.current_episode_id = str(uuid.uuid4())
            runtime.env.reset()

        runtime.last_info = info
        return {
            "episode_id": runtime.current_episode_id,
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
def get_metrics() -> Dict[str, Any]:
    with runtime.state_lock:
        return runtime.last_info.get("metrics", {})


@app.get("/training-status")
def training_status() -> Dict[str, Any]:
    return runtime.training


@app.get("/replay/{episode_id}", response_model=ReplayResponse)
def get_replay(episode_id: str) -> ReplayResponse:
    if episode_id not in runtime.replays:
        raise HTTPException(status_code=404, detail="episode replay not found")
    return ReplayResponse(episode_id=episode_id, events=runtime.replays[episode_id])


@app.post("/train")
def start_training(req: TrainRequest) -> Dict[str, Any]:
    if runtime.training["running"]:
        raise HTTPException(status_code=409, detail="training already in progress")

    runtime.training["running"] = True
    runtime.training["last_error"] = None

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

            runtime.training["last_run"] = result
            runtime.training["evaluation"] = eval_summary
        except Exception as exc:  # pragma: no cover
            runtime.training["last_error"] = str(exc)
        finally:
            runtime.training["running"] = False

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
    runtime.training["benchmark"] = result
    return result
