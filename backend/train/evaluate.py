from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from backend.env.risk import breach_rate, uptime_ratio
from backend.env.security_env import CySentSecurityEnv


def run_episode(env: CySentSecurityEnv, policy: str, model: PPO | None = None) -> Dict[str, Any]:
    obs, info = env.reset()
    done = False
    trunc = False
    turn_count = 0
    terminal_reason = "max_steps"

    rewards: List[float] = []
    while not (done or trunc):
        if policy == "random":
            action = env.action_space.sample()
        else:
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, trunc, info = env.step(int(action))
        turn_count += 1
        rewards.append(float(reward))
        if done:
            terminal_reason = str(info.get("termination_reason", "terminated"))
        elif trunc:
            terminal_reason = "max_steps"

    assets = info["assets"]
    return {
        "episode_reward": float(np.sum(rewards)),
        "breach_rate": float(breach_rate(assets)),
        "survival_turns": int(turn_count),
        "uptime": float(uptime_ratio(assets)),
        "final_risk": float(info["network_risk"]),
        "terminated": bool(done),
        "truncated": bool(trunc),
        "terminal_reason": terminal_reason,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["episode_reward", "breach_rate", "survival_turns", "uptime", "final_risk"]
    out: Dict[str, Any] = {"episodes": len(rows)}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=np.float32)
        out[f"avg_{key}"] = float(values.mean())
        out[f"std_{key}"] = float(values.std())
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_security_score(summary: Dict[str, Any], max_steps: int) -> float:
    # Component scores normalized to 0..100 where higher is better.
    breach_score = 100.0 * (1.0 - _clamp(float(summary.get("avg_breach_rate", 1.0)), 0.0, 1.0))
    uptime_score = 100.0 * _clamp(float(summary.get("avg_uptime", 0.0)), 0.0, 1.0)
    risk_score = 100.0 * (1.0 - _clamp(float(summary.get("avg_final_risk", 1.0)), 0.0, 1.0))

    survival_ratio = float(summary.get("avg_survival_turns", 0.0)) / max(float(max_steps), 1.0)
    survival_score = 100.0 * _clamp(survival_ratio, 0.0, 1.0)

    # Reward normalization range chosen for CySent stability.
    reward = float(summary.get("avg_episode_reward", 0.0))
    reward_score = 100.0 * _clamp((reward + 50.0) / 200.0, 0.0, 1.0)

    score = (
        0.35 * breach_score
        + 0.25 * uptime_score
        + 0.20 * risk_score
        + 0.10 * survival_score
        + 0.10 * reward_score
    )
    return float(score)


def _resolve_run_dir(model_path: str, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)

    mp = Path(model_path)
    if mp.suffix.lower() == ".zip":
        parent = mp.parent
        if (parent / "config.json").exists():
            return parent

    return mp.parent


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}

    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        return {}

    return {}


def _load_vecnormalize_metadata(run_dir: Path) -> Dict[str, Any]:
    vec_path = run_dir / "vecnormalize.pkl"
    if not vec_path.exists():
        return {"exists": False, "enabled": False}

    try:
        with vec_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            return {"exists": True, **payload}
    except Exception:
        pass

    return {"exists": True, "enabled": None, "reason": "unknown_format"}


def evaluate(
    model_path: str,
    episodes: int,
    max_steps: int,
    seed: int,
    run_dir: str | None = None,
) -> Dict[str, Any]:
    resolved_run_dir = _resolve_run_dir(model_path=model_path, run_dir=run_dir)
    run_config = _load_run_config(resolved_run_dir)
    vec_meta = _load_vecnormalize_metadata(resolved_run_dir)

    trained_env = CySentSecurityEnv(max_steps=max_steps, seed=seed)
    random_env = CySentSecurityEnv(max_steps=max_steps, seed=seed)

    model = PPO.load(model_path)

    trained_runs: List[Dict[str, Any]] = []
    random_runs: List[Dict[str, Any]] = []

    episode_seeds = [seed + i for i in range(episodes)]
    for episode_seed in episode_seeds:
        trained_env.seed(episode_seed)
        random_env.seed(episode_seed)
        trained_runs.append(run_episode(trained_env, policy="trained", model=model))
        random_runs.append(run_episode(random_env, policy="random"))

    trained_max_steps_violations = sum(1 for r in trained_runs if r["survival_turns"] > max_steps)
    random_max_steps_violations = sum(1 for r in random_runs if r["survival_turns"] > max_steps)

    trained_summary = aggregate(trained_runs)
    random_summary = aggregate(random_runs)

    trained_score = compute_security_score(trained_summary, max_steps=max_steps)
    random_score = compute_security_score(random_summary, max_steps=max_steps)

    return {
        "trained_policy": trained_summary,
        "random_policy": random_summary,
        "security_score": {
            "trained": trained_score,
            "random": random_score,
            "delta": float(trained_score - random_score),
            "weights": {
                "breach_rate": 0.35,
                "uptime": 0.25,
                "final_risk": 0.20,
                "survival_turns": 0.10,
                "episode_reward": 0.10,
            },
        },
        "validation": {
            "max_steps": max_steps,
            "trained_overflow_episodes": trained_max_steps_violations,
            "random_overflow_episodes": random_max_steps_violations,
            "episode_seeds": episode_seeds,
            "fair_comparison": {
                "same_episode_seeds": True,
                "same_max_steps": True,
                "same_metric_schema": True,
                "normalization_pairing": vec_meta,
            },
            "run_dir": str(resolved_run_dir),
            "config_name": run_config.get("config_name"),
            "run_name": run_config.get("run_name"),
            "seed": run_config.get("seed", seed),
        },
    }


def write_summary_row(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path_json = output_dir / "evaluation_runs.json"
    row_path_csv = output_dir / "evaluation_runs.csv"

    row = {
        "run_name": summary["validation"].get("run_name"),
        "seed": summary["validation"].get("seed"),
        "reward": summary["trained_policy"].get("avg_episode_reward"),
        "breach": summary["trained_policy"].get("avg_breach_rate"),
        "uptime": summary["trained_policy"].get("avg_uptime"),
        "risk": summary["trained_policy"].get("avg_final_risk"),
        "score": summary["security_score"].get("trained"),
    }

    rows: List[Dict[str, Any]] = []
    if row_path_json.exists():
        try:
            loaded = json.loads(row_path_json.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = loaded
        except json.JSONDecodeError:
            rows = []

    rows.append(row)
    row_path_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = ["run_name", "seed", "reward", "breach", "uptime", "risk", "score"]
    with row_path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CySent policy vs random")
    parser.add_argument("--model-path", type=str, default="backend/train/artifacts/cysent_ppo.zip")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="backend/train/artifacts/eval_summary.json")
    args = parser.parse_args()

    summary = evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        run_dir=args.run_dir or None,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_summary_row(summary, out.parent)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
