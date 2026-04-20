from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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


def evaluate(model_path: str, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    trained_env = CySentSecurityEnv(max_steps=max_steps, seed=seed)
    random_env = CySentSecurityEnv(max_steps=max_steps, seed=seed + 7)

    model = PPO.load(model_path)

    trained_runs: List[Dict[str, Any]] = []
    random_runs: List[Dict[str, Any]] = []
    for i in range(episodes):
        trained_env.seed(seed + i)
        random_env.seed(seed + 10_000 + i)
        trained_runs.append(run_episode(trained_env, policy="trained", model=model))
        random_runs.append(run_episode(random_env, policy="random"))

    trained_max_steps_violations = sum(1 for r in trained_runs if r["survival_turns"] > max_steps)
    random_max_steps_violations = sum(1 for r in random_runs if r["survival_turns"] > max_steps)

    return {
        "trained_policy": aggregate(trained_runs),
        "random_policy": aggregate(random_runs),
        "validation": {
            "max_steps": max_steps,
            "trained_overflow_episodes": trained_max_steps_violations,
            "random_overflow_episodes": random_max_steps_violations,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CySent policy vs random")
    parser.add_argument("--model-path", type=str, default="backend/train/artifacts/cysent_ppo.zip")
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
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
