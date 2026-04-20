from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from backend.env.security_env import CySentSecurityEnv
from backend.train.evaluate import aggregate, run_episode


def evaluate_policy(
    name: str,
    policy_kind: str,
    model_path: Optional[str],
    episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    env = CySentSecurityEnv(max_steps=max_steps, seed=seed)

    model: PPO | None = None
    if policy_kind != "random":
        if model_path is None or not Path(model_path).exists():
            return {
                "policy": name,
                "status": "missing_model",
                "model_path": model_path,
            }
        model = PPO.load(model_path)

    rows: List[Dict[str, Any]] = []
    for i in range(episodes):
        env.seed(seed + i)
        rows.append(run_episode(env, policy=policy_kind, model=model))

    return {
        "policy": name,
        "status": "ok",
        "model_path": model_path,
        "summary": aggregate(rows),
        "episodes": rows,
    }


def build_benchmark(
    episodes: int,
    max_steps: int,
    seed: int,
    baseline_model: Optional[str],
    tuned_model: Optional[str],
    cloud_model: Optional[str],
    output: str,
) -> Dict[str, Any]:
    results = {
        "random_policy": evaluate_policy(
            name="random_policy",
            policy_kind="random",
            model_path=None,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
        ),
        "ppo_baseline": evaluate_policy(
            name="ppo_baseline",
            policy_kind="trained",
            model_path=baseline_model,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed + 1_000,
        ),
        "ppo_tuned": evaluate_policy(
            name="ppo_tuned",
            policy_kind="trained",
            model_path=tuned_model,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed + 2_000,
        ),
        "cloud_trained": evaluate_policy(
            name="cloud_trained",
            policy_kind="trained",
            model_path=cloud_model,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed + 3_000,
        ),
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _plot_benchmark(results, output_path.parent)
    return results


def _plot_benchmark(results: Dict[str, Any], output_dir: Path) -> None:
    policy_order = ["random_policy", "ppo_baseline", "ppo_tuned", "cloud_trained"]
    labels = []

    reward_vals: List[float] = []
    breach_vals: List[float] = []
    uptime_vals: List[float] = []
    risk_vals: List[float] = []

    for key in policy_order:
        row = results[key]
        labels.append(key.replace("_", " "))
        if row.get("status") != "ok":
            reward_vals.append(np.nan)
            breach_vals.append(np.nan)
            uptime_vals.append(np.nan)
            risk_vals.append(np.nan)
            continue

        summary = row["summary"]
        reward_vals.append(float(summary.get("avg_episode_reward", np.nan)))
        breach_vals.append(float(summary.get("avg_breach_rate", np.nan)))
        uptime_vals.append(float(summary.get("avg_uptime", np.nan)))
        risk_vals.append(float(summary.get("avg_final_risk", np.nan)))

    _bar_chart(output_dir / "benchmark_reward.png", labels, reward_vals, "Avg Episode Reward")
    _bar_chart(output_dir / "benchmark_breach.png", labels, breach_vals, "Avg Breach Rate", y_lim=(0.0, 1.0))
    _bar_chart(output_dir / "benchmark_uptime.png", labels, uptime_vals, "Avg Uptime", y_lim=(0.0, 1.0))
    _bar_chart(output_dir / "benchmark_risk.png", labels, risk_vals, "Avg Final Risk", y_lim=(0.0, 1.0))


def _bar_chart(path: Path, labels: List[str], values: List[float], title: str, y_lim: tuple[float, float] | None = None) -> None:
    fig = plt.figure(figsize=(9, 4.8))
    x = np.arange(len(labels))
    plt.bar(x, values, color=["#991b1b", "#0f766e", "#166534", "#0369a1"])
    plt.xticks(x, labels, rotation=15, ha="right")
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CySent policies")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-model", type=str, default="backend/train/artifacts/cysent_ppo.zip")
    parser.add_argument("--tuned-model", type=str, default="backend/train/artifacts/best_model/best_model.zip")
    parser.add_argument("--cloud-model", type=str, default="")
    parser.add_argument("--output", type=str, default="backend/train/artifacts/benchmark_summary.json")
    args = parser.parse_args()

    summary = build_benchmark(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        baseline_model=args.baseline_model,
        tuned_model=args.tuned_model,
        cloud_model=args.cloud_model if args.cloud_model else None,
        output=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
