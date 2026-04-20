from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from backend.env.security_env import CySentSecurityEnv, maybe_register_openenv_env


def linear_schedule(initial_value: float):
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return _schedule


class CySentMetricsCallback(BaseCallback):
    """Collect and persist training metrics for experiment tracking."""

    def __init__(self, output_dir: str, flush_freq: int = 2_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.flush_freq = flush_freq
        self.rows: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", np.array([], dtype=np.float32))
        if len(infos) == 0:
            return True

        risk_values: List[float] = []
        breach_values: List[float] = []
        uptime_values: List[float] = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            risk_values.append(float(info.get("network_risk", 0.0)))

            assets = info.get("assets", [])
            if isinstance(assets, list) and assets:
                breaches = sum(1 for a in assets if bool(a.get("compromised", False))) / len(assets)
                uptime = sum(1 for a in assets if bool(a.get("uptime_status", False))) / len(assets)
            else:
                breaches = 0.0
                uptime = 1.0

            breach_values.append(float(breaches))
            uptime_values.append(float(uptime))

        step_reward = float(np.mean(rewards)) if len(rewards) else 0.0
        row = {
            "timesteps": float(self.num_timesteps),
            "step_reward": step_reward,
            "network_risk": float(np.mean(risk_values)) if risk_values else 0.0,
            "breach_rate": float(np.mean(breach_values)) if breach_values else 0.0,
            "uptime": float(np.mean(uptime_values)) if uptime_values else 1.0,
        }
        self.rows.append(row)

        if self.num_timesteps % self.flush_freq == 0:
            self._flush()

        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)
        csv_path = self.output_dir / "training_metrics.csv"
        df.to_csv(csv_path, index=False)

        self._plot(df, "step_reward", "reward_curve.png", "Step Reward")
        self._plot(df, "breach_rate", "breach_curve.png", "Breach Rate")
        self._plot(df, "uptime", "uptime_curve.png", "Uptime")
        self._plot(df, "network_risk", "risk_curve.png", "Network Risk")

    def _plot(self, df: pd.DataFrame, y_col: str, file_name: str, y_label: str) -> None:
        fig = plt.figure(figsize=(9, 4.5))
        plt.plot(df["timesteps"], df[y_col], linewidth=2.0)
        plt.title(f"CySent Training {y_label}")
        plt.xlabel("Timesteps")
        plt.ylabel(y_label)
        plt.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.output_dir / file_name, dpi=140)
        plt.close(fig)


def build_env(seed: int, max_steps: int, n_envs: int, monitor_dir: str) -> VecEnv:
    env = make_vec_env(
        CySentSecurityEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"max_steps": max_steps},
        monitor_dir=monitor_dir,
    )
    return env


def build_eval_env(seed: int, max_steps: int) -> DummyVecEnv:
    def _make() -> Monitor:
        return Monitor(CySentSecurityEnv(max_steps=max_steps, seed=seed))

    return DummyVecEnv([_make])


def train(
    total_timesteps: int,
    model_path: str,
    seed: int,
    max_steps: int,
    n_envs: int = 4,
) -> Dict[str, Any]:
    maybe_register_openenv_env()

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    runs_dir = out.parent / "runs"
    logs_dir = out.parent / "logs"
    checkpoints_dir = out.parent / "checkpoints"
    monitor_dir = out.parent / "monitor"

    env = build_env(seed=seed, max_steps=max_steps, n_envs=n_envs, monitor_dir=str(monitor_dir))
    eval_env = build_eval_env(seed=seed + 999, max_steps=max_steps)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=linear_schedule(2.5e-4),
        n_steps=512,
        batch_size=256,
        gamma=0.997,
        gae_lambda=0.97,
        ent_coef=0.003,
        clip_range=0.20,
        vf_coef=0.6,
        max_grad_norm=0.5,
        target_kl=0.03,
        tensorboard_log=str(runs_dir),
        seed=seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5_000 // max(n_envs, 1), 1),
        save_path=str(checkpoints_dir),
        name_prefix="cysent_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out.parent / "best_model"),
        log_path=str(logs_dir),
        eval_freq=max(10_000 // max(n_envs, 1), 1),
        deterministic=True,
        render=False,
    )

    metrics_callback = CySentMetricsCallback(output_dir=str(out.parent), flush_freq=2_500)
    callbacks = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="cysent_ppo")

    model.save(str(out))

    env.close()
    eval_env.close()

    metrics = {
        "model_path": str(out),
        "total_timesteps": total_timesteps,
        "seed": seed,
        "max_steps": max_steps,
        "n_envs": n_envs,
        "tensorboard_log_dir": str(runs_dir),
        "checkpoint_dir": str(checkpoints_dir),
        "best_model_dir": str(out.parent / "best_model"),
        "monitor_dir": str(monitor_dir),
        "tuned_hyperparameters": {
            "learning_rate": "linear_schedule(2.5e-4)",
            "n_steps": 512,
            "batch_size": 256,
            "gamma": 0.997,
            "gae_lambda": 0.97,
            "ent_coef": 0.003,
            "clip_range": 0.20,
            "target_kl": 0.03,
        },
    }

    with (out.parent / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CySent PPO policy")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-path", type=str, default="backend/train/artifacts/cysent_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--n-envs", type=int, default=4)

    args = parser.parse_args()
    summary = train(
        total_timesteps=args.timesteps,
        model_path=args.model_path,
        seed=args.seed,
        max_steps=args.max_steps,
        n_envs=args.n_envs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
