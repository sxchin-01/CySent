from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from backend.env.security_env import CySentSecurityEnv, maybe_register_openenv_env

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "CySent_v1_locked",
    "description": "Locked baseline for reproducible RL experimentation.",
    "env": {
        "max_steps": 150,
        "n_envs": 4,
        "register_openenv": True,
    },
    "training": {
        "timesteps": 100000,
        "seed": 42,
    },
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 2.5e-4,
        "n_steps": 512,
        "batch_size": 256,
        "gamma": 0.997,
        "gae_lambda": 0.97,
        "ent_coef": 0.003,
        "clip_range": 0.20,
        "vf_coef": 0.6,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
    },
    "evaluation": {
        "max_steps": 150,
        "episodes": 50,
        "seed": 42,
        "same_episode_seeds": True,
        "normalization": "disabled",
    },
    "scoring": {
        "weights": {
            "breach_rate": 0.35,
            "uptime": 0.25,
            "final_risk": 0.20,
            "survival_turns": 0.10,
            "episode_reward": 0.10,
        }
    },
}


def linear_schedule(initial_value: float):
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return _schedule


class CySentMetricsCallback(BaseCallback):
    """Collect and persist training metrics for experiment tracking."""

    def __init__(self, output_dir: str, flush_freq: int = 2000, verbose: int = 0) -> None:
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

        row = {
            "timesteps": float(self.num_timesteps),
            "step_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
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
        df.to_csv(self.output_dir / "training_metrics.csv", index=False)

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


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = config
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        out[key.strip()] = _parse_value(raw.strip())
    return out


def _dict_diff(base: Dict[str, Any], cur: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    keys = sorted(set(base.keys()) | set(cur.keys()))
    for key in keys:
        base_v = base.get(key)
        cur_v = cur.get(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(base_v, dict) and isinstance(cur_v, dict):
            diff.update(_dict_diff(base_v, cur_v, path))
        elif base_v != cur_v:
            diff[path] = {"baseline": base_v, "current": cur_v}
    return diff


def _load_config(config_path: str | None, overrides: List[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    baseline = copy.deepcopy(DEFAULT_CONFIG)
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a mapping/object at root")

        _deep_update(config, loaded)

    parsed_overrides = _parse_overrides(overrides)
    for key, value in parsed_overrides.items():
        _set_dotted(config, key, value)

    return config, baseline


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_run_dir(root: Path, run_name: str, seed: int) -> Path:
    base = root / f"{run_name}_seed{seed}"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base

    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{run_name}_seed{seed}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_env(seed: int, max_steps: int, n_envs: int, monitor_dir: str) -> VecEnv:
    return make_vec_env(
        CySentSecurityEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"max_steps": max_steps},
        monitor_dir=monitor_dir,
    )


def _build_eval_env(seed: int, max_steps: int) -> DummyVecEnv:
    return DummyVecEnv([lambda: Monitor(CySentSecurityEnv(max_steps=max_steps, seed=seed))])


def _set_deterministic(seed: int) -> None:
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _update_summary_registry(artifacts_root: Path, row: Dict[str, Any]) -> None:
    json_path = artifacts_root / "experiment_runs.json"
    csv_path = artifacts_root / "experiment_runs.csv"

    rows: List[Dict[str, Any]] = []
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = loaded
        except json.JSONDecodeError:
            rows = []

    rows.append(row)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def train(
    total_timesteps: int,
    model_path: str,
    seed: int,
    max_steps: int,
    n_envs: int = 4,
    run_name: str = "cysent_run",
    config_path: str | None = None,
    overrides: List[str] | None = None,
) -> Dict[str, Any]:
    overrides = overrides or []

    config, baseline = _load_config(config_path=config_path, overrides=overrides)

    # CLI arguments remain authoritative when explicitly provided.
    config["training"]["timesteps"] = int(total_timesteps)
    config["training"]["seed"] = int(seed)
    config["env"]["max_steps"] = int(max_steps)
    config["env"]["n_envs"] = int(n_envs)

    if bool(config["env"].get("register_openenv", True)):
        maybe_register_openenv_env()

    _set_deterministic(int(config["training"]["seed"]))

    model_alias = Path(model_path)
    artifacts_root = model_alias.parent if model_alias.parent != Path("") else Path("backend/train/artifacts")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    run_dir = _ensure_run_dir(artifacts_root, run_name=run_name, seed=int(config["training"]["seed"]))
    run_runs_dir = run_dir / "runs"
    run_logs_dir = run_dir / "logs"
    run_checkpoints_dir = run_dir / "checkpoints"
    run_monitor_dir = run_dir / "monitor"

    env = _build_env(
        seed=int(config["training"]["seed"]),
        max_steps=int(config["env"]["max_steps"]),
        n_envs=int(config["env"]["n_envs"]),
        monitor_dir=str(run_monitor_dir),
    )
    eval_env = _build_eval_env(
        seed=int(config["training"]["seed"]) + 999,
        max_steps=int(config["env"]["max_steps"]),
    )

    ppo_cfg = config["ppo"]
    model = PPO(
        ppo_cfg.get("policy", "MlpPolicy"),
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=linear_schedule(float(ppo_cfg["learning_rate"])),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        clip_range=float(ppo_cfg["clip_range"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        target_kl=float(ppo_cfg["target_kl"]),
        tensorboard_log=str(run_runs_dir),
        seed=int(config["training"]["seed"]),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000 // max(int(config["env"]["n_envs"]), 1), 1),
        save_path=str(run_checkpoints_dir),
        name_prefix="cysent_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_logs_dir),
        eval_freq=max(10000 // max(int(config["env"]["n_envs"]), 1), 1),
        deterministic=True,
        render=False,
    )

    metrics_callback = CySentMetricsCallback(output_dir=str(run_dir), flush_freq=2500)
    callbacks = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    model.learn(
        total_timesteps=int(config["training"]["timesteps"]),
        callback=callbacks,
        tb_log_name=f"{run_name}_seed{int(config['training']['seed'])}",
    )

    run_model = run_dir / "model"
    model.save(str(run_model))

    # Compatibility alias path used in existing commands.
    alias_zip = model_alias.with_suffix(".zip")
    run_zip = run_model.with_suffix(".zip")
    if run_zip.exists() and alias_zip.resolve() != run_zip.resolve():
        alias_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_zip, alias_zip)

    # Explicit normalization pairing artifact (disabled baseline, no hidden state).
    vecnorm_payload = {
        "enabled": False,
        "reason": "VecNormalize disabled in CySent_v1_locked to keep PPO pipeline unchanged.",
    }
    with (run_dir / "vecnormalize.pkl").open("wb") as f:
        pickle.dump(vecnorm_payload, f)

    env.close()
    eval_env.close()

    tracked_files = [
        Path("backend/env/security_env.py"),
        Path("backend/env/reward.py"),
        Path("backend/env/risk.py"),
        Path("backend/train/train_ppo.py"),
    ]
    file_hashes = {str(p): _file_sha256(p) for p in tracked_files if p.exists()}

    config_snapshot = {
        "config_name": config.get("name", "CySent_v1_locked"),
        "run_name": run_name,
        "seed": int(config["training"]["seed"]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "resolved_config": config,
        "diff_from_locked_baseline": _dict_diff(baseline, config),
        "tracked_file_hashes": file_hashes,
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    metrics = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_path": str(run_zip),
        "model_alias_path": str(alias_zip),
        "total_timesteps": int(config["training"]["timesteps"]),
        "seed": int(config["training"]["seed"]),
        "max_steps": int(config["env"]["max_steps"]),
        "n_envs": int(config["env"]["n_envs"]),
        "tensorboard_log_dir": str(run_runs_dir),
        "checkpoint_dir": str(run_checkpoints_dir),
        "best_model_dir": str(run_dir / "best_model"),
        "monitor_dir": str(run_monitor_dir),
        "vecnormalize_path": str(run_dir / "vecnormalize.pkl"),
        "config_path": str(run_dir / "config.json"),
        "metrics_path": str(run_dir / "metrics.json"),
        "training_curves": {
            "reward": str(run_dir / "reward_curve.png"),
            "breach": str(run_dir / "breach_curve.png"),
            "uptime": str(run_dir / "uptime_curve.png"),
            "risk": str(run_dir / "risk_curve.png"),
        },
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _update_summary_registry(
        artifacts_root,
        {
            "run_name": run_name,
            "seed": int(config["training"]["seed"]),
            "run_dir": str(run_dir),
            "model_path": str(run_zip),
            "timesteps": int(config["training"]["timesteps"]),
            "max_steps": int(config["env"]["max_steps"]),
        },
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CySent PPO policy")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--model-path", type=str, default="backend/train/artifacts/cysent_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="cysent_run")
    parser.add_argument("--config", type=str, default="configs/v1_locked.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="One-variable experiment override, e.g. --override ppo.ent_coef=0.005",
    )

    args = parser.parse_args()
    summary = train(
        total_timesteps=args.timesteps,
        model_path=args.model_path,
        seed=args.seed,
        max_steps=args.max_steps,
        n_envs=args.n_envs,
        run_name=args.run_name,
        config_path=args.config,
        overrides=args.override,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
