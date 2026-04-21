from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    PPO = None


VALID_ACTIONS: List[str] = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compact_input(info: Dict[str, Any]) -> str:
    risk = _safe_float(info.get("network_risk", 0.0))
    rb = info.get("risk_breakdown", {}) if isinstance(info.get("risk_breakdown", {}), dict) else {}
    red = info.get("red_log", {}) if isinstance(info.get("red_log", {}), dict) else {}
    assets = info.get("assets", []) if isinstance(info.get("assets", []), list) else []

    compromised = sum(1 for a in assets if bool(a.get("compromised", False)))
    infected = sum(1 for a in assets if bool(a.get("infected", False)))
    isolated = sum(1 for a in assets if bool(a.get("isolated", False)))

    attack = str(red.get("attack", "unknown"))
    target = str(red.get("target", "unknown"))

    phrases = [
        f"risk={risk:.2f}",
        f"attack={attack}",
        f"target={target}",
        f"compromised={compromised}",
        f"infected={infected}",
        f"isolated={isolated}",
        f"credential_exposure={_safe_float(rb.get('credential_exposure', 0.0)):.2f}",
        f"patch_debt={_safe_float(rb.get('patch_debt', 0.0)):.2f}",
        f"segmentation_gap={_safe_float(rb.get('segmentation_gap', 0.0)):.2f}",
        f"monitoring_weakness={_safe_float(rb.get('monitoring_weakness', 0.0)):.2f}",
        f"ransomware_spread={_safe_float(rb.get('ransomware_spread', 0.0)):.2f}",
    ]
    return ", ".join(phrases)


def _heuristic_action(info: Dict[str, Any], rng: np.random.Generator) -> str:
    risk = _safe_float(info.get("network_risk", 0.0))
    rb = info.get("risk_breakdown", {}) if isinstance(info.get("risk_breakdown", {}), dict) else {}
    red = info.get("red_log", {}) if isinstance(info.get("red_log", {}), dict) else {}
    assets = info.get("assets", []) if isinstance(info.get("assets", []), list) else []

    compromised = sum(1 for a in assets if bool(a.get("compromised", False)))
    infected = sum(1 for a in assets if bool(a.get("infected", False)))

    credential_exposure = _safe_float(rb.get("credential_exposure", 0.0))
    patch_debt = _safe_float(rb.get("patch_debt", 0.0))
    segmentation_gap = _safe_float(rb.get("segmentation_gap", 0.0))
    monitoring_weakness = _safe_float(rb.get("monitoring_weakness", 0.0))
    ransomware_spread = _safe_float(rb.get("ransomware_spread", 0.0))

    attack = str(red.get("attack", "")).lower()
    target = str(red.get("target", "")).lower()

    if compromised > 0 and (risk > 0.72 or infected > 0):
        if rng.random() < 0.5:
            return "investigate_top_alert"
        return "isolate_suspicious_host"

    if "credential" in attack or "phishing" in attack or credential_exposure > 0.55:
        if rng.random() < 0.8:
            return "rotate_credentials"
        return "phishing_training"

    if ransomware_spread > 0.45:
        if infected > 0 and rng.random() < 0.7:
            return "restore_backup"
        return "investigate_top_alert"

    if ("finance" in target and risk > 0.55) or segmentation_gap > 0.55:
        return "segment_finance_database"

    if patch_debt > 0.55:
        return "patch_auth_server" if rng.random() < 0.7 else "patch_web_server"

    if monitoring_weakness > 0.48:
        return "increase_monitoring"

    if infected > 0 and risk > 0.45:
        return "isolate_suspicious_host"

    if risk < 0.25 and compromised == 0 and infected == 0:
        return "do_nothing"

    return "increase_monitoring"


def _load_ppo(model_path: str | None):
    if not model_path or PPO is None:
        return None
    p = Path(model_path)
    if not p.exists():
        return None
    try:
        return PPO.load(str(p))
    except Exception:
        return None


def _state_to_record(info: Dict[str, Any], action_name: str) -> Dict[str, str]:
    return {
        "instruction": (
            "You are CySent BLUE defender. Choose exactly one valid defense action label "
            "from the CySent action schema."
        ),
        "input": _compact_input(info),
        "output": action_name,
    }


def build_dataset(
    output_path: Path,
    rows: int,
    seed: int,
    max_steps: int,
    ppo_model_path: str | None,
) -> Tuple[int, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = ["bank", "hospital", "saas", "government", "manufacturing"]
    attackers = ["ransomware_gang", "silent_apt", "credential_thief", "insider_saboteur", "botnet"]
    difficulties = ["easy", "medium", "hard"]

    env = CySentSecurityEnv(max_steps=max_steps, seed=seed)
    ppo_model = _load_ppo(ppo_model_path)

    records: List[Dict[str, str]] = []
    action_hist: Dict[str, int] = {name: 0 for name in VALID_ACTIONS}

    while len(records) < rows:
        scenario = scenarios[int(rng.integers(0, len(scenarios)))]
        attacker = attackers[int(rng.integers(0, len(attackers)))]
        difficulty = difficulties[int(rng.integers(0, len(difficulties)))]

        obs, info = env.reset(
            seed=int(rng.integers(1, 1_000_000)),
            options={
                "scenario": scenario,
                "attacker": attacker,
                "difficulty": difficulty,
                "action_source": "ppo_agent",
                "strategy_mode": "balanced",
                "intelligence_enabled": True,
            },
        )

        done = False
        truncated = False
        local_steps = 0

        while not (done or truncated) and len(records) < rows:
            label = _heuristic_action(info, rng)
            if label not in action_hist:
                label = "do_nothing"
            records.append(_state_to_record(info, label))
            action_hist[label] += 1

            if ppo_model is not None and rng.random() < 0.35:
                action, _ = ppo_model.predict(obs, deterministic=True)
                step_action = int(action)
            else:
                step_action = int(env.action_space.sample())

            obs, _, done, truncated, info = env.step(step_action)
            local_steps += 1
            if local_steps >= max_steps:
                break

    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return len(records), action_hist


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CySent instruction dataset for Colab/Unsloth fine-tuning")
    parser.add_argument("--output", type=str, default="datasets/cysent_action_dataset.jsonl")
    parser.add_argument("--rows", type=int, default=1200, help="Total rows (recommended: 500-3000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--ppo-model", type=str, default="backend/train/artifacts/best_model/best_model.zip")
    args = parser.parse_args()

    target_rows = max(500, min(3000, int(args.rows)))
    out_path = Path(args.output)

    count, hist = build_dataset(
        output_path=out_path,
        rows=target_rows,
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        ppo_model_path=args.ppo_model,
    )

    print(f"Wrote {count} rows to {out_path}")
    print("Action distribution:")
    for name in VALID_ACTIONS:
        print(f"  {name}: {hist.get(name, 0)}")


if __name__ == "__main__":
    main()
