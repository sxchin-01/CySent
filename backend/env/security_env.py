from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import compute_reward
from .risk import compute_network_risk, compute_risk_breakdown
from .threat_engine import ThreatEngine

try:
    import openenv  # type: ignore
except Exception:  # pragma: no cover - optional runtime integration
    openenv = None


ASSET_NAMES = [
    "HR Systems",
    "Employee Email",
    "Web Server",
    "Auth Server",
    "Finance Database",
    "Backup Infrastructure",
    "SOC Monitoring Console",
]

ACTION_NAMES = {
    0: "do_nothing",
    1: "patch_hr_systems",
    2: "patch_web_server",
    3: "patch_auth_server",
    4: "rotate_credentials",
    5: "isolate_suspicious_host",
    6: "increase_monitoring",
    7: "restore_backup",
    8: "deploy_honeypot",
    9: "phishing_training",
    10: "investigate_top_alert",
    11: "segment_finance_database",
}


@dataclass
class EpisodeMetrics:
    total_reward: float = 0.0
    breaches: int = 0
    downtime_events: int = 0
    successful_attacks: int = 0
    prevented_attacks: int = 0


class CySentSecurityEnv(gym.Env[np.ndarray, int]):
    """Gymnasium-compatible cybersecurity RL environment for CySent."""

    metadata = {"render_modes": ["human"], "render_fps": 8}

    def __init__(self, max_steps: int = 150, seed: Optional[int] = None) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.segmented_finance = False
        self.honeypot_timer = 0
        self.training_mode = True
        self.previous_action_name = "reset"

        self.action_space = spaces.Discrete(12)

        # Per asset: patch, infected, isolated, compromised, criticality,
        # credential_risk, detection_level, backup_health, uptime_status
        per_asset_features = 9
        global_features = 4
        n_features = len(ASSET_NAMES) * per_asset_features + global_features

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_features,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.threat_engine = ThreatEngine(seed=seed)

        self.assets: List[Dict[str, Any]] = []
        self.episode_metrics = EpisodeMetrics()
        self.replay: List[Dict[str, Any]] = []
        self._reset_assets()

    def _reset_assets(self) -> None:
        criticality = {
            "HR Systems": 0.55,
            "Employee Email": 0.65,
            "Web Server": 0.78,
            "Auth Server": 0.88,
            "Finance Database": 1.00,
            "Backup Infrastructure": 0.82,
            "SOC Monitoring Console": 0.72,
        }

        self.assets = []
        for name in ASSET_NAMES:
            self.assets.append(
                {
                    "name": name,
                    "patch_level": float(self._rng.uniform(0.45, 0.80)),
                    "infected": False,
                    "isolated": False,
                    "compromised": False,
                    "criticality_score": criticality[name],
                    "credential_risk": float(self._rng.uniform(0.20, 0.50)),
                    "detection_level": float(self._rng.uniform(0.35, 0.70)),
                    "backup_health": float(self._rng.uniform(0.65, 0.95)),
                    "uptime_status": True,
                }
            )

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.threat_engine.seed(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.segmented_finance = False
        self.honeypot_timer = 0
        self.episode_metrics = EpisodeMetrics()
        self.replay = []
        self.previous_action_name = "reset"
        self._reset_assets()

        obs = self._get_observation()
        info = self._build_info(last_action="reset", red_log={})
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"invalid action: {action}"
        self.current_step += 1

        assets_prev = copy.deepcopy(self.assets)
        action_name, action_cost = self._apply_blue_action(action)

        if self.honeypot_timer > 0:
            self.honeypot_timer -= 1

        target_idx, attack = self.threat_engine.choose_attack(self.assets)
        red_log = self.threat_engine.apply_attack(
            assets=self.assets,
            target_idx=target_idx,
            attack_type=attack,
            segmented_finance=self.segmented_finance,
            honeypot_active=self.honeypot_timer > 0,
        )

        terminated, termination_reason = self._is_terminal()
        truncated = self.current_step >= self.max_steps

        reward = compute_reward(
            assets_prev=assets_prev,
            assets_curr=self.assets,
            action_name=action_name,
            red_log=red_log,
            action_cost=action_cost,
            context={
                "segmented_finance": self.segmented_finance,
                "last_action": action_name,
                "previous_action": self.previous_action_name,
                "red_success": bool(red_log.get("success", False)),
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": termination_reason,
            },
        )

        self.episode_metrics.total_reward += reward
        self.episode_metrics.successful_attacks += int(red_log.get("success", False))
        self.episode_metrics.prevented_attacks += int(not red_log.get("success", False))
        self.episode_metrics.breaches = sum(1 for a in self.assets if a["compromised"])
        self.episode_metrics.downtime_events = sum(1 for a in self.assets if not a["uptime_status"])

        info = self._build_info(last_action=action_name, red_log=red_log)
        info["termination_reason"] = termination_reason
        self.previous_action_name = action_name
        self.replay.append(
            {
                "step": self.current_step,
                "action": action_name,
                "red": red_log,
                "reward": reward,
                "network_risk": info["network_risk"],
            }
        )

        return self._get_observation(), float(reward), terminated, truncated, info

    def _is_terminal(self) -> Tuple[bool, str]:
        # End episode early on catastrophic breach state.
        compromised_critical = [
            a for a in self.assets if a["compromised"] and a["criticality_score"] >= 0.88
        ]
        if len(compromised_critical) >= 2:
            return True, "critical_breach"

        downtime_ratio = sum(1 for a in self.assets if not a["uptime_status"]) / len(self.assets)
        if downtime_ratio >= 0.5:
            return True, "downtime_cascade"

        return False, "active"

    def _apply_blue_action(self, action: int) -> Tuple[str, float]:
        action_name = ACTION_NAMES[action]

        def patch(name: str) -> None:
            asset = self._find_asset(name)
            asset["patch_level"] = float(np.clip(asset["patch_level"] + 0.25, 0.0, 1.0))
            asset["infected"] = False

        if action == 0:
            return action_name, 0.02
        if action == 1:
            patch("HR Systems")
            return action_name, 0.10
        if action == 2:
            patch("Web Server")
            return action_name, 0.10
        if action == 3:
            patch("Auth Server")
            return action_name, 0.12
        if action == 4:
            for asset in self.assets:
                asset["credential_risk"] = float(np.clip(asset["credential_risk"] - 0.20, 0.0, 1.0))
            return action_name, 0.18
        if action == 5:
            target = max(self.assets, key=lambda a: (float(a["infected"]), float(a["credential_risk"])))
            target["isolated"] = True
            target["uptime_status"] = False
            return action_name, 0.14
        if action == 6:
            for asset in self.assets:
                asset["detection_level"] = float(np.clip(asset["detection_level"] + 0.15, 0.0, 1.0))
            return action_name, 0.12
        if action == 7:
            for asset in self.assets:
                if not asset["uptime_status"]:
                    asset["uptime_status"] = True
                    asset["infected"] = False
                    asset["compromised"] = False
                asset["backup_health"] = float(np.clip(asset["backup_health"] + 0.20, 0.0, 1.0))
            return action_name, 0.20
        if action == 8:
            self.honeypot_timer = 5
            return action_name, 0.09
        if action == 9:
            email = self._find_asset("Employee Email")
            email["credential_risk"] = float(np.clip(email["credential_risk"] - 0.25, 0.0, 1.0))
            email["detection_level"] = float(np.clip(email["detection_level"] + 0.12, 0.0, 1.0))
            return action_name, 0.08
        if action == 10:
            target = max(self.assets, key=lambda a: float(a["credential_risk"]) + float(a["detection_level"]))
            target["infected"] = False
            target["compromised"] = False
            target["credential_risk"] = float(np.clip(target["credential_risk"] - 0.10, 0.0, 1.0))
            return action_name, 0.10
        if action == 11:
            self.segmented_finance = True
            finance = self._find_asset("Finance Database")
            finance["isolated"] = True
            return action_name, 0.11

        return action_name, 0.0

    def _find_asset(self, name: str) -> Dict[str, Any]:
        for asset in self.assets:
            if asset["name"] == name:
                return asset
        raise ValueError(f"asset not found: {name}")

    def _get_observation(self) -> np.ndarray:
        values: List[float] = []
        for asset in self.assets:
            values.extend(
                [
                    float(asset["patch_level"]),
                    1.0 if asset["infected"] else 0.0,
                    1.0 if asset["isolated"] else 0.0,
                    1.0 if asset["compromised"] else 0.0,
                    float(asset["criticality_score"]),
                    float(asset["credential_risk"]),
                    float(asset["detection_level"]),
                    float(asset["backup_health"]),
                    1.0 if asset["uptime_status"] else 0.0,
                ]
            )

        network_risk = compute_network_risk(
            self.assets,
            context={"segmented_finance": self.segmented_finance},
        )
        compromised_ratio = sum(1 for a in self.assets if a["compromised"]) / len(self.assets)
        downtime_ratio = sum(1 for a in self.assets if not a["uptime_status"]) / len(self.assets)
        values.extend(
            [
                network_risk,
                compromised_ratio,
                downtime_ratio,
                float(self.honeypot_timer / 5.0),
            ]
        )

        return np.asarray(values, dtype=np.float32)

    def _build_info(self, last_action: str, red_log: Dict[str, Any]) -> Dict[str, Any]:
        risk_context = {
            "segmented_finance": self.segmented_finance,
            "last_action": last_action,
            "red_success": bool(red_log.get("success", False)),
        }
        network_risk = compute_network_risk(self.assets, context=risk_context)
        risk_breakdown = compute_risk_breakdown(self.assets, context=risk_context)
        return {
            "step": self.current_step,
            "last_action": last_action,
            "network_risk": network_risk,
            "risk_breakdown": risk_breakdown,
            "assets": copy.deepcopy(self.assets),
            "red_log": red_log,
            "metrics": {
                "total_reward": self.episode_metrics.total_reward,
                "breach_count": self.episode_metrics.breaches,
                "downtime_count": self.episode_metrics.downtime_events,
                "successful_attacks": self.episode_metrics.successful_attacks,
                "prevented_attacks": self.episode_metrics.prevented_attacks,
            },
        }

    def render(self) -> None:
        info = self._build_info(last_action="render", red_log={})
        print(f"Step={info['step']} Risk={info['network_risk']:.3f}")


def maybe_register_openenv_env() -> None:
    """Optional OpenEnv registration if library is installed in runtime."""
    if openenv is None:
        return

    register = getattr(openenv, "register", None)
    if callable(register):
        register(
            id="CySentSecurity-v0",
            entry_point="backend.env.security_env:CySentSecurityEnv",
        )
