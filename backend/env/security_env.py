from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config_loader import CANONICAL_ASSET_NAMES, RuntimeProfiles, load_runtime_profiles
from .events import build_turn_events, summarize_events
from .reward import compute_reward
from .risk import compute_network_risk, compute_risk_breakdown
from .threat_engine import ThreatEngine

try:
    import openenv  # type: ignore
except Exception:  # pragma: no cover - optional runtime integration
    openenv = None


ASSET_NAMES = list(CANONICAL_ASSET_NAMES)

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

    def __init__(
        self,
        max_steps: int = 150,
        seed: Optional[int] = None,
        scenario: str = "legacy",
        difficulty: str = "medium",
        attacker: str = "legacy_default",
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.segmented_finance = False
        self.honeypot_timer = 0
        self.training_mode = True
        self.previous_action_name = "reset"

        self.default_profile_selection = {
            "scenario": scenario,
            "difficulty": difficulty,
            "attacker": attacker,
        }
        self.current_profile_selection = dict(self.default_profile_selection)

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
        self.runtime_profiles: RuntimeProfiles = load_runtime_profiles(
            scenario=scenario,
            difficulty=difficulty,
            attacker=attacker,
        )

        self.threat_engine = ThreatEngine(
            seed=seed,
            scenario_profile=self.runtime_profiles.scenario,
            difficulty_profile=self.runtime_profiles.difficulty,
            attacker_profile=self.runtime_profiles.attacker,
        )

        self.assets: List[Dict[str, Any]] = []
        self.episode_metrics = EpisodeMetrics()
        self.replay: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.last_narrative = ""
        self._reset_assets()

    def _resolve_profiles(self, options: Optional[Dict[str, Any]]) -> RuntimeProfiles:
        if not options:
            self.current_profile_selection = dict(self.default_profile_selection)
            return load_runtime_profiles(**self.current_profile_selection)

        scenario = str(options.get("scenario", self.default_profile_selection["scenario"]))
        difficulty = str(options.get("difficulty", self.default_profile_selection["difficulty"]))
        attacker = str(options.get("attacker", self.default_profile_selection["attacker"]))

        self.current_profile_selection = {
            "scenario": scenario,
            "difficulty": difficulty,
            "attacker": attacker,
        }
        return load_runtime_profiles(scenario=scenario, difficulty=difficulty, attacker=attacker)

    def _sample_range(self, low: float, high: float) -> float:
        return float(self._rng.uniform(low, high))

    def _reset_assets(self) -> None:
        scenario = self.runtime_profiles.scenario
        detection_maturity = float(scenario.get("detection_maturity", 0.55))

        self.assets = []
        for name in ASSET_NAMES:
            profile = scenario["asset_profiles"][name]
            patch_level = self._sample_range(*profile["patch_level"])
            detection_level = np.clip(self._sample_range(*profile["detection_level"]) * (0.8 + 0.4 * detection_maturity), 0.0, 1.0)
            credential_risk = self._sample_range(*profile["credential_risk"])
            backup_status = self._sample_range(*profile["backup_status"])
            criticality = float(profile["criticality"])
            dependency = float(profile["business_dependency"])

            self.assets.append(
                {
                    "name": name,
                    "patch_level": float(np.clip(patch_level, 0.0, 1.0)),
                    "infected": False,
                    "isolated": False,
                    "compromised": False,
                    "detection_level": float(np.clip(detection_level, 0.0, 1.0)),
                    "credential_risk": float(np.clip(credential_risk, 0.0, 1.0)),
                    "uptime_status": True,
                    "uptime": 1.0,
                    "backup_status": float(np.clip(backup_status, 0.0, 1.0)),
                    "criticality": float(np.clip(criticality, 0.0, 1.0)),
                    "business_dependency": float(np.clip(dependency, 0.0, 1.0)),
                    # Backward-compat aliases consumed by existing reward/risk logic.
                    "criticality_score": float(np.clip(criticality, 0.0, 1.0)),
                    "backup_health": float(np.clip(backup_status, 0.0, 1.0)),
                }
            )

    def _sync_asset_fields(self) -> None:
        for asset in self.assets:
            asset["backup_health"] = float(np.clip(asset.get("backup_status", asset.get("backup_health", 1.0)), 0.0, 1.0))
            asset["criticality_score"] = float(np.clip(asset.get("criticality", asset.get("criticality_score", 0.5)), 0.0, 1.0))
            asset["uptime"] = float(1.0 if asset.get("uptime_status", True) else max(0.0, float(asset.get("uptime", 0.0))))

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

        self.runtime_profiles = self._resolve_profiles(options)
        self.threat_engine.configure(
            scenario_profile=self.runtime_profiles.scenario,
            difficulty_profile=self.runtime_profiles.difficulty,
            attacker_profile=self.runtime_profiles.attacker,
        )

        self.current_step = 0
        self.segmented_finance = False
        self.honeypot_timer = 0
        self.episode_metrics = EpisodeMetrics()
        self.replay = []
        self.events = []
        self.last_narrative = ""
        self.previous_action_name = "reset"

        self._reset_assets()
        self._sync_asset_fields()

        obs = self._get_observation()
        info = self._build_info(last_action="reset", red_log={}, events=[], narrative="Environment initialized.")
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"invalid action: {action}"
        self.current_step += 1

        assets_prev = copy.deepcopy(self.assets)
        action_name, action_cost = self._apply_blue_action(action)

        if self.honeypot_timer > 0:
            self.honeypot_timer -= 1

        attack_choice = self.threat_engine.choose_attack(self.assets, step=self.current_step)
        red_log = self.threat_engine.apply_attack(
            assets=self.assets,
            attack_choice=attack_choice,
            segmented_finance=self.segmented_finance,
            honeypot_active=self.honeypot_timer > 0,
        )

        self._sync_asset_fields()

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

        chain_context = red_log.get("chain", {}) if isinstance(red_log.get("chain", {}), dict) else {}
        turn_events = build_turn_events(
            turn=self.current_step,
            action_name=action_name,
            red_log=red_log,
            chain_context=chain_context,
        )
        self.events.extend(turn_events)
        narrative = summarize_events(turn_events)
        self.last_narrative = narrative

        info = self._build_info(last_action=action_name, red_log=red_log, events=turn_events, narrative=narrative)
        info["termination_reason"] = termination_reason
        self.previous_action_name = action_name
        self.replay.append(
            {
                "step": self.current_step,
                "action": action_name,
                "red": red_log,
                "reward": reward,
                "network_risk": info["network_risk"],
                "events": turn_events,
                "narrative": narrative,
            }
        )

        return self._get_observation(), float(reward), terminated, truncated, info

    def _is_terminal(self) -> Tuple[bool, str]:
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
        recovery_speed = float(self.runtime_profiles.scenario.get("recovery_speed", 0.55))

        def patch(name: str) -> None:
            asset = self._find_asset(name)
            asset["patch_level"] = float(np.clip(asset["patch_level"] + (0.20 + 0.12 * recovery_speed), 0.0, 1.0))
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
            target["uptime"] = float(np.clip(target.get("uptime", 1.0) - 0.20, 0.0, 1.0))
            return action_name, 0.14
        if action == 6:
            for asset in self.assets:
                asset["detection_level"] = float(np.clip(asset["detection_level"] + 0.15, 0.0, 1.0))
            return action_name, 0.12
        if action == 7:
            restore_gain = 0.15 + 0.15 * recovery_speed
            for asset in self.assets:
                if not asset["uptime_status"]:
                    asset["uptime_status"] = True
                    asset["infected"] = False
                    asset["compromised"] = False
                asset["backup_status"] = float(np.clip(asset["backup_status"] + restore_gain, 0.0, 1.0))
                asset["uptime"] = float(np.clip(asset.get("uptime", 1.0) + restore_gain, 0.0, 1.0))
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

    def _build_info(
        self,
        *,
        last_action: str,
        red_log: Dict[str, Any],
        events: List[Dict[str, Any]],
        narrative: str,
    ) -> Dict[str, Any]:
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
            "events": events,
            "narrative": narrative,
            "profile": {
                "scenario": self.runtime_profiles.scenario.get("name", "legacy"),
                "difficulty": self.runtime_profiles.difficulty.get("name", "medium"),
                "attacker": self.runtime_profiles.attacker.get("name", "legacy_default"),
                "detection_maturity": self.runtime_profiles.scenario.get("detection_maturity", 0.55),
                "recovery_speed": self.runtime_profiles.scenario.get("recovery_speed", 0.55),
                "critical_assets": self.runtime_profiles.scenario.get("critical_assets", []),
                "attack_priorities": self.runtime_profiles.scenario.get("attack_priorities", []),
            },
            "metrics": {
                "total_reward": self.episode_metrics.total_reward,
                "breach_count": self.episode_metrics.breaches,
                "downtime_count": self.episode_metrics.downtime_events,
                "successful_attacks": self.episode_metrics.successful_attacks,
                "prevented_attacks": self.episode_metrics.prevented_attacks,
            },
        }

    def render(self) -> None:
        info = self._build_info(last_action="render", red_log={}, events=[], narrative=self.last_narrative)
        profile = info["profile"]
        print(
            f"Step={info['step']} Risk={info['network_risk']:.3f} "
            f"Scenario={profile['scenario']} Difficulty={profile['difficulty']} Attacker={profile['attacker']}"
        )


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
