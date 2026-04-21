from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.intelligence.controller import IntelligenceController
from backend.intelligence.forecast import forecast_threats
from backend.intelligence.intelligence_log import build_incident_log
from backend.intelligence.llm_adapter import NoopLLMSummaryAdapter
from backend.intelligence.posture import summarize_posture
from backend.intelligence.reasoning import build_action_reasoning

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
        intelligence_enabled: bool = True,
        strategy_mode: str = "balanced",
        action_source: str = "ppo_ai",
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.segmented_finance = False
        self.honeypot_timer = 0
        self.training_mode = True
        self.previous_action_name = "reset"
        self.pending_effects: List[Dict[str, Any]] = []
        self.action_cooldowns: Dict[str, int] = {}
        self.defender_budget_max = 0.32
        self.defender_budget_remaining = 0.32
        self.credential_reset_friction_turns = 0
        self.recent_actions: List[str] = []
        self.current_alerts: List[Dict[str, Any]] = []
        self.attack_pressure = {"auth": 0.0, "finance": 0.0, "backup": 0.0}

        self.default_profile_selection = {
            "scenario": scenario,
            "difficulty": difficulty,
            "attacker": attacker,
        }
        self.current_profile_selection = dict(self.default_profile_selection)

        self.default_runtime_controls: Dict[str, Any] = {
            "strategy_mode": strategy_mode,
            "action_source": action_source,
            "intelligence_enabled": bool(intelligence_enabled),
        }
        self.runtime_controls: Dict[str, Any] = dict(self.default_runtime_controls)

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

        self.intelligence_controller = IntelligenceController(
            strategy_mode=str(self.runtime_controls["strategy_mode"]),
            action_source=str(self.runtime_controls["action_source"]),
        )
        self.llm_adapter = NoopLLMSummaryAdapter()

        self.assets: List[Dict[str, Any]] = []
        self.episode_metrics = EpisodeMetrics()
        self.replay: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.recent_red_logs: List[Dict[str, Any]] = []
        self.intelligence_replay: List[Dict[str, Any]] = []
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

    def _resolve_runtime_controls(self, options: Optional[Dict[str, Any]]) -> None:
        if not options:
            self.runtime_controls = dict(self.default_runtime_controls)
        else:
            self.runtime_controls = {
                "strategy_mode": str(options.get("strategy_mode", self.default_runtime_controls["strategy_mode"])),
                "action_source": str(options.get("action_source", self.default_runtime_controls["action_source"])),
                "intelligence_enabled": bool(
                    options.get("intelligence_enabled", self.default_runtime_controls["intelligence_enabled"])
                ),
            }

        self.intelligence_controller.configure(
            strategy_mode=str(self.runtime_controls["strategy_mode"]),
            action_source=str(self.runtime_controls["action_source"]),
        )

    def _sample_range(self, low: float, high: float) -> float:
        return float(self._rng.uniform(low, high))

    def _reset_assets(self) -> None:
        scenario = self.runtime_profiles.scenario
        detection_maturity = float(scenario.get("detection_maturity", 0.55))

        self.assets = []
        for name in ASSET_NAMES:
            profile = scenario["asset_profiles"][name]
            patch_level = self._sample_range(*profile["patch_level"])
            detection_level = np.clip(
                self._sample_range(*profile["detection_level"]) * (0.8 + 0.4 * detection_maturity),
                0.0,
                1.0,
            )
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
                    "downtime_cost": float(np.clip(profile.get("downtime_cost", dependency), 0.0, 1.0)),
                    "patch_speed": float(np.clip(profile.get("patch_speed", 0.5), 0.0, 1.0)),
                    "exposure": float(np.clip(profile.get("exposure", 0.5), 0.0, 1.0)),
                    "business_value": float(np.clip(profile.get("business_value", dependency), 0.0, 1.0)),
                    # Backward-compat aliases consumed by existing reward/risk logic.
                    "criticality_score": float(np.clip(criticality, 0.0, 1.0)),
                    "backup_health": float(np.clip(backup_status, 0.0, 1.0)),
                }
            )

    def _sync_asset_fields(self) -> None:
        for asset in self.assets:
            asset["backup_health"] = float(
                np.clip(asset.get("backup_status", asset.get("backup_health", 1.0)), 0.0, 1.0)
            )
            asset["criticality_score"] = float(
                np.clip(asset.get("criticality", asset.get("criticality_score", 0.5)), 0.0, 1.0)
            )
            asset["uptime"] = float(
                1.0 if asset.get("uptime_status", True) else max(0.0, float(asset.get("uptime", 0.0)))
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

        self.runtime_profiles = self._resolve_profiles(options)
        self._resolve_runtime_controls(options)

        self.threat_engine.configure(
            scenario_profile=self.runtime_profiles.scenario,
            difficulty_profile=self.runtime_profiles.difficulty,
            attacker_profile=self.runtime_profiles.attacker,
        )

        self.current_step = 0
        self.segmented_finance = False
        self.honeypot_timer = 0
        self.pending_effects = []
        self.action_cooldowns = {}
        budget_label = str(self.runtime_profiles.scenario.get("budget", "medium"))
        self.defender_budget_max = {"high": 0.42, "medium": 0.32, "constrained": 0.24}.get(budget_label, 0.32)
        self.defender_budget_remaining = self.defender_budget_max
        self.credential_reset_friction_turns = 0
        self.recent_actions = []
        self.current_alerts = []
        self.attack_pressure = {"auth": 0.0, "finance": 0.0, "backup": 0.0}
        self.episode_metrics = EpisodeMetrics()
        self.replay = []
        self.events = []
        self.recent_red_logs = []
        self.intelligence_replay = []
        self.last_narrative = ""
        self.previous_action_name = "reset"

        self._reset_assets()
        self._sync_asset_fields()

        obs = self._get_observation()
        info = self._build_info(
            last_action="reset",
            red_log={},
            events=[],
            narrative="Environment initialized.",
            intelligence={
                "enabled": bool(self.runtime_controls["intelligence_enabled"]),
                "strategy_mode": self.runtime_controls["strategy_mode"],
                "action_source": self.runtime_controls["action_source"],
            },
        )
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"invalid action: {action}"
        self.current_step += 1

        self.defender_budget_remaining = self.defender_budget_max
        for key in list(self.action_cooldowns.keys()):
            self.action_cooldowns[key] = max(0, int(self.action_cooldowns[key]) - 1)
        delayed_effect_events = self._apply_pending_effects()
        if self.credential_reset_friction_turns > 0:
            self.credential_reset_friction_turns -= 1

        assets_prev = copy.deepcopy(self.assets)
        action_name, action_cost, blue_action_notes = self._apply_blue_action(action)
        self.recent_actions.append(action_name)
        if len(self.recent_actions) > 8:
            self.recent_actions = self.recent_actions[-8:]

        if self.honeypot_timer > 0:
            self.honeypot_timer -= 1

        repeated_action = self.recent_actions[-1] if self.recent_actions else ""
        repeat_streak = 0
        for prev_action in reversed(self.recent_actions):
            if prev_action == repeated_action:
                repeat_streak += 1
            else:
                break

        attack_choice = self.threat_engine.choose_attack(
            self.assets,
            step=self.current_step,
            defender_context={
                "repeated_action": repeated_action,
                "repeat_streak": repeat_streak,
                "last_blue_action": action_name,
                "segmented_finance": bool(self.segmented_finance),
                "recent_blue_actions": list(self.recent_actions[-6:]),
            },
        )
        red_log = self.threat_engine.apply_attack(
            assets=self.assets,
            attack_choice=attack_choice,
            segmented_finance=self.segmented_finance,
            honeypot_active=self.honeypot_timer > 0,
            attacker_context={
                "auth_compromised_bonus": float(self.attack_pressure.get("auth", 0.0)),
                "finance_compromised_bonus": float(self.attack_pressure.get("finance", 0.0)),
                "backup_disruption_bonus": float(self.attack_pressure.get("backup", 0.0)),
                "last_blue_action": action_name,
            },
        )
        cascade_events = self._apply_cascading_effects(red_log)
        alert_events = self._generate_alerts(red_log)
        self.recent_red_logs.append(copy.deepcopy(red_log))
        if len(self.recent_red_logs) > 12:
            self.recent_red_logs = self.recent_red_logs[-12:]

        self._sync_asset_fields()

        risk_context = {
            "segmented_finance": self.segmented_finance,
            "last_action": action_name,
            "red_success": bool(red_log.get("success", False)),
        }
        network_risk_now = compute_network_risk(self.assets, context=risk_context)
        risk_breakdown_now = compute_risk_breakdown(self.assets, context=risk_context)

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
        turn_events.extend(delayed_effect_events)
        turn_events.extend(blue_action_notes)
        turn_events.extend(cascade_events)
        turn_events.extend(alert_events)
        if red_log.get("pivot"):
            turn_events.append(
                {
                    "turn": self.current_step,
                    "actor": "red",
                    "event_type": "attacker_pivot",
                    "target": str(red_log.get("target", "network")),
                    "success": True,
                    "details": {"pivot": red_log.get("pivot", "")},
                    "summary": str(red_log.get("pivot", "")),
                }
            )
        if "stealth_buildup" in red_log:
            turn_events.append(
                {
                    "turn": self.current_step,
                    "actor": "red",
                    "event_type": "latent_threat",
                    "target": "network",
                    "success": bool(red_log.get("success", False)),
                    "details": {
                        "stealth_meter": red_log.get("stealth_meter", 0.0),
                        "threat_pressure": red_log.get("threat_pressure", 0.0),
                        "stealth_buildup": red_log.get("stealth_buildup", 0.0),
                    },
                    "summary": "RED latent pressure updated through stealth operations.",
                }
            )

        intelligence_payload: Dict[str, Any] = {
            "enabled": bool(self.runtime_controls["intelligence_enabled"]),
            "strategy_mode": self.runtime_controls["strategy_mode"],
            "action_source": self.runtime_controls["action_source"],
        }
        if bool(self.runtime_controls["intelligence_enabled"]):
            threat_forecast = forecast_threats(
                assets=self.assets,
                attacker_profile=self.runtime_profiles.attacker,
                recent_red_logs=self.recent_red_logs,
                current_red_log=red_log,
            )
            posture = summarize_posture(
                assets=self.assets,
                network_risk=network_risk_now,
                risk_breakdown=risk_breakdown_now,
                scenario_name=str(self.runtime_profiles.scenario.get("name", "legacy")),
            )
            recommendation = self.intelligence_controller.recommend_action(
                assets=self.assets,
                risk_breakdown=risk_breakdown_now,
                forecast=threat_forecast,
            )
            reasoning = build_action_reasoning(
                action_name=action_name,
                assets_prev=assets_prev,
                assets_curr=self.assets,
                red_log=red_log,
                forecast=threat_forecast,
                strategy_mode=str(self.runtime_controls["strategy_mode"]),
                posture_level=str(posture.get("level", "guarded")),
            )
            comparison = self.intelligence_controller.compare_actions(
                executed_action_name=action_name,
                recommendation=recommendation,
            )

            intelligence_payload.update(
                {
                    "forecast": threat_forecast,
                    "posture": posture,
                    "reasoning": reasoning,
                    "recommendation": recommendation,
                    "action_comparison": comparison,
                }
            )
            intelligence_payload["llm_summary_candidate"] = self.llm_adapter.summarize_turn(intelligence_payload)
            incident_log = build_incident_log(self.current_step, intelligence_payload)
            intelligence_payload["incident_log"] = incident_log
            self.intelligence_replay.append(incident_log)
            turn_events.append(
                {
                    "turn": self.current_step,
                    "actor": "intelligence",
                    "event_type": "decision_intelligence",
                    "target": "network",
                    "success": True,
                    "details": {
                        "strategy_mode": self.runtime_controls["strategy_mode"],
                        "action_source": self.runtime_controls["action_source"],
                        "decision_confidence": reasoning.get("decision_confidence", 0.0),
                    },
                    "summary": incident_log["summary"],
                }
            )

        self.events.extend(turn_events)
        narrative = summarize_events(turn_events)
        self.last_narrative = narrative

        info = self._build_info(
            last_action=action_name,
            red_log=red_log,
            events=turn_events,
            narrative=narrative,
            intelligence=intelligence_payload,
            network_risk_override=network_risk_now,
            risk_breakdown_override=risk_breakdown_now,
        )
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
                "intelligence": intelligence_payload,
                "budget_remaining": self.defender_budget_remaining,
                "cooldowns": copy.deepcopy(self.action_cooldowns),
                "alerts": copy.deepcopy(self.current_alerts),
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

    def _apply_blue_action(self, action: int) -> Tuple[str, float, List[Dict[str, Any]]]:
        action_name = ACTION_NAMES[action]
        recovery_speed = float(self.runtime_profiles.scenario.get("recovery_speed", 0.55))
        notes: List[Dict[str, Any]] = []

        action_costs = {
            "do_nothing": 0.02,
            "patch_hr_systems": 0.10,
            "patch_web_server": 0.10,
            "patch_auth_server": 0.12,
            "rotate_credentials": 0.18,
            "isolate_suspicious_host": 0.14,
            "increase_monitoring": 0.12,
            "restore_backup": 0.20,
            "deploy_honeypot": 0.09,
            "phishing_training": 0.08,
            "investigate_top_alert": 0.10,
            "segment_finance_database": 0.11,
        }

        requested_cost = float(action_costs.get(action_name, 0.0))
        if requested_cost > self.defender_budget_remaining:
            notes.append(
                {
                    "turn": self.current_step,
                    "actor": "blue",
                    "event_type": "budget_blocked",
                    "target": "network",
                    "success": False,
                    "details": {
                        "requested_action": action_name,
                        "requested_cost": requested_cost,
                        "budget_remaining": self.defender_budget_remaining,
                    },
                    "summary": f"Blue could not execute {action_name}; per-turn budget was exhausted.",
                }
            )
            return "do_nothing", 0.02, notes

        major_actions = {"rotate_credentials", "segment_finance_database", "restore_backup", "investigate_top_alert"}
        if action_name in major_actions and self.action_cooldowns.get(action_name, 0) > 0:
            notes.append(
                {
                    "turn": self.current_step,
                    "actor": "blue",
                    "event_type": "cooldown_blocked",
                    "target": "network",
                    "success": False,
                    "details": {
                        "requested_action": action_name,
                        "cooldown_remaining": self.action_cooldowns.get(action_name, 0),
                    },
                    "summary": f"Blue attempted {action_name}, but the action is on cooldown.",
                }
            )
            return "do_nothing", 0.02, notes

        self.defender_budget_remaining = max(0.0, self.defender_budget_remaining - requested_cost)
        if action_name in major_actions:
            self.action_cooldowns[action_name] = 2

        def patch(name: str) -> None:
            asset = self._find_asset(name)
            patch_speed = float(asset.get("patch_speed", 0.5))
            asset["patch_level"] = float(np.clip(asset["patch_level"] + (0.10 + 0.16 * recovery_speed * patch_speed), 0.0, 1.0))
            asset["infected"] = False

        if action == 0:
            return action_name, requested_cost, notes
        if action == 1:
            delay = int(self._rng.integers(1, 3))
            self.pending_effects.append({"due_turn": self.current_step + delay, "effect": "patch", "target": "HR Systems"})
            notes.append(self._delayed_note(action_name, "HR Systems", delay))
            return action_name, requested_cost, notes
        if action == 2:
            delay = int(self._rng.integers(1, 3))
            self.pending_effects.append({"due_turn": self.current_step + delay, "effect": "patch", "target": "Web Server"})
            notes.append(self._delayed_note(action_name, "Web Server", delay))
            return action_name, requested_cost, notes
        if action == 3:
            delay = int(self._rng.integers(1, 3))
            self.pending_effects.append({"due_turn": self.current_step + delay, "effect": "patch", "target": "Auth Server"})
            notes.append(self._delayed_note(action_name, "Auth Server", delay))
            return action_name, requested_cost, notes
        if action == 4:
            for asset in self.assets:
                asset["credential_risk"] = float(np.clip(asset["credential_risk"] - 0.15, 0.0, 1.0))
            self.credential_reset_friction_turns = 1
            notes.append(
                {
                    "turn": self.current_step,
                    "actor": "blue",
                    "event_type": "defense_friction",
                    "target": "network",
                    "success": True,
                    "details": {"source_action": action_name, "duration_turns": 1},
                    "summary": "Credential rotation introduced temporary operational friction for one turn.",
                }
            )
            return action_name, requested_cost + 0.04, notes
        if action == 5:
            target = max(self.assets, key=lambda a: (float(a["infected"]), float(a["credential_risk"])))
            target["isolated"] = True
            target["uptime_status"] = False
            target["uptime"] = float(np.clip(target.get("uptime", 1.0) - 0.20, 0.0, 1.0))
            return action_name, requested_cost, notes
        if action == 6:
            for asset in self.assets:
                asset["detection_level"] = float(np.clip(asset["detection_level"] + 0.15, 0.0, 1.0))
            return action_name, requested_cost, notes
        if action == 7:
            restore_gain = 0.15 + 0.15 * recovery_speed
            for asset in self.assets:
                if not asset["uptime_status"]:
                    asset["uptime_status"] = True
                    asset["infected"] = False
                    asset["compromised"] = False
                asset["backup_status"] = float(np.clip(asset["backup_status"] + restore_gain, 0.0, 1.0))
                asset["uptime"] = float(np.clip(asset.get("uptime", 1.0) + restore_gain, 0.0, 1.0))
            self.attack_pressure["backup"] = max(0.0, float(self.attack_pressure.get("backup", 0.0)) - 0.08)
            return action_name, requested_cost, notes
        if action == 8:
            self.honeypot_timer = 5
            return action_name, requested_cost, notes
        if action == 9:
            email = self._find_asset("Employee Email")
            email["credential_risk"] = float(np.clip(email["credential_risk"] - 0.25, 0.0, 1.0))
            email["detection_level"] = float(np.clip(email["detection_level"] + 0.12, 0.0, 1.0))
            return action_name, requested_cost, notes
        if action == 10:
            if self.current_alerts:
                target_alert = sorted(self.current_alerts, key=lambda a: float(a.get("severity_score", 0.0)), reverse=True)[0]
                if bool(target_alert.get("true_positive", False)):
                    target = self._find_asset(str(target_alert.get("target", "Auth Server")))
                    target["infected"] = False
                    target["compromised"] = False
                    target["credential_risk"] = float(np.clip(target["credential_risk"] - 0.14, 0.0, 1.0))
                    notes.append(
                        {
                            "turn": self.current_step,
                            "actor": "blue",
                            "event_type": "alert_investigated",
                            "target": target["name"],
                            "success": True,
                            "details": {"severity": target_alert.get("severity", "high"), "true_positive": True},
                            "summary": f"Blue investigated a {target_alert.get('severity', 'high')} alert and contained compromise on {target['name']}.",
                        }
                    )
                else:
                    target = self._find_asset(str(target_alert.get("target", "SOC Monitoring Console")))
                    target["detection_level"] = float(np.clip(target["detection_level"] + 0.06, 0.0, 1.0))
                    notes.append(
                        {
                            "turn": self.current_step,
                            "actor": "blue",
                            "event_type": "alert_investigated",
                            "target": target["name"],
                            "success": False,
                            "details": {"severity": target_alert.get("severity", "low"), "true_positive": False},
                            "summary": "Blue investigated a false-positive alert; tuning improved but no threat was removed.",
                        }
                    )
                self.current_alerts = [a for a in self.current_alerts if a.get("id") != target_alert.get("id")]
            else:
                target = max(self.assets, key=lambda a: float(a["credential_risk"]) + float(a["detection_level"]))
                target["infected"] = False
                target["compromised"] = False
                target["credential_risk"] = float(np.clip(target["credential_risk"] - 0.10, 0.0, 1.0))
            return action_name, requested_cost, notes
        if action == 11:
            self.pending_effects.append({"due_turn": self.current_step + 1, "effect": "segment_finance", "target": "Finance Database"})
            notes.append(self._delayed_note(action_name, "Finance Database", 1))
            return action_name, requested_cost, notes

        return action_name, 0.0, notes

    def _delayed_note(self, action_name: str, target: str, delay: int) -> Dict[str, Any]:
        return {
            "turn": self.current_step,
            "actor": "blue",
            "event_type": "delayed_action_queued",
            "target": target,
            "success": True,
            "details": {"action": action_name, "delay_turns": delay},
            "summary": f"Blue queued {action_name} on {target}; effect will apply in {delay} turn(s).",
        }

    def _apply_pending_effects(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for effect in self.pending_effects:
            if int(effect.get("due_turn", 10**9)) > self.current_step:
                remaining.append(effect)
                continue

            effect_type = str(effect.get("effect", ""))
            target_name = str(effect.get("target", "network"))
            if effect_type == "patch":
                target = self._find_asset(target_name)
                patch_speed = float(target.get("patch_speed", 0.5))
                recovery_speed = float(self.runtime_profiles.scenario.get("recovery_speed", 0.55))
                target["patch_level"] = float(np.clip(target["patch_level"] + (0.10 + 0.16 * recovery_speed * patch_speed), 0.0, 1.0))
                target["infected"] = False
                events.append(
                    {
                        "turn": self.current_step,
                        "actor": "blue",
                        "event_type": "delayed_action_applied",
                        "target": target_name,
                        "success": True,
                        "details": {"effect": "patch"},
                        "summary": f"Queued patch completed on {target_name}.",
                    }
                )
            elif effect_type == "segment_finance":
                self.segmented_finance = True
                finance = self._find_asset("Finance Database")
                finance["isolated"] = True
                events.append(
                    {
                        "turn": self.current_step,
                        "actor": "blue",
                        "event_type": "delayed_action_applied",
                        "target": "Finance Database",
                        "success": True,
                        "details": {"effect": "segment_finance"},
                        "summary": "Finance segmentation is now active.",
                    }
                )

        self.pending_effects = remaining
        return events

    def _apply_cascading_effects(self, red_log: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        target_name = str(red_log.get("target", ""))
        success = bool(red_log.get("success", False))
        if not success:
            self.attack_pressure["auth"] = max(0.0, self.attack_pressure["auth"] - 0.02)
            self.attack_pressure["finance"] = max(0.0, self.attack_pressure["finance"] - 0.02)
            self.attack_pressure["backup"] = max(0.0, self.attack_pressure["backup"] - 0.02)
            return events

        if target_name == "Auth Server":
            for asset in self.assets:
                asset["credential_risk"] = float(np.clip(asset["credential_risk"] + 0.05, 0.0, 1.0))
            self.attack_pressure["auth"] = min(0.24, self.attack_pressure["auth"] + 0.07)
            events.append(
                {
                    "turn": self.current_step,
                    "actor": "system",
                    "event_type": "cascade_effect",
                    "target": "network",
                    "success": True,
                    "details": {"source": "Auth Server", "effect": "credential_theft_and_lateral_pressure"},
                    "summary": "Auth compromise increased credential theft and lateral movement pressure across assets.",
                }
            )

        if target_name == "Finance Database":
            self.attack_pressure["finance"] = min(0.24, self.attack_pressure["finance"] + 0.08)
            events.append(
                {
                    "turn": self.current_step,
                    "actor": "system",
                    "event_type": "cascade_effect",
                    "target": "Finance Database",
                    "success": True,
                    "details": {"source": "Finance Database", "effect": "exfiltration_pressure"},
                    "summary": "Finance compromise increased probability of successful exfiltration attempts.",
                }
            )

        if target_name == "Backup Infrastructure":
            for asset in self.assets:
                asset["backup_status"] = float(np.clip(asset.get("backup_status", 1.0) - 0.08, 0.0, 1.0))
            self.attack_pressure["backup"] = min(0.24, self.attack_pressure["backup"] + 0.08)
            events.append(
                {
                    "turn": self.current_step,
                    "actor": "system",
                    "event_type": "cascade_effect",
                    "target": "network",
                    "success": True,
                    "details": {"source": "Backup Infrastructure", "effect": "weakened_recovery"},
                    "summary": "Backup disruption weakened organization-wide recovery resilience.",
                }
            )

        return events

    def _generate_alerts(self, red_log: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        self.current_alerts = []

        detection_maturity = float(self.runtime_profiles.scenario.get("detection_maturity", 0.55))
        success = bool(red_log.get("success", False))
        attack = str(red_log.get("attack", ""))
        target = str(red_log.get("target", "network"))

        if attack != "no_attack":
            base_score = 0.45 + (0.30 if success else 0.12)
            if attack in {"ransomware_attempt", "lateral_movement", "data_exfiltration", "privilege_escalation"}:
                base_score += 0.20
            base_score += 0.10 * (1.0 - detection_maturity)
            severity_score = float(np.clip(base_score, 0.0, 1.0))
            severity = (
                "critical"
                if severity_score >= 0.85
                else "high"
                if severity_score >= 0.65
                else "medium"
                if severity_score >= 0.45
                else "low"
            )
            alert = {
                "id": f"a{self.current_step}_0",
                "target": target,
                "severity": severity,
                "severity_score": severity_score,
                "true_positive": True,
                "attack": attack,
            }
            self.current_alerts.append(alert)
            events.append(
                {
                    "turn": self.current_step,
                    "actor": "sensor",
                    "event_type": "alert_generated",
                    "target": target,
                    "success": True,
                    "details": alert,
                    "summary": f"{severity.capitalize()} alert generated for {target} ({attack}).",
                }
            )

        false_positive_chance = float(np.clip(0.25 - 0.15 * detection_maturity, 0.04, 0.30))
        if self._rng.random() < false_positive_chance:
            fp_target = self.assets[int(self._rng.integers(0, len(self.assets)))]["name"]
            fp_score = float(np.clip(0.30 + self._rng.uniform(0.0, 0.35), 0.0, 1.0))
            fp_severity = "medium" if fp_score > 0.5 else "low"
            false_alert = {
                "id": f"a{self.current_step}_fp",
                "target": fp_target,
                "severity": fp_severity,
                "severity_score": fp_score,
                "true_positive": False,
                "attack": "noise",
            }
            self.current_alerts.append(false_alert)
            events.append(
                {
                    "turn": self.current_step,
                    "actor": "sensor",
                    "event_type": "alert_noise",
                    "target": fp_target,
                    "success": True,
                    "details": false_alert,
                    "summary": f"{fp_severity.capitalize()} false-positive alert surfaced on {fp_target}.",
                }
            )

        return events

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
        intelligence: Optional[Dict[str, Any]] = None,
        network_risk_override: Optional[float] = None,
        risk_breakdown_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if network_risk_override is None or risk_breakdown_override is None:
            risk_context = {
                "segmented_finance": self.segmented_finance,
                "last_action": last_action,
                "red_success": bool(red_log.get("success", False)),
            }
            network_risk = compute_network_risk(self.assets, context=risk_context)
            risk_breakdown = compute_risk_breakdown(self.assets, context=risk_context)
        else:
            network_risk = float(network_risk_override)
            risk_breakdown = dict(risk_breakdown_override)

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
                "budget": self.runtime_profiles.scenario.get("budget", "medium"),
                "strategy_mode": self.runtime_controls["strategy_mode"],
                "action_source": self.runtime_controls["action_source"],
                "intelligence_enabled": bool(self.runtime_controls["intelligence_enabled"]),
            },
            "defender": {
                "budget_max": float(self.defender_budget_max),
                "budget_remaining": float(self.defender_budget_remaining),
                "cooldowns": copy.deepcopy(self.action_cooldowns),
                "pending_effects": len(self.pending_effects),
            },
            "alerts": copy.deepcopy(self.current_alerts),
            "intelligence": intelligence
            or {
                "enabled": bool(self.runtime_controls["intelligence_enabled"]),
                "strategy_mode": self.runtime_controls["strategy_mode"],
                "action_source": self.runtime_controls["action_source"],
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
        info = self._build_info(
            last_action="render",
            red_log={},
            events=[],
            narrative=self.last_narrative,
            intelligence={
                "enabled": bool(self.runtime_controls["intelligence_enabled"]),
                "strategy_mode": self.runtime_controls["strategy_mode"],
                "action_source": self.runtime_controls["action_source"],
            },
        )
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
