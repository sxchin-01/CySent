from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .risk import clamp, compute_asset_risk

ATTACK_TYPES: List[str] = [
    "phishing_email",
    "password_spray",
    "malware_dropper",
    "credential_theft",
    "privilege_escalation",
    "lateral_movement",
    "ransomware_attempt",
    "data_exfiltration",
    "insider_misuse",
    "recon_scan",
]


class ThreatEngine:
    """RED threat engine with profile-driven behavior and multi-turn attack campaigns."""

    def __init__(
        self,
        seed: int | None = None,
        scenario_profile: Optional[Dict[str, Any]] = None,
        difficulty_profile: Optional[Dict[str, Any]] = None,
        attacker_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._scenario_profile = scenario_profile or {}
        self._difficulty_profile = difficulty_profile or {}
        self._attacker_profile = attacker_profile or {}

        self._campaign_counter = 0
        self._active_chain: Optional[Dict[str, Any]] = None
        self._attacker_goal = self._resolve_goal(self._attacker_profile)
        self._blue_action_memory: List[str] = []
        self._stealth_meter = 0.0
        self._threat_pressure = 0.0
        self._goal_progress = 0.0

    def seed(self, seed: int | None) -> None:
        self._rng.seed(seed)

    def configure(
        self,
        *,
        scenario_profile: Dict[str, Any],
        difficulty_profile: Dict[str, Any],
        attacker_profile: Dict[str, Any],
    ) -> None:
        self._scenario_profile = scenario_profile
        self._difficulty_profile = difficulty_profile
        self._attacker_profile = attacker_profile
        self._attacker_goal = self._resolve_goal(self._attacker_profile)
        self.reset_episode_state()

    def reset_episode_state(self) -> None:
        self._active_chain = None
        self._blue_action_memory = []
        self._stealth_meter = 0.0
        self._threat_pressure = 0.0
        self._goal_progress = 0.0

    def choose_attack(
        self,
        assets: List[Dict[str, Any]],
        step: int,
        defender_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ingest_defender_context(defender_context)

        attack_frequency = float(self._difficulty_profile.get("attack_frequency", 0.84))
        attack_frequency = clamp(attack_frequency + 0.15 * self._threat_pressure, 0.05, 1.0)
        if self._rng.random() > clamp(attack_frequency, 0.05, 1.0):
            return {
                "scheduled": False,
                "target_idx": None,
                "attack_type": "no_attack",
                "chain": {},
                "notes": "Threat actor paused active operations this turn.",
            }

        if self._rng.random() < float(self._difficulty_profile.get("insider_event_chance", 0.04)):
            target_idx = self._select_target_idx(assets, hint_attack="insider_misuse")
            return {
                "scheduled": True,
                "target_idx": target_idx,
                "attack_type": "insider_misuse",
                "chain": {},
                "notes": "Opportunistic insider activity surfaced this turn.",
            }

        chain_attack = self._select_chain_attack(assets)
        if chain_attack is not None:
            return chain_attack

        target_idx = self._select_target_idx(assets, hint_attack=None)
        attack_type = self._sample_attack_type(assets[target_idx], defender_context=defender_context)
        return {
            "scheduled": True,
            "target_idx": target_idx,
            "attack_type": attack_type,
            "chain": {},
            "notes": "Standalone tactic selected by attacker profile.",
        }

    def apply_attack(
        self,
        *,
        assets: List[Dict[str, Any]],
        attack_choice: Dict[str, Any],
        segmented_finance: bool,
        honeypot_active: bool,
        attacker_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not attack_choice.get("scheduled", False):
            return {
                "target": "network",
                "attack": "no_attack",
                "success": False,
                "escalation": False,
                "stealth": False,
                "zero_day": False,
                "notes": str(attack_choice.get("notes", "No attack executed.")),
                "chain": {},
            }

        target_idx = attack_choice.get("target_idx")
        attack_type = str(attack_choice.get("attack_type", "no_attack"))
        if target_idx is None or attack_type == "no_attack":
            return {
                "target": "network",
                "attack": "no_attack",
                "success": False,
                "escalation": False,
                "stealth": False,
                "zero_day": False,
                "notes": "Attack planning failed to find a viable target.",
                "chain": {},
            }

        asset = assets[int(target_idx)]
        log: Dict[str, Any] = {
            "target": asset["name"],
            "attack": attack_type,
            "success": False,
            "escalation": False,
            "stealth": False,
            "zero_day": False,
            "notes": str(attack_choice.get("notes", "")),
            "chain": dict(attack_choice.get("chain", {})),
            "attacker_goal": self._attacker_goal,
        }

        if asset["isolated"]:
            log["notes"] = "Target was isolated; attack impact was reduced."

        success_probability = self._success_probability(asset=asset, attack_type=attack_type)

        ctx = attacker_context or {}
        success_probability += self._goal_attack_bonus(attack_type)
        success_probability += float(ctx.get("auth_compromised_bonus", 0.0)) * (
            1.0 if attack_type in {"credential_theft", "lateral_movement", "privilege_escalation"} else 0.2
        )
        success_probability += float(ctx.get("finance_compromised_bonus", 0.0)) * (
            1.0 if attack_type == "data_exfiltration" else 0.25
        )
        success_probability += float(ctx.get("backup_disruption_bonus", 0.0)) * (
            1.0 if attack_type in {"ransomware_attempt", "lateral_movement"} else 0.2
        )

        success_probability *= float(self._difficulty_profile.get("exploit_success_multiplier", 1.0))

        zero_day = self._rng.random() < float(self._difficulty_profile.get("zero_day_chance", 0.06))
        if zero_day:
            success_probability += 0.14
            log["zero_day"] = True

        if asset["isolated"]:
            success_probability *= 0.45
        if honeypot_active and attack_type in {"recon_scan", "lateral_movement", "credential_theft"}:
            success_probability *= 0.60
            log["notes"] = "Honeypot diverted part of attack pressure."

        success = self._rng.random() < clamp(success_probability, 0.05, 0.98)
        log["success"] = bool(success)

        stealth_probability = float(self._difficulty_profile.get("stealth_probability", 0.35))
        stealth_probability *= float(self._attacker_profile.get("stealth_modifier", 1.0))
        stealth_probability = clamp(stealth_probability, 0.02, 0.98)
        log["stealth"] = bool(self._rng.random() < stealth_probability)

        if success:
            self._mutate_state(asset=asset, attack_type=attack_type, stealth=bool(log["stealth"]))
            if attack_type in {"credential_theft", "privilege_escalation", "lateral_movement", "insider_misuse"}:
                log["escalation"] = self._maybe_escalate(
                    assets=assets,
                    source_idx=int(target_idx),
                    segmented_finance=segmented_finance,
                )

        chain_outcome = self._update_chain_state(
            success=success,
            target_idx=int(target_idx),
            attack_type=attack_type,
            incoming_chain=log.get("chain", {}),
        )
        log["chain"] = chain_outcome

        stealth_before = self._stealth_meter
        self._update_hidden_state(
            attack_type=attack_type,
            success=success,
            stealth=bool(log["stealth"]),
            defender_action=str((attacker_context or {}).get("last_blue_action", "")),
        )
        log["stealth_meter"] = float(clamp(self._stealth_meter, 0.0, 1.0))
        log["threat_pressure"] = float(clamp(self._threat_pressure, 0.0, 1.0))
        log["stealth_buildup"] = float(self._stealth_meter - stealth_before)
        log["goal_progress"] = float(clamp(self._goal_progress, 0.0, 1.0))
        pivot_reason = self._get_pivot_reason()
        if pivot_reason:
            log["pivot"] = pivot_reason

        self._sync_asset_legacy_fields(asset)
        return log

    def _select_chain_attack(self, assets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        escalation_speed = float(self._difficulty_profile.get("escalation_speed", 1.0))

        if self._active_chain is not None:
            continue_probability = clamp(0.55 * escalation_speed, 0.25, 0.95)
            if self._rng.random() < continue_probability:
                stage_idx = int(self._active_chain["stage_index"])
                attack_type = self._active_chain["stages"][stage_idx]
                target_idx = int(self._active_chain["target_idx"])
                return {
                    "scheduled": True,
                    "target_idx": target_idx,
                    "attack_type": attack_type,
                    "chain": {
                        "chain_id": self._active_chain["chain_id"],
                        "stage": attack_type,
                        "stage_index": stage_idx,
                        "stage_total": len(self._active_chain["stages"]),
                    },
                    "notes": "Continuing active multi-stage campaign.",
                }

        chain_bias = float(self._attacker_profile.get("chain_bias", 0.8))
        start_probability = clamp(chain_bias * 0.65, 0.10, 0.95)
        if self._rng.random() > start_probability:
            return None

        templates = self._attacker_profile.get("chain_templates", [])
        if not isinstance(templates, list) or not templates:
            templates = self._default_goal_templates()

        stages = self._rng.choice(templates)
        if not isinstance(stages, list) or not stages:
            return None

        attack_type = str(stages[0])
        target_idx = self._select_target_idx(assets, hint_attack=attack_type)

        self._campaign_counter += 1
        self._active_chain = {
            "chain_id": f"camp_{self._campaign_counter}",
            "stages": [str(s) for s in stages],
            "stage_index": 0,
            "target_idx": target_idx,
        }

        return {
            "scheduled": True,
            "target_idx": target_idx,
            "attack_type": attack_type,
            "chain": {
                "chain_id": self._active_chain["chain_id"],
                "stage": attack_type,
                "stage_index": 0,
                "stage_total": len(self._active_chain["stages"]),
            },
            "notes": "Started a structured multi-stage campaign.",
        }

    def _sample_attack_type(
        self,
        asset: Dict[str, Any],
        defender_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_probs = {
            "phishing_email": 0.10 + (1.0 - asset["detection_level"]) * 0.08,
            "password_spray": 0.10 + asset["credential_risk"] * 0.12,
            "malware_dropper": 0.10 + (1.0 - asset["patch_level"]) * 0.12,
            "credential_theft": 0.10 + asset["credential_risk"] * 0.12,
            "privilege_escalation": 0.08 + (1.0 - asset["patch_level"]) * 0.10,
            "lateral_movement": 0.08 + (0.25 if asset["compromised"] else 0.0),
            "ransomware_attempt": 0.08 + (1.0 - asset.get("backup_status", asset["backup_health"])) * 0.10,
            "data_exfiltration": 0.08 + (0.30 if asset["compromised"] else 0.0),
            "insider_misuse": 0.09,
            "recon_scan": 0.09 + (1.0 - asset["detection_level"]) * 0.06,
        }

        bias = self._attacker_profile.get("attack_bias", {})
        if isinstance(bias, dict):
            for attack_name, weight in bias.items():
                if attack_name in base_probs:
                    base_probs[attack_name] = max(0.0, base_probs[attack_name] + float(weight))

        adaptation = defender_context or {}
        repeated_action = str(adaptation.get("repeated_action", ""))
        streak = int(adaptation.get("repeat_streak", 0))
        finance_segmented = bool(adaptation.get("segmented_finance", False))
        investigate_bias = sum(1 for action in self._blue_action_memory[-4:] if action == "investigate_top_alert")
        if streak >= 2:
            multiplier = min(0.10 * streak, 0.35)
            if repeated_action in {"rotate_credentials", "phishing_training"}:
                base_probs["malware_dropper"] += multiplier
                base_probs["ransomware_attempt"] += multiplier
            elif repeated_action in {"patch_auth_server", "patch_web_server", "patch_hr_systems"}:
                base_probs["credential_theft"] += multiplier
                base_probs["password_spray"] += multiplier
            elif repeated_action == "segment_finance_database":
                base_probs["insider_misuse"] += multiplier
                base_probs["phishing_email"] += multiplier * 0.8
                base_probs["credential_theft"] += multiplier
            elif repeated_action == "investigate_top_alert":
                base_probs["recon_scan"] -= multiplier * 0.6
                base_probs["privilege_escalation"] += multiplier
                base_probs["data_exfiltration"] += multiplier * 0.8

        if finance_segmented:
            base_probs["insider_misuse"] += 0.12
            base_probs["credential_theft"] += 0.10
            base_probs["phishing_email"] += 0.08
            base_probs["data_exfiltration"] -= 0.08

        if investigate_bias >= 2:
            base_probs["recon_scan"] *= 0.70
            base_probs["phishing_email"] *= 0.90
            base_probs["privilege_escalation"] += 0.10
            base_probs["lateral_movement"] += 0.10

        if self._attacker_goal == "stealth_persistence" and self._stealth_meter > 0.35:
            base_probs["data_exfiltration"] += 0.12
            base_probs["privilege_escalation"] += 0.08
            base_probs["ransomware_attempt"] -= 0.10

        return self._sample_categorical(base_probs)

    def _select_target_idx(self, assets: List[Dict[str, Any]], hint_attack: Optional[str]) -> int:
        priorities = set(self._scenario_profile.get("attack_priorities", []))
        critical_assets = set(self._scenario_profile.get("critical_assets", []))
        criticality_bias = float(self._attacker_profile.get("target_criticality_bias", 1.0))
        backup_bias = float(self._attacker_profile.get("target_backup_bias", 1.0))

        weights: List[float] = []
        for asset in assets:
            w = compute_asset_risk(asset)
            w += 0.22 if asset["name"] in priorities else 0.0
            w += 0.30 * float(asset.get("criticality", asset.get("criticality_score", 0.5))) * criticality_bias
            w += 0.10 * float(asset.get("exposure", 0.5))
            w += 0.10 * float(asset.get("business_value", asset.get("business_dependency", 0.5)))
            if self._attacker_goal == "downtime":
                w += 0.14 * float(asset.get("downtime_cost", 0.5))
            elif self._attacker_goal == "stealth_persistence":
                w += 0.08 * (1.0 - float(asset.get("detection_level", 0.5)))
            elif self._attacker_goal == "takeover":
                w += 0.10 * float(asset.get("credential_risk", 0.5))
            elif self._attacker_goal == "sabotage":
                w += 0.08 * float(asset.get("downtime_cost", 0.5))

            backup_status = float(asset.get("backup_status", asset.get("backup_health", 1.0)))
            if hint_attack == "ransomware_attempt":
                w += (1.0 - backup_status) * 0.25 * backup_bias

            if asset["name"] in critical_assets:
                w += 0.10
            if asset["isolated"]:
                w *= 0.55

            weights.append(max(0.05, float(w)))

        return self._weighted_choice(weights)

    def _success_probability(self, asset: Dict[str, Any], attack_type: str) -> float:
        patch = clamp(float(asset["patch_level"]), 0.0, 1.0)
        detect = clamp(float(asset["detection_level"]), 0.0, 1.0)
        cred = clamp(float(asset["credential_risk"]), 0.0, 1.0)
        backup = clamp(float(asset.get("backup_status", asset.get("backup_health", 1.0))), 0.0, 1.0)

        mapping = {
            "phishing_email": 0.28 + cred * 0.25 + (1.0 - detect) * 0.20,
            "password_spray": 0.20 + cred * 0.35,
            "malware_dropper": 0.20 + (1.0 - patch) * 0.35,
            "credential_theft": 0.18 + cred * 0.40,
            "privilege_escalation": 0.15 + (1.0 - patch) * 0.40,
            "lateral_movement": 0.15 + (0.35 if asset["infected"] else 0.0),
            "ransomware_attempt": 0.16 + (1.0 - backup) * 0.35,
            "data_exfiltration": 0.12 + (0.40 if asset["compromised"] else 0.0),
            "insider_misuse": 0.10 + cred * 0.25,
            "recon_scan": 0.22 + (1.0 - detect) * 0.20,
            "no_attack": 0.0,
        }
        return clamp(mapping.get(attack_type, 0.2), 0.05, 0.95)

    def _mutate_state(self, asset: Dict[str, Any], attack_type: str, stealth: bool) -> None:
        if attack_type in {"phishing_email", "password_spray", "credential_theft", "insider_misuse"}:
            asset["credential_risk"] = clamp(asset["credential_risk"] + 0.16, 0.0, 1.0)

        if attack_type in {"malware_dropper", "lateral_movement", "ransomware_attempt"}:
            asset["infected"] = True
            asset["patch_level"] = clamp(asset["patch_level"] - 0.06, 0.0, 1.0)

        if attack_type in {"privilege_escalation", "lateral_movement", "data_exfiltration", "ransomware_attempt"}:
            asset["compromised"] = True
            if attack_type in {"ransomware_attempt", "data_exfiltration"}:
                asset["uptime_status"] = False
                asset["uptime"] = clamp(float(asset.get("uptime", 1.0)) - 0.25, 0.0, 1.0)

        if attack_type == "recon_scan":
            asset["detection_level"] = clamp(asset["detection_level"] - 0.08, 0.0, 1.0)

        if attack_type == "ransomware_attempt":
            asset["backup_status"] = clamp(asset.get("backup_status", 1.0) - 0.18, 0.0, 1.0)

        if stealth:
            asset["detection_level"] = clamp(asset["detection_level"] - 0.04, 0.0, 1.0)
        else:
            asset["detection_level"] = clamp(asset["detection_level"] + 0.02, 0.0, 1.0)

    def _maybe_escalate(self, assets: List[Dict[str, Any]], source_idx: int, segmented_finance: bool) -> bool:
        candidate_idxs = [i for i in range(len(assets)) if i != source_idx and not assets[i]["isolated"]]
        if not candidate_idxs:
            return False

        if segmented_finance:
            candidate_idxs = [i for i in candidate_idxs if assets[i]["name"] != "Finance Database"]
            if not candidate_idxs:
                return False

        escalation_speed = float(self._difficulty_profile.get("escalation_speed", 1.0))
        escalation_probability = clamp(0.45 * escalation_speed, 0.12, 0.95)
        if self._rng.random() < escalation_probability:
            idx = self._rng.choice(candidate_idxs)
            assets[idx]["infected"] = True
            assets[idx]["credential_risk"] = clamp(assets[idx]["credential_risk"] + 0.10, 0.0, 1.0)
            self._sync_asset_legacy_fields(assets[idx])
            return True

        return False

    def _update_chain_state(
        self,
        *,
        success: bool,
        target_idx: int,
        attack_type: str,
        incoming_chain: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not incoming_chain:
            return {}

        chain_id = str(incoming_chain.get("chain_id", ""))
        if self._active_chain is None or self._active_chain.get("chain_id") != chain_id:
            return dict(incoming_chain)

        stage_idx = int(self._active_chain["stage_index"])
        stage_total = len(self._active_chain["stages"])

        out = {
            "chain_id": chain_id,
            "stage": attack_type,
            "stage_index": stage_idx,
            "stage_total": stage_total,
            "advance": False,
            "complete": False,
        }

        if success:
            self._active_chain["target_idx"] = target_idx
            self._active_chain["stage_index"] = stage_idx + 1
            out["advance"] = True

            if self._active_chain["stage_index"] >= stage_total:
                out["complete"] = True
                self._active_chain = None
        else:
            if self._rng.random() > 0.45:
                self._active_chain = None

        return out

    def _sync_asset_legacy_fields(self, asset: Dict[str, Any]) -> None:
        asset["backup_health"] = clamp(float(asset.get("backup_status", asset.get("backup_health", 1.0))), 0.0, 1.0)
        asset["criticality_score"] = clamp(
            float(asset.get("criticality", asset.get("criticality_score", 0.5))),
            0.0,
            1.0,
        )

    def _weighted_choice(self, weights: List[float]) -> int:
        adjusted = [max(0.05, w) for w in weights]
        total = sum(adjusted)
        pick = self._rng.random() * total
        cum = 0.0
        for i, w in enumerate(adjusted):
            cum += w
            if pick <= cum:
                return i
        return len(adjusted) - 1

    def _sample_categorical(self, probs: Dict[str, float]) -> str:
        total = sum(max(0.0, v) for v in probs.values())
        pick = self._rng.random() * total
        cum = 0.0
        for k, v in probs.items():
            cum += max(0.0, v)
            if pick <= cum:
                return k
        return list(probs.keys())[-1]

    def _resolve_goal(self, attacker_profile: Any) -> str:
        if isinstance(attacker_profile, dict):
            explicit_goal = str(attacker_profile.get("goal", "")).strip().lower()
            if explicit_goal:
                return explicit_goal
            name = str(attacker_profile.get("name", "")).lower()
        else:
            name = str(attacker_profile or "").lower()
        if "ransomware" in name:
            return "downtime"
        if "apt" in name:
            return "stealth_persistence"
        if "credential" in name:
            return "takeover"
        if "insider" in name:
            return "sabotage"
        return "balanced"

    def _goal_attack_bonus(self, attack_type: str) -> float:
        if self._attacker_goal == "downtime":
            if attack_type == "ransomware_attempt":
                return 0.08
            if attack_type in {"lateral_movement", "malware_dropper"}:
                return 0.04
        elif self._attacker_goal == "stealth_persistence":
            if attack_type in {"recon_scan", "malware_dropper", "privilege_escalation", "data_exfiltration"}:
                return 0.05
            if attack_type == "ransomware_attempt":
                return -0.04
        elif self._attacker_goal == "takeover":
            if attack_type in {"phishing_email", "password_spray", "credential_theft", "privilege_escalation"}:
                return 0.06
        elif self._attacker_goal == "sabotage":
            if attack_type in {"insider_misuse", "ransomware_attempt", "malware_dropper"}:
                return 0.06
        elif self._attacker_goal == "broad_disruption":
            if attack_type in {"malware_dropper", "phishing_email", "password_spray", "lateral_movement"}:
                return 0.05
        return 0.0

    def _default_goal_templates(self) -> List[List[str]]:
        if self._attacker_goal == "downtime":
            return [
                ["recon_scan", "credential_theft", "lateral_movement", "ransomware_attempt"],
                ["malware_dropper", "lateral_movement", "ransomware_attempt"],
            ]
        if self._attacker_goal == "stealth_persistence":
            return [
                ["recon_scan", "malware_dropper", "privilege_escalation", "data_exfiltration"],
                ["recon_scan", "credential_theft", "lateral_movement", "data_exfiltration"],
            ]
        if self._attacker_goal == "takeover":
            return [
                ["phishing_email", "credential_theft", "privilege_escalation", "lateral_movement"],
                ["password_spray", "credential_theft", "privilege_escalation"],
            ]
        if self._attacker_goal == "sabotage":
            return [
                ["insider_misuse", "credential_theft", "privilege_escalation", "ransomware_attempt"],
                ["insider_misuse", "lateral_movement", "data_exfiltration"],
            ]
        if self._attacker_goal == "broad_disruption":
            return [
                ["recon_scan", "malware_dropper", "lateral_movement"],
                ["password_spray", "credential_theft", "malware_dropper"],
            ]
        return [["recon_scan", "credential_theft", "lateral_movement"]]

    def _ingest_defender_context(self, defender_context: Optional[Dict[str, Any]]) -> None:
        if not defender_context:
            self._threat_pressure = clamp(self._threat_pressure * 0.99, 0.0, 1.0)
            return

        action = str(defender_context.get("last_blue_action", "")).strip()
        if action:
            self._blue_action_memory.append(action)
            if len(self._blue_action_memory) > 16:
                self._blue_action_memory = self._blue_action_memory[-16:]

        # Investigations reduce latent stealth pressure before attack selection.
        if action == "investigate_top_alert":
            self._stealth_meter = clamp(self._stealth_meter - 0.12, 0.0, 1.0)
            self._threat_pressure = clamp(self._threat_pressure - 0.05, 0.0, 1.0)
        elif action == "increase_monitoring":
            self._stealth_meter = clamp(self._stealth_meter - 0.06, 0.0, 1.0)

    def _update_hidden_state(self, *, attack_type: str, success: bool, stealth: bool, defender_action: str) -> None:
        stealth_gain = 0.0
        pressure_gain = 0.0

        if attack_type in {"recon_scan", "credential_theft", "privilege_escalation", "lateral_movement"}:
            stealth_gain += 0.04
        if stealth:
            stealth_gain += 0.08
        if success:
            pressure_gain += 0.05
            if attack_type in {"credential_theft", "lateral_movement", "ransomware_attempt", "data_exfiltration"}:
                pressure_gain += 0.04

        if defender_action == "investigate_top_alert":
            stealth_gain -= 0.10
            pressure_gain -= 0.03
        elif defender_action == "increase_monitoring":
            stealth_gain -= 0.05

        if self._attacker_goal == "stealth_persistence":
            stealth_gain *= 1.15
        elif self._attacker_goal == "downtime":
            pressure_gain *= 1.20

        self._stealth_meter = clamp(self._stealth_meter + stealth_gain, 0.0, 1.0)
        self._threat_pressure = clamp(self._threat_pressure + pressure_gain, 0.0, 1.0)
        self._goal_progress = clamp(self._goal_progress + max(0.0, pressure_gain), 0.0, 1.0)

    def _get_pivot_reason(self) -> str:
        if not self._blue_action_memory:
            return ""
        recent = self._blue_action_memory[-4:]
        if recent.count("patch_auth_server") >= 2:
            return "Blue repeatedly patched auth; RED pivoted toward phishing and credential theft."
        if recent.count("segment_finance_database") >= 1:
            return "Finance segmentation detected; RED pivoted to insider/web/auth paths."
        if recent.count("investigate_top_alert") >= 2:
            return "Frequent investigations observed; RED reduced noise and shifted to stealthier tactics."
        return ""
