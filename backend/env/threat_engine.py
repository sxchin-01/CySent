from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

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
    """RED threat engine that drives stateful attacks from hand-crafted logic."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def seed(self, seed: int | None) -> None:
        self._rng.seed(seed)

    def choose_attack(self, assets: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Choose target asset and attack type from dynamic network state."""
        risks = [compute_asset_risk(asset) for asset in assets]
        idx = self._weighted_choice(risks)
        asset = assets[idx]

        # Attack distribution shifts with observed weakness.
        probs = {
            "phishing_email": 0.10 + (1.0 - asset["detection_level"]) * 0.08,
            "password_spray": 0.10 + asset["credential_risk"] * 0.12,
            "malware_dropper": 0.10 + (1.0 - asset["patch_level"]) * 0.12,
            "credential_theft": 0.10 + asset["credential_risk"] * 0.12,
            "privilege_escalation": 0.08 + (1.0 - asset["patch_level"]) * 0.10,
            "lateral_movement": 0.08 + (0.25 if asset["compromised"] else 0.0),
            "ransomware_attempt": 0.08 + (1.0 - asset["backup_health"]) * 0.10,
            "data_exfiltration": 0.08 + (0.30 if asset["compromised"] else 0.0),
            "insider_misuse": 0.09,
            "recon_scan": 0.09 + (1.0 - asset["detection_level"]) * 0.06,
        }

        attack = self._sample_categorical(probs)
        return idx, attack

    def apply_attack(
        self,
        assets: List[Dict[str, Any]],
        target_idx: int,
        attack_type: str,
        segmented_finance: bool,
        honeypot_active: bool,
    ) -> Dict[str, Any]:
        asset = assets[target_idx]
        log: Dict[str, Any] = {
            "target": asset["name"],
            "attack": attack_type,
            "success": False,
            "escalation": False,
            "notes": "",
        }

        if asset["isolated"]:
            log["notes"] = "target is isolated; attack had reduced impact"

        base_success = self._success_probability(asset, attack_type)
        if asset["isolated"]:
            base_success *= 0.45
        if honeypot_active and attack_type in {"recon_scan", "lateral_movement", "credential_theft"}:
            base_success *= 0.6
            log["notes"] = "honeypot diverted part of attack pressure"

        success = self._rng.random() < base_success
        log["success"] = success

        if success:
            self._mutate_state(asset, attack_type)
            if attack_type in {"credential_theft", "privilege_escalation", "lateral_movement"}:
                log["escalation"] = self._maybe_escalate(assets, target_idx, segmented_finance)

        return log

    def _success_probability(self, asset: Dict[str, Any], attack_type: str) -> float:
        patch = clamp(float(asset["patch_level"]), 0.0, 1.0)
        detect = clamp(float(asset["detection_level"]), 0.0, 1.0)
        cred = clamp(float(asset["credential_risk"]), 0.0, 1.0)
        backup = clamp(float(asset["backup_health"]), 0.0, 1.0)

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
        }

        return clamp(mapping[attack_type], 0.05, 0.95)

    def _mutate_state(self, asset: Dict[str, Any], attack_type: str) -> None:
        if attack_type in {"phishing_email", "password_spray", "credential_theft", "insider_misuse"}:
            asset["credential_risk"] = clamp(asset["credential_risk"] + 0.16, 0.0, 1.0)

        if attack_type in {"malware_dropper", "lateral_movement", "ransomware_attempt"}:
            asset["infected"] = True
            asset["patch_level"] = clamp(asset["patch_level"] - 0.06, 0.0, 1.0)

        if attack_type in {"privilege_escalation", "lateral_movement", "data_exfiltration", "ransomware_attempt"}:
            asset["compromised"] = True
            asset["uptime_status"] = False if attack_type in {"ransomware_attempt", "data_exfiltration"} else asset["uptime_status"]

        if attack_type == "recon_scan":
            asset["detection_level"] = clamp(asset["detection_level"] - 0.08, 0.0, 1.0)

        if attack_type == "ransomware_attempt":
            asset["backup_health"] = clamp(asset["backup_health"] - 0.18, 0.0, 1.0)

    def _maybe_escalate(self, assets: List[Dict[str, Any]], source_idx: int, segmented_finance: bool) -> bool:
        candidate_idxs = [i for i in range(len(assets)) if i != source_idx and not assets[i]["isolated"]]
        if not candidate_idxs:
            return False

        if segmented_finance:
            candidate_idxs = [
                i
                for i in candidate_idxs
                if assets[i]["name"] != "Finance Database"
            ]
            if not candidate_idxs:
                return False

        if self._rng.random() < 0.45:
            idx = self._rng.choice(candidate_idxs)
            assets[idx]["infected"] = True
            assets[idx]["credential_risk"] = clamp(assets[idx]["credential_risk"] + 0.10, 0.0, 1.0)
            return True

        return False

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
