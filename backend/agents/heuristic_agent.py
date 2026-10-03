from __future__ import annotations

from typing import Any, Dict, List

from backend.env.security_env import ACTION_NAMES


ACTION_IDS = {name: action_id for action_id, name in ACTION_NAMES.items()}


class HeuristicAgent:
    """Deterministic BLUE baseline using only observable environment state."""

    def __init__(self) -> None:
        self.previous_action = "reset"

    def reset(self) -> None:
        self.previous_action = "reset"

    def predict_action(self, state: Dict[str, Any]) -> int:
        assets = list(state.get("assets", []))
        alerts = list(state.get("alerts", []))
        defender = state.get("defender", {}) if isinstance(state.get("defender", {}), dict) else {}
        cooldowns = defender.get("cooldowns", {}) if isinstance(defender.get("cooldowns", {}), dict) else {}
        risk = state.get("risk_breakdown", {}) if isinstance(state.get("risk_breakdown", {}), dict) else {}

        def ready(action_name: str) -> bool:
            return int(cooldowns.get(action_name, 0)) <= 0

        def asset(name: str) -> Dict[str, Any]:
            return next((item for item in assets if item.get("name") == name), {})

        critical_incidents = [
            item
            for item in assets
            if float(item.get("criticality_score", item.get("criticality", 0.0))) >= 0.88
            and (bool(item.get("infected")) or bool(item.get("compromised")))
        ]
        unisolated_incidents = [
            item
            for item in assets
            if (bool(item.get("infected")) or bool(item.get("compromised")))
            and not bool(item.get("isolated"))
        ]
        down_assets = [item for item in assets if not bool(item.get("uptime_status", True))]
        severe_alert = any(float(item.get("severity_score", 0.0)) >= 0.65 for item in alerts)

        if critical_incidents and unisolated_incidents:
            choice = "isolate_suspicious_host"
        elif severe_alert and ready("investigate_top_alert"):
            choice = "investigate_top_alert"
        elif down_assets and ready("restore_backup"):
            choice = "restore_backup"
        elif unisolated_incidents:
            choice = "isolate_suspicious_host"
        elif float(risk.get("credential_exposure", 0.0)) >= 0.48 and ready("rotate_credentials"):
            choice = "rotate_credentials"
        elif (
            float(risk.get("segmentation_gap", 0.0)) >= 0.5
            and float(risk.get("asset_exposure", 0.0)) >= 0.16
            and ready("segment_finance_database")
        ):
            choice = "segment_finance_database"
        elif float(asset("Auth Server").get("patch_level", 1.0)) < 0.72:
            choice = "patch_auth_server"
        elif float(asset("Web Server").get("patch_level", 1.0)) < 0.72:
            choice = "patch_web_server"
        elif float(asset("HR Systems").get("patch_level", 1.0)) < 0.65:
            choice = "patch_hr_systems"
        elif alerts and ready("investigate_top_alert"):
            choice = "investigate_top_alert"
        else:
            detection_levels: List[float] = [float(item.get("detection_level", 0.0)) for item in assets]
            mean_detection = sum(detection_levels) / max(len(detection_levels), 1)
            if mean_detection < 0.68 and self.previous_action != "increase_monitoring":
                choice = "increase_monitoring"
            elif float(risk.get("credential_exposure", 0.0)) >= 0.34 and self.previous_action != "phishing_training":
                choice = "phishing_training"
            elif self.previous_action != "deploy_honeypot":
                choice = "deploy_honeypot"
            else:
                choice = "do_nothing"

        self.previous_action = choice
        return ACTION_IDS[choice]
