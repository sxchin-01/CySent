from __future__ import annotations

from typing import Any, Dict, List, Optional


ACTION_TO_ID: Dict[str, int] = {
    "do_nothing": 0,
    "patch_hr_systems": 1,
    "patch_web_server": 2,
    "patch_auth_server": 3,
    "rotate_credentials": 4,
    "isolate_suspicious_host": 5,
    "increase_monitoring": 6,
    "restore_backup": 7,
    "deploy_honeypot": 8,
    "phishing_training": 9,
    "investigate_top_alert": 10,
    "segment_finance_database": 11,
}


class IntelligenceController:
    """Action-source abstraction for PPO, heuristic, and human-manual control modes."""

    def __init__(self, strategy_mode: str = "balanced", action_source: str = "ppo_ai") -> None:
        self.strategy_mode = strategy_mode
        self.action_source = action_source

    def configure(self, *, strategy_mode: str, action_source: str) -> None:
        self.strategy_mode = strategy_mode
        self.action_source = action_source

    def recommend_action(
        self,
        *,
        assets: List[Dict[str, Any]],
        risk_breakdown: Dict[str, float],
        forecast: Dict[str, Any],
        manual_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        source = self.action_source
        if source == "human_manual" and manual_action is not None:
            action_id = int(manual_action)
            return {
                "source": source,
                "recommended_action_id": action_id,
                "recommended_action_name": self._name_for_id(action_id),
                "confidence": 1.0,
                "rationale": "Manual action supplied by human operator.",
            }

        if source in {"heuristic_ai", "ppo_ai"}:
            choice = self._heuristic_choice(assets=assets, risk_breakdown=risk_breakdown, forecast=forecast)
            # For ppo_ai source, this is a recommendation wrapper only; external PPO action remains authoritative.
            confidence = 0.62 if source == "heuristic_ai" else 0.55
            return {
                "source": source,
                "recommended_action_id": choice["action_id"],
                "recommended_action_name": choice["action_name"],
                "confidence": confidence,
                "rationale": choice["reason"],
            }

        return {
            "source": source,
            "recommended_action_id": 0,
            "recommended_action_name": "do_nothing",
            "confidence": 0.20,
            "rationale": "Unknown source; defaulting to passive stance.",
        }

    def compare_actions(self, *, executed_action_name: str, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        recommended_name = str(recommendation.get("recommended_action_name", "do_nothing"))
        aligned = executed_action_name == recommended_name
        return {
            "aligned": aligned,
            "executed_action": executed_action_name,
            "recommended_action": recommended_name,
            "delta": "none" if aligned else "different",
        }

    def _heuristic_choice(
        self,
        *,
        assets: List[Dict[str, Any]],
        risk_breakdown: Dict[str, float],
        forecast: Dict[str, Any],
    ) -> Dict[str, Any]:
        infected = any(bool(a.get("infected", False)) for a in assets)
        compromised = any(bool(a.get("compromised", False)) for a in assets)
        downtime = any(not bool(a.get("uptime_status", True)) for a in assets)

        credential_exposure = float(risk_breakdown.get("credential_exposure", 0.0))
        patch_debt = float(risk_breakdown.get("patch_debt", 0.0))
        primary = forecast.get("primary", {}) if isinstance(forecast.get("primary", {}), dict) else {}
        primary_attack = str(primary.get("attack", ""))

        mode = self.strategy_mode

        if mode == "conservative":
            if downtime:
                return self._pack("restore_backup", "Conservative mode prioritizes service continuity.")
            if credential_exposure > 0.42:
                return self._pack("rotate_credentials", "Conservative mode reduces identity risk proactively.")
            return self._pack("increase_monitoring", "Conservative mode increases visibility before intervention.")

        if mode == "aggressive":
            if compromised or infected:
                return self._pack("investigate_top_alert", "Aggressive mode neutralizes active compromise quickly.")
            if primary_attack in {"lateral_movement", "ransomware_attempt", "data_exfiltration"}:
                return self._pack("segment_finance_database", "Aggressive mode constrains blast radius under severe threat.")
            return self._pack("isolate_suspicious_host", "Aggressive mode isolates likely attack paths.")

        # balanced
        if compromised:
            return self._pack("investigate_top_alert", "Balanced mode responds to confirmed compromise.")
        if credential_exposure > 0.45 or primary_attack in {"credential_theft", "password_spray", "phishing_email"}:
            return self._pack("rotate_credentials", "Balanced mode addresses identity attack pressure.")
        if patch_debt > 0.46:
            return self._pack("patch_auth_server", "Balanced mode reduces exploitability on critical auth surface.")
        return self._pack("increase_monitoring", "Balanced mode keeps defensive telemetry strong.")

    def _pack(self, action_name: str, reason: str) -> Dict[str, Any]:
        action_id = ACTION_TO_ID[action_name]
        return {"action_id": action_id, "action_name": action_name, "reason": reason}

    def _name_for_id(self, action_id: int) -> str:
        for name, idx in ACTION_TO_ID.items():
            if idx == action_id:
                return name
        return "do_nothing"
