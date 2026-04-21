from __future__ import annotations

from typing import Any, Dict


def build_incident_log(turn: int, intelligence: Dict[str, Any]) -> Dict[str, Any]:
    forecast = intelligence.get("forecast", {}) if isinstance(intelligence.get("forecast", {}), dict) else {}
    primary = forecast.get("primary", {}) if isinstance(forecast.get("primary", {}), dict) else {}
    reasoning = intelligence.get("reasoning", {}) if isinstance(intelligence.get("reasoning", {}), dict) else {}

    attack = str(primary.get("attack", "unknown"))
    prob = float(primary.get("probability", 0.0))
    confidence = float(reasoning.get("decision_confidence", 0.0))
    action = str(reasoning.get("action", "unknown"))

    summary = (
        f"Turn {turn}: Threat forecast shifted toward {attack} ({prob:.0%}). "
        f"Action {action} executed with confidence {confidence:.2f}."
    )
    return {
        "turn": turn,
        "forecast_primary": {"attack": attack, "probability": prob},
        "decision_confidence": confidence,
        "executed_action": action,
        "summary": summary,
    }
