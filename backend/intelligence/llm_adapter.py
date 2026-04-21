from __future__ import annotations

from typing import Any, Dict, Protocol


class LLMSummaryAdapter(Protocol):
    """Optional adapter protocol for future LLM-backed turn and replay summaries."""

    def summarize_turn(self, intelligence_payload: Dict[str, Any]) -> str:
        ...


class NoopLLMSummaryAdapter:
    """Default local adapter that provides deterministic text without external calls."""

    def summarize_turn(self, intelligence_payload: Dict[str, Any]) -> str:
        posture = intelligence_payload.get("posture", {}) if isinstance(intelligence_payload.get("posture", {}), dict) else {}
        forecast = intelligence_payload.get("forecast", {}) if isinstance(intelligence_payload.get("forecast", {}), dict) else {}
        reasoning = intelligence_payload.get("reasoning", {}) if isinstance(intelligence_payload.get("reasoning", {}), dict) else {}

        posture_level = str(posture.get("level", "unknown"))
        primary = forecast.get("primary", {}) if isinstance(forecast.get("primary", {}), dict) else {}
        attack = str(primary.get("attack", "unknown"))
        prob = float(primary.get("probability", 0.0))
        action = str(reasoning.get("action", "unknown"))
        conf = float(reasoning.get("decision_confidence", 0.0))

        return (
            f"Posture {posture_level}; forecast favors {attack} ({prob:.0%}); "
            f"action {action} with confidence {conf:.2f}."
        )
