from __future__ import annotations

from typing import Any, Dict, List


def _average(items: List[float]) -> float:
    return sum(items) / max(len(items), 1)


def _confidence_from_signals(
    *,
    severity: float,
    clarity: float,
    converging_signals: int,
) -> float:
    score = 0.35 * severity + 0.35 * clarity + 0.30 * min(converging_signals / 4.0, 1.0)
    return max(0.0, min(1.0, score))


def build_action_reasoning(
    *,
    action_name: str,
    assets_prev: List[Dict[str, Any]],
    assets_curr: List[Dict[str, Any]],
    red_log: Dict[str, Any],
    forecast: Dict[str, Any],
    strategy_mode: str,
    posture_level: str,
) -> Dict[str, Any]:
    prev_cred = _average([float(a.get("credential_risk", 0.0)) for a in assets_prev])
    curr_cred = _average([float(a.get("credential_risk", 0.0)) for a in assets_curr])
    prev_patch = _average([float(a.get("patch_level", 1.0)) for a in assets_prev])
    curr_patch = _average([float(a.get("patch_level", 1.0)) for a in assets_curr])

    auth_prev = next((a for a in assets_prev if a.get("name") == "Auth Server"), None)
    finance_prev = next((a for a in assets_prev if a.get("name") == "Finance Database"), None)

    primary = forecast.get("primary", {}) if isinstance(forecast.get("primary", {}), dict) else {}
    primary_attack = str(primary.get("attack", "unknown"))
    primary_prob = float(primary.get("probability", 0.0))

    signals: List[Dict[str, Any]] = []

    if prev_cred > 0.42 or primary_attack in {"credential_theft", "password_spray", "phishing_email"}:
        signals.append({"signal": "credential_exposure", "value": round(prev_cred, 4), "impact": "high"})
    if auth_prev is not None and float(auth_prev.get("patch_level", 1.0)) < 0.65:
        signals.append({"signal": "auth_patch_weakness", "value": float(auth_prev.get("patch_level", 1.0)), "impact": "high"})
    if finance_prev is not None and (bool(finance_prev.get("infected", False)) or bool(finance_prev.get("compromised", False))):
        signals.append({"signal": "finance_pressure", "value": 1.0, "impact": "high"})
    if bool(red_log.get("success", False)):
        signals.append({"signal": "recent_attack_success", "value": 1.0, "impact": "high"})
    if posture_level in {"elevated", "critical"}:
        signals.append({"signal": "posture_alert", "value": posture_level, "impact": "high"})

    reason_map = {
        "rotate_credentials": "Rotated credentials because credential exposure and identity attack likelihood were elevated.",
        "segment_finance_database": "Segmented finance database because lateral movement pressure toward critical systems increased.",
        "patch_auth_server": "Patched auth server due to weak patch posture and repeated exploitation risk.",
        "patch_web_server": "Patched web server to reduce exploitability on exposed internet-facing surfaces.",
        "restore_backup": "Restored backup to recover uptime and reduce ransomware impact.",
        "isolate_suspicious_host": "Isolated suspicious host to contain spread and reduce lateral movement.",
        "increase_monitoring": "Increased monitoring to improve detection against stealthy attacker behavior.",
        "investigate_top_alert": "Investigated top alert to remove active compromise indicators.",
        "deploy_honeypot": "Deployed honeypot to divert reconnaissance and credential theft activity.",
        "phishing_training": "Ran phishing training to lower user susceptibility against identity attacks.",
        "patch_hr_systems": "Patched HR systems to reduce vulnerability exposure in user-facing workflows.",
        "do_nothing": "Held current action to preserve stability while monitoring threat changes.",
    }

    explanation = reason_map.get(action_name, f"Executed {action_name} based on current risk signals.")

    if primary_attack != "unknown":
        explanation += f" Forecast primary threat: {primary_attack} ({primary_prob:.0%})."

    severity = min(1.0, 0.25 + 0.75 * primary_prob)
    clarity = min(1.0, 0.50 + abs(curr_cred - prev_cred) + abs(curr_patch - prev_patch))
    confidence = _confidence_from_signals(
        severity=severity,
        clarity=clarity,
        converging_signals=len(signals),
    )

    return {
        "action": action_name,
        "strategy_mode": strategy_mode,
        "signals": signals,
        "threat_focus": {
            "primary_attack": primary_attack,
            "primary_probability": float(primary_prob),
        },
        "decision_confidence": float(confidence),
        "explanation": explanation,
    }
