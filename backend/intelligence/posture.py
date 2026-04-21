from __future__ import annotations

from typing import Any, Dict, List


def summarize_posture(
    *,
    assets: List[Dict[str, Any]],
    network_risk: float,
    risk_breakdown: Dict[str, float],
    scenario_name: str,
) -> Dict[str, Any]:
    compromised = sum(1 for a in assets if bool(a.get("compromised", False)))
    infected = sum(1 for a in assets if bool(a.get("infected", False)))
    down = sum(1 for a in assets if not bool(a.get("uptime_status", True)))

    credential_exposure = float(risk_breakdown.get("credential_exposure", 0.0))
    patch_debt = float(risk_breakdown.get("patch_debt", 0.0))

    if network_risk < 0.22 and compromised == 0 and down == 0:
        level = "healthy"
    elif network_risk < 0.40 and compromised <= 1:
        level = "guarded"
    elif network_risk < 0.62:
        level = "elevated"
    else:
        level = "critical"

    highlights: List[str] = []
    if credential_exposure > 0.45:
        highlights.append("credential exposure elevated")
    if patch_debt > 0.45:
        highlights.append("patch debt is accumulating")
    if compromised > 0:
        highlights.append(f"{compromised} compromised assets present")
    if infected > 0:
        highlights.append(f"{infected} infected assets observed")
    if down > 0:
        highlights.append(f"{down} services degraded")
    if not highlights:
        highlights.append("core controls remain stable")

    summary = f"{scenario_name.title()} posture {level}. " + "; ".join(highlights) + "."

    return {
        "level": level,
        "summary": summary,
        "highlights": highlights,
        "network_risk": float(network_risk),
    }
