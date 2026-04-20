from __future__ import annotations

from typing import Any, Dict, List, Optional

from .risk import breach_rate, compute_network_risk, uptime_ratio


def compute_reward(
    assets_prev: List[Dict[str, Any]],
    assets_curr: List[Dict[str, Any]],
    action_name: str,
    red_log: Dict[str, Any],
    action_cost: float,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """Reward shaping aligned to prevention, resilience, and efficient defense operations."""
    ctx = context or {}
    prev_risk = compute_network_risk(assets_prev, context=ctx)
    curr_risk = compute_network_risk(assets_curr, context=ctx)

    prev_uptime = uptime_ratio(assets_prev)
    curr_uptime = uptime_ratio(assets_curr)

    prev_breach = breach_rate(assets_prev)
    curr_breach = breach_rate(assets_curr)

    # Dense objective terms that directly align with core mission outcomes.
    risk_term = (prev_risk - curr_risk) * 10.0
    uptime_term = (curr_uptime - prev_uptime) * 6.0
    breach_term = (prev_breach - curr_breach) * 8.0

    # Fast recovery and prevention bonuses.
    recovered = sum(1 for p, c in zip(assets_prev, assets_curr) if (not p["uptime_status"]) and c["uptime_status"])
    cleaned = sum(1 for p, c in zip(assets_prev, assets_curr) if p["infected"] and (not c["infected"]))
    prevention_bonus = 0.45 if (not red_log.get("success", False)) and action_name != "do_nothing" else 0.0

    reward = risk_term + uptime_term + breach_term + recovered * 0.9 + cleaned * 0.7 + prevention_bonus

    # Strong critical-asset incentives, with explicit finance protection focus.
    finance_prev = next(a for a in assets_prev if a["name"] == "Finance Database")
    finance_curr = next(a for a in assets_curr if a["name"] == "Finance Database")
    if (not finance_prev["compromised"]) and finance_curr["compromised"]:
        reward -= 2.0
    if finance_prev["infected"] and (not finance_curr["infected"]):
        reward += 0.6

    # Action economy and anti-reward-hacking controls.
    reward -= action_cost

    previous_action = str(ctx.get("previous_action", ""))
    if previous_action == action_name and curr_risk >= prev_risk:
        reward -= 0.18

    action_was_wasteful = False
    if action_name == "restore_backup":
        had_downtime = any(not a["uptime_status"] for a in assets_prev)
        had_low_backup = any(float(a["backup_health"]) < 0.65 for a in assets_prev)
        action_was_wasteful = not (had_downtime or had_low_backup)
    elif action_name == "isolate_suspicious_host":
        had_incident = any(a["infected"] or a["compromised"] for a in assets_prev)
        action_was_wasteful = not had_incident

    if action_was_wasteful:
        reward -= 0.35

    if action_name == "do_nothing" and (
        curr_risk > 0.33 or red_log.get("success", False) or any(a["infected"] for a in assets_curr)
    ):
        reward -= 0.55

    # Hard negative outcomes and neglect penalties.
    compromised_assets = sum(1 for a in assets_curr if a["compromised"])
    down_assets = sum(1 for a in assets_curr if not a["uptime_status"])
    reward -= compromised_assets * 0.35
    reward -= down_assets * 0.20

    # Terminal penalties discourage catastrophic but short-horizon behavior.
    if bool(ctx.get("terminated", False)):
        reason = str(ctx.get("termination_reason", ""))
        if reason == "critical_breach":
            reward -= 3.5
        elif reason == "downtime_cascade":
            reward -= 2.8
        else:
            reward -= 2.0

    return float(reward)
