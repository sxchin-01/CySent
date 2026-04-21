from __future__ import annotations

from typing import Any, Dict, List, Optional

from .risk import breach_rate, compute_network_risk, uptime_ratio


def _find_asset(assets: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for asset in assets:
        if asset["name"] == name:
            return asset
    raise ValueError(f"missing asset: {name}")


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

    # Keep dense risk incentive but avoid overly action-centric optimization.
    risk_term = (prev_risk - curr_risk) * 13.0
    uptime_term = (curr_uptime - prev_uptime) * 3.2
    breach_term = (prev_breach - curr_breach) * 9.0

    # Fast recovery and prevention bonuses.
    recovered = sum(1 for p, c in zip(assets_prev, assets_curr) if (not p["uptime_status"]) and c["uptime_status"])
    cleaned = sum(1 for p, c in zip(assets_prev, assets_curr) if p["infected"] and (not c["infected"]))
    recovered_compromised = sum(1 for p, c in zip(assets_prev, assets_curr) if p["compromised"] and (not c["compromised"]))
    infected_prev = sum(1 for a in assets_prev if a["infected"])
    infected_curr = sum(1 for a in assets_curr if a["infected"])
    compromised_prev = sum(1 for a in assets_prev if a["compromised"])
    compromised_curr = sum(1 for a in assets_curr if a["compromised"])

    chain_ctx = red_log.get("chain", {}) if isinstance(red_log.get("chain", {}), dict) else {}
    chain_interrupted = bool(chain_ctx) and (not bool(chain_ctx.get("advance", False))) and (not bool(red_log.get("success", False)))

    prevention_bonus = 0.20 if (not red_log.get("success", False)) and action_name != "do_nothing" else 0.0
    if chain_interrupted:
        prevention_bonus += 0.70

    reward = risk_term + uptime_term + breach_term + recovered * 0.7 + cleaned * 1.2 + recovered_compromised * 1.3 + prevention_bonus

    if (infected_prev + compromised_prev) > (infected_curr + compromised_curr):
        reward += 1.0

    # Strong critical-asset incentives, with explicit finance protection focus.
    finance_prev = next(a for a in assets_prev if a["name"] == "Finance Database")
    finance_curr = next(a for a in assets_curr if a["name"] == "Finance Database")
    auth_prev = _find_asset(assets_prev, "Auth Server")
    auth_curr = _find_asset(assets_curr, "Auth Server")
    red_success = bool(red_log.get("success", False))
    attack_type = str(red_log.get("attack", ""))

    if (not finance_prev["compromised"]) and finance_curr["compromised"]:
        reward -= 2.0
    if finance_prev["infected"] and (not finance_curr["infected"]):
        reward += 0.6

    # Keep action nudges small; outcomes should dominate.
    if action_name == "patch_auth_server" and auth_prev["patch_level"] < 0.75 and (not auth_prev["compromised"]) and curr_risk < prev_risk:
        reward += 0.35
    if action_name == "rotate_credentials" and attack_type in {"phishing_email", "password_spray", "credential_theft"} and curr_risk < prev_risk:
        reward += 0.45
    if action_name == "segment_finance_database" and (not bool(ctx.get("segmented_finance", False))) and (not finance_prev["compromised"]) and curr_risk < prev_risk:
        reward += 0.45
    if action_name == "investigate_top_alert" and (red_success or any(a["infected"] or a["compromised"] for a in assets_prev)) and (infected_curr + compromised_curr) < (infected_prev + compromised_prev):
        reward += 0.35

    # Outcome-dominant rewards.
    if (not red_success) and attack_type == "credential_theft":
        reward += 3.8
    if (not red_success) and attack_type in {"privilege_escalation", "lateral_movement"}:
        reward += 3.0
    if (not red_success) and attack_type == "phishing_email":
        reward += 2.0
    if (not finance_curr["compromised"]) and attack_type in {"lateral_movement", "data_exfiltration", "ransomware_attempt"}:
        reward += 2.4
    risk_delta = prev_risk - curr_risk
    if risk_delta > 0.05:
        reward += 2.5

    # Delayed credit: give small credit if prior action likely enabled this prevention.
    previous_action = str(ctx.get("previous_action", ""))
    if (not red_success) and previous_action in {
        "patch_auth_server",
        "patch_web_server",
        "rotate_credentials",
        "segment_finance_database",
        "investigate_top_alert",
        "increase_monitoring",
    }:
        reward += 0.55

    # Small threat-neglect penalties (-2 to -8 range).
    if red_success and attack_type == "credential_theft":
        reward -= 5.0

    # Late-episode posture debt on critical systems.
    if bool(ctx.get("truncated", False)):
        if auth_curr["patch_level"] < 0.60:
            reward -= 3.0
        if (finance_curr["patch_level"] < 0.62) and (not bool(ctx.get("segmented_finance", False))):
            reward -= 3.0

    response_actions = {
        "patch_auth_server",
        "patch_web_server",
        "rotate_credentials",
        "segment_finance_database",
        "investigate_top_alert",
        "isolate_suspicious_host",
        "restore_backup",
    }
    high_severity_attacks = {"credential_theft", "privilege_escalation", "lateral_movement", "ransomware_attempt", "data_exfiltration"}
    if red_success and attack_type in high_severity_attacks and action_name not in response_actions:
        reward -= 4.0

    # Action-quality penalties for unjustified proactive moves.
    credential_threat = attack_type in {"phishing_email", "password_spray", "credential_theft"}
    credential_exposure_prev = sum(float(a["credential_risk"]) for a in assets_prev) / max(len(assets_prev), 1)
    if action_name == "rotate_credentials" and (not credential_threat) and credential_exposure_prev < 0.42:
        reward -= 1.8
    if action_name == "segment_finance_database" and bool(ctx.get("segmented_finance", False)):
        reward -= 2.2
    web_prev = _find_asset(assets_prev, "Web Server")
    if action_name == "patch_auth_server" and auth_prev["patch_level"] > 0.82 and (not auth_prev["infected"]) and (not auth_prev["compromised"]):
        reward -= 1.6
    if action_name == "patch_web_server" and web_prev["patch_level"] > 0.82 and (not web_prev["infected"]) and (not web_prev["compromised"]):
        reward -= 1.6
    unresolved_alert = red_success or any(a["infected"] or a["compromised"] for a in assets_prev)
    if action_name == "investigate_top_alert" and (not unresolved_alert):
        reward -= 1.5

    # Action economy and anti-reward-hacking controls.
    reward -= action_cost

    if previous_action == action_name and curr_risk >= prev_risk:
        reward -= 0.45

    spam_sensitive = {"rotate_credentials", "patch_auth_server", "patch_web_server", "investigate_top_alert", "segment_finance_database"}
    if previous_action == action_name and action_name in spam_sensitive:
        if curr_risk >= prev_risk:
            reward -= 1.0
        if red_success:
            reward -= 0.8

    # Repetition cooldown: small escalating penalties for repeated low-value actions.
    low_value_actions = {"do_nothing", "increase_monitoring", "phishing_training", "deploy_honeypot"}
    threat_exists = any(a["infected"] or a["compromised"] for a in assets_curr)
    if previous_action == action_name and action_name in low_value_actions and threat_exists:
        repeat_penalty = 1.0
        if curr_risk >= prev_risk:
            repeat_penalty += 1.0
        if red_success:
            repeat_penalty += 1.0
        reward -= repeat_penalty

    action_was_wasteful = False
    if action_name == "restore_backup":
        had_downtime = any(not a["uptime_status"] for a in assets_prev)
        had_low_backup = any(float(a["backup_health"]) < 0.65 for a in assets_prev)
        action_was_wasteful = not (had_downtime or had_low_backup)
    elif action_name == "isolate_suspicious_host":
        had_incident = any(a["infected"] or a["compromised"] for a in assets_prev)
        action_was_wasteful = not had_incident

    if action_was_wasteful:
        reward -= 1.6

    if action_name == "do_nothing" and (
        curr_risk > 0.33 or red_log.get("success", False) or any(a["infected"] for a in assets_curr)
    ):
        reward -= 3.5

    # Risk quality: reward sustained low-risk posture and penalize latent hidden-risk growth.
    if curr_risk < 0.25 and (not red_success):
        reward += 0.65
    hidden_pressure = float(red_log.get("threat_pressure", 0.0))
    stealth_buildup = float(red_log.get("stealth_buildup", 0.0))
    if curr_uptime > 0.90 and (hidden_pressure > 0.55 or stealth_buildup > 0.03):
        reward -= 1.3
    if action_name in {"investigate_top_alert", "increase_monitoring"} and stealth_buildup < 0.0:
        reward += 0.7

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

    # Keep PPO-safe scale without changing signal ordering.
    reward = max(-12.0, min(12.0, reward))
    return float(reward)
