from __future__ import annotations

from typing import Any, Dict, List, Optional


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_asset_risk(asset: Dict[str, Any], *, segmented_finance: bool = False) -> float:
    """Compute a normalized risk score for a single asset in [0, 1]."""
    patch_risk = 1.0 - clamp(float(asset["patch_level"]), 0.0, 1.0)
    infected_risk = 1.0 if asset["infected"] else 0.0
    compromised_risk = 1.0 if asset["compromised"] else 0.0
    credential_risk = clamp(float(asset["credential_risk"]), 0.0, 1.0)
    monitoring_weakness = 1.0 - clamp(float(asset["detection_level"]), 0.0, 1.0)
    backup_risk = 1.0 - clamp(float(asset.get("backup_health", 1.0)), 0.0, 1.0)
    uptime_penalty = 1.0 if not asset["uptime_status"] else 0.0

    # Ransomware spread pressure is elevated when backup resilience is weak on infected hosts.
    ransomware_spread_risk = backup_risk * (1.0 if (asset["infected"] or asset["compromised"]) else 0.35)

    raw = (
        0.12 * patch_risk
        + 0.18 * infected_risk
        + 0.30 * compromised_risk
        + 0.12 * credential_risk
        + 0.10 * monitoring_weakness
        + 0.06 * backup_risk
        + 0.07 * ransomware_spread_risk
        + 0.05 * uptime_penalty
    )

    # Critical assets amplify impact and should bias policy toward prevention.
    criticality = clamp(float(asset["criticality_score"]), 0.0, 1.0)
    criticality_weight = 0.65 + 0.55 * criticality

    # Finance posture should be strongly protected; no segmentation increases blast radius.
    finance_escalation = 0.0
    if asset["name"] == "Finance Database":
        finance_escalation += 0.07 if not segmented_finance else 0.0
        finance_escalation += 0.08 if asset["compromised"] else 0.0
        finance_escalation += 0.04 if asset["infected"] else 0.0

    return clamp(raw * criticality_weight + finance_escalation, 0.0, 1.0)


def compute_risk_breakdown(
    assets: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return normalized risk components used to compute overall network risk."""
    if not assets:
        return {
            "asset_exposure": 0.0,
            "compromised_hosts": 0.0,
            "infected_hosts": 0.0,
            "patch_debt": 0.0,
            "credential_exposure": 0.0,
            "monitoring_weakness": 0.0,
            "ransomware_spread": 0.0,
            "segmentation_gap": 0.0,
            "ignored_alert_pressure": 0.0,
            "network_risk": 0.0,
        }

    ctx = context or {}
    segmented_finance = bool(ctx.get("segmented_finance", False))
    n_assets = len(assets)

    asset_risks = [compute_asset_risk(a, segmented_finance=segmented_finance) for a in assets]
    weights = [clamp(float(a["criticality_score"]), 0.1, 1.0) for a in assets]
    asset_exposure = clamp(sum(r * w for r, w in zip(asset_risks, weights)) / max(sum(weights), 1e-6), 0.0, 1.0)

    compromised_hosts = sum(1 for a in assets if a["compromised"]) / n_assets
    infected_hosts = sum(1 for a in assets if a["infected"]) / n_assets
    patch_debt = sum(1.0 - clamp(float(a["patch_level"]), 0.0, 1.0) for a in assets) / n_assets
    credential_exposure = sum(clamp(float(a["credential_risk"]), 0.0, 1.0) for a in assets) / n_assets
    monitoring_weakness = sum(1.0 - clamp(float(a["detection_level"]), 0.0, 1.0) for a in assets) / n_assets
    ransomware_spread = sum(
        (1.0 - clamp(float(a.get("backup_health", 1.0)), 0.0, 1.0))
        * (1.0 if (a["infected"] or a["compromised"]) else 0.35)
        for a in assets
    ) / n_assets

    segmentation_gap = 0.0 if segmented_finance else 1.0

    last_action = str(ctx.get("last_action", ""))
    red_success = bool(ctx.get("red_success", False))
    ignored_alert_pressure = 0.0
    if red_success and last_action in {"do_nothing", "patch_hr_systems", "patch_web_server", "patch_auth_server"}:
        ignored_alert_pressure = 1.0

    network_risk = clamp(
        0.45 * asset_exposure
        + 0.12 * compromised_hosts
        + 0.08 * infected_hosts
        + 0.08 * patch_debt
        + 0.07 * credential_exposure
        + 0.06 * monitoring_weakness
        + 0.05 * ransomware_spread
        + 0.05 * segmentation_gap
        + 0.04 * ignored_alert_pressure,
        0.0,
        1.0,
    )

    return {
        "asset_exposure": float(asset_exposure),
        "compromised_hosts": float(compromised_hosts),
        "infected_hosts": float(infected_hosts),
        "patch_debt": float(patch_debt),
        "credential_exposure": float(credential_exposure),
        "monitoring_weakness": float(monitoring_weakness),
        "ransomware_spread": float(ransomware_spread),
        "segmentation_gap": float(segmentation_gap),
        "ignored_alert_pressure": float(ignored_alert_pressure),
        "network_risk": float(network_risk),
    }


def compute_network_risk(
    assets: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute normalized enterprise risk in [0, 1], lower is safer."""
    return compute_risk_breakdown(assets, context=context)["network_risk"]


def breach_rate(assets: List[Dict[str, Any]]) -> float:
    if not assets:
        return 0.0
    breached = sum(1 for a in assets if a["compromised"])
    return breached / len(assets)


def uptime_ratio(assets: List[Dict[str, Any]]) -> float:
    if not assets:
        return 1.0
    up = sum(1 for a in assets if a["uptime_status"])
    return up / len(assets)
