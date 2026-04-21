from __future__ import annotations

from typing import Any, Dict, List


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(clean.values())
    if total <= 0.0:
        n = max(len(clean), 1)
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def _asset_means(assets: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(len(assets), 1)
    return {
        "patch_debt": sum(1.0 - float(a.get("patch_level", 1.0)) for a in assets) / n,
        "credential_exposure": sum(float(a.get("credential_risk", 0.0)) for a in assets) / n,
        "monitoring_weakness": sum(1.0 - float(a.get("detection_level", 1.0)) for a in assets) / n,
        "backup_weakness": sum(1.0 - float(a.get("backup_status", a.get("backup_health", 1.0))) for a in assets) / n,
        "compromised_ratio": sum(1 for a in assets if bool(a.get("compromised", False))) / n,
        "infected_ratio": sum(1 for a in assets if bool(a.get("infected", False))) / n,
    }


def forecast_threats(
    *,
    assets: List[Dict[str, Any]],
    attacker_profile: Dict[str, Any],
    recent_red_logs: List[Dict[str, Any]],
    current_red_log: Dict[str, Any],
) -> Dict[str, Any]:
    means = _asset_means(assets)
    bias = attacker_profile.get("attack_bias", {}) if isinstance(attacker_profile.get("attack_bias"), dict) else {}

    scores: Dict[str, float] = {
        "phishing_email": 0.10 + means["credential_exposure"] * 0.45,
        "password_spray": 0.09 + means["credential_exposure"] * 0.50,
        "malware_dropper": 0.08 + means["patch_debt"] * 0.45,
        "credential_theft": 0.10 + means["credential_exposure"] * 0.55,
        "privilege_escalation": 0.08 + means["patch_debt"] * 0.50,
        "lateral_movement": 0.08 + means["infected_ratio"] * 0.65 + means["compromised_ratio"] * 0.40,
        "ransomware_attempt": 0.07 + means["backup_weakness"] * 0.60 + means["infected_ratio"] * 0.35,
        "data_exfiltration": 0.07 + means["compromised_ratio"] * 0.65,
        "insider_misuse": 0.05 + means["credential_exposure"] * 0.20,
        "recon_scan": 0.08 + means["monitoring_weakness"] * 0.45,
    }

    for attack, w in bias.items():
        if attack in scores:
            scores[attack] += float(w)

    # Chain-aware next stage boost.
    chain = current_red_log.get("chain", {}) if isinstance(current_red_log.get("chain", {}), dict) else {}
    if chain:
        stage_idx = int(chain.get("stage_index", 0))
        stage_total = int(chain.get("stage_total", 0))
        templates = attacker_profile.get("chain_templates", []) if isinstance(attacker_profile.get("chain_templates", []), list) else []
        for template in templates:
            if not isinstance(template, list) or not template:
                continue
            if stage_idx + 1 < len(template):
                nxt = str(template[stage_idx + 1])
                if nxt in scores:
                    scores[nxt] += 0.22

    # Momentum from recent successful attacks.
    for red in recent_red_logs[-5:]:
        if bool(red.get("success", False)):
            atk = str(red.get("attack", ""))
            if atk in scores:
                scores[atk] += 0.08

    probs = _normalize(scores)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    primary = ranked[0]
    secondary = ranked[1] if len(ranked) > 1 else ranked[0]

    return {
        "primary": {"attack": primary[0], "probability": float(primary[1])},
        "secondary": {"attack": secondary[0], "probability": float(secondary[1])},
        "top_predictions": [
            {"attack": k, "probability": float(v)} for k, v in ranked[:5]
        ],
        "signal_snapshot": {
            "patch_debt": float(means["patch_debt"]),
            "credential_exposure": float(means["credential_exposure"]),
            "monitoring_weakness": float(means["monitoring_weakness"]),
            "backup_weakness": float(means["backup_weakness"]),
            "infected_ratio": float(means["infected_ratio"]),
            "compromised_ratio": float(means["compromised_ratio"]),
        },
    }
