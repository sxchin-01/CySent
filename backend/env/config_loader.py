from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover - dependency provided in runtime env
    raise RuntimeError("PyYAML is required for CySent profile loading") from exc


BASE_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = BASE_DIR / "scenarios"
PROFILE_DIR = BASE_DIR / "profiles"


CANONICAL_ASSET_NAMES = [
    "HR Systems",
    "Employee Email",
    "Web Server",
    "Auth Server",
    "Finance Database",
    "Backup Infrastructure",
    "SOC Monitoring Console",
]


LEGACY_SCENARIO: Dict[str, Any] = {
    "name": "legacy",
    "description": "Legacy CySent baseline profile used for backward compatibility.",
    "detection_maturity": 0.55,
    "budget": "balanced",
    "recovery_speed": 0.55,
    "critical_assets": ["Finance Database", "Auth Server"],
    "attack_priorities": ["Finance Database", "Auth Server", "Employee Email", "Web Server"],
    "network_topology": [
        ["Employee Email", "HR Systems"],
        ["Employee Email", "Auth Server"],
        ["Web Server", "Auth Server"],
        ["Auth Server", "Finance Database"],
        ["Backup Infrastructure", "Finance Database"],
        ["SOC Monitoring Console", "Auth Server"],
    ],
    "asset_profiles": {
        "HR Systems": {
            "patch_level": [0.45, 0.80],
            "detection_level": [0.35, 0.65],
            "credential_risk": [0.25, 0.50],
            "backup_status": [0.70, 0.95],
            "criticality": 0.55,
            "business_dependency": 0.62,
        },
        "Employee Email": {
            "patch_level": [0.48, 0.78],
            "detection_level": [0.34, 0.62],
            "credential_risk": [0.30, 0.58],
            "backup_status": [0.65, 0.90],
            "criticality": 0.65,
            "business_dependency": 0.70,
        },
        "Web Server": {
            "patch_level": [0.44, 0.75],
            "detection_level": [0.36, 0.64],
            "credential_risk": [0.24, 0.45],
            "backup_status": [0.66, 0.90],
            "criticality": 0.78,
            "business_dependency": 0.80,
        },
        "Auth Server": {
            "patch_level": [0.42, 0.72],
            "detection_level": [0.40, 0.70],
            "credential_risk": [0.22, 0.50],
            "backup_status": [0.70, 0.92],
            "criticality": 0.88,
            "business_dependency": 0.90,
        },
        "Finance Database": {
            "patch_level": [0.45, 0.74],
            "detection_level": [0.42, 0.72],
            "credential_risk": [0.20, 0.44],
            "backup_status": [0.76, 0.95],
            "criticality": 1.0,
            "business_dependency": 0.95,
        },
        "Backup Infrastructure": {
            "patch_level": [0.50, 0.82],
            "detection_level": [0.30, 0.58],
            "credential_risk": [0.18, 0.40],
            "backup_status": [0.82, 0.98],
            "criticality": 0.82,
            "business_dependency": 0.86,
        },
        "SOC Monitoring Console": {
            "patch_level": [0.55, 0.84],
            "detection_level": [0.55, 0.84],
            "credential_risk": [0.16, 0.35],
            "backup_status": [0.68, 0.90],
            "criticality": 0.72,
            "business_dependency": 0.75,
        },
    },
}


DIFFICULTY_DEFAULTS: Dict[str, Dict[str, float]] = {
    "easy": {
        "attack_frequency": 0.72,
        "exploit_success_multiplier": 0.88,
        "stealth_probability": 0.25,
        "escalation_speed": 0.78,
        "zero_day_chance": 0.03,
        "insider_event_chance": 0.02,
    },
    "medium": {
        "attack_frequency": 0.84,
        "exploit_success_multiplier": 1.0,
        "stealth_probability": 0.35,
        "escalation_speed": 1.0,
        "zero_day_chance": 0.06,
        "insider_event_chance": 0.04,
    },
    "hard": {
        "attack_frequency": 0.95,
        "exploit_success_multiplier": 1.12,
        "stealth_probability": 0.48,
        "escalation_speed": 1.18,
        "zero_day_chance": 0.10,
        "insider_event_chance": 0.07,
    },
}


ATTACKER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "legacy_default": {
        "description": "Balanced legacy attacker profile approximating original CySent behavior.",
        "attack_bias": {
            "phishing_email": 0.11,
            "password_spray": 0.10,
            "malware_dropper": 0.10,
            "credential_theft": 0.11,
            "privilege_escalation": 0.09,
            "lateral_movement": 0.09,
            "ransomware_attempt": 0.09,
            "data_exfiltration": 0.09,
            "insider_misuse": 0.08,
            "recon_scan": 0.10,
        },
        "chain_bias": 0.55,
        "stealth_modifier": 1.0,
        "target_criticality_bias": 1.0,
        "target_backup_bias": 1.0,
        "chain_templates": [
            ["phishing_email", "credential_theft", "lateral_movement"],
            ["recon_scan", "malware_dropper", "privilege_escalation"],
        ],
    },
    "ransomware_gang": {
        "description": "Aggressive disruption actor focused on encryption and downtime.",
        "attack_bias": {
            "lateral_movement": 0.22,
            "ransomware_attempt": 0.28,
            "credential_theft": 0.16,
            "data_exfiltration": 0.10,
            "recon_scan": 0.08,
        },
        "chain_bias": 0.85,
        "stealth_modifier": 0.90,
        "target_criticality_bias": 1.20,
        "target_backup_bias": 1.35,
        "chain_templates": [
            ["phishing_email", "credential_theft", "lateral_movement", "ransomware_attempt"],
            ["recon_scan", "malware_dropper", "lateral_movement", "ransomware_attempt"],
        ],
    },
    "credential_thief": {
        "description": "Identity-centric attacker abusing authentication controls.",
        "attack_bias": {
            "phishing_email": 0.26,
            "password_spray": 0.24,
            "credential_theft": 0.27,
            "privilege_escalation": 0.12,
            "insider_misuse": 0.06,
        },
        "chain_bias": 0.80,
        "stealth_modifier": 1.05,
        "target_criticality_bias": 1.05,
        "target_backup_bias": 0.95,
        "chain_templates": [
            ["phishing_email", "credential_theft", "privilege_escalation", "lateral_movement"],
            ["password_spray", "credential_theft", "data_exfiltration"],
        ],
    },
    "silent_apt": {
        "description": "Stealth-focused actor prioritizing persistence and exfiltration.",
        "attack_bias": {
            "recon_scan": 0.20,
            "malware_dropper": 0.12,
            "privilege_escalation": 0.18,
            "lateral_movement": 0.16,
            "data_exfiltration": 0.24,
        },
        "chain_bias": 0.92,
        "stealth_modifier": 1.20,
        "target_criticality_bias": 1.15,
        "target_backup_bias": 0.90,
        "chain_templates": [
            ["recon_scan", "malware_dropper", "privilege_escalation", "data_exfiltration"],
            ["recon_scan", "credential_theft", "lateral_movement", "data_exfiltration"],
        ],
    },
    "insider_saboteur": {
        "description": "Abusive insider actor with privileged misuse potential.",
        "attack_bias": {
            "insider_misuse": 0.30,
            "credential_theft": 0.20,
            "privilege_escalation": 0.16,
            "ransomware_attempt": 0.14,
            "data_exfiltration": 0.14,
        },
        "chain_bias": 0.78,
        "stealth_modifier": 1.10,
        "target_criticality_bias": 1.25,
        "target_backup_bias": 1.05,
        "chain_templates": [
            ["insider_misuse", "credential_theft", "privilege_escalation", "ransomware_attempt"],
            ["insider_misuse", "lateral_movement", "data_exfiltration"],
        ],
    },
    "botnet": {
        "description": "Noisy distributed attacker focused on broad compromise attempts.",
        "attack_bias": {
            "recon_scan": 0.18,
            "phishing_email": 0.14,
            "malware_dropper": 0.22,
            "password_spray": 0.18,
            "lateral_movement": 0.14,
        },
        "chain_bias": 0.72,
        "stealth_modifier": 0.82,
        "target_criticality_bias": 1.00,
        "target_backup_bias": 1.08,
        "chain_templates": [
            ["recon_scan", "malware_dropper", "lateral_movement"],
            ["password_spray", "credential_theft", "malware_dropper"],
        ],
    },
}


@dataclass(frozen=True)
class RuntimeProfiles:
    scenario: Dict[str, Any]
    difficulty: Dict[str, Any]
    attacker: Dict[str, Any]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_range(raw: Any, default_lo: float, default_hi: float) -> Tuple[float, float]:
    if isinstance(raw, list) and len(raw) == 2:
        lo = float(raw[0])
        hi = float(raw[1])
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        return _clamp01(lo), _clamp01(hi)
    if isinstance(raw, (int, float)):
        val = _clamp01(float(raw))
        return val, val
    return _clamp01(default_lo), _clamp01(default_hi)


def _merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_scenario_profile(name: str) -> Dict[str, Any]:
    scenario_name = (name or "legacy").strip().lower()
    profile = LEGACY_SCENARIO

    if scenario_name != "legacy":
        loaded = _read_yaml(SCENARIO_DIR / f"{scenario_name}.yaml")
        if loaded:
            profile = _merge_dict(LEGACY_SCENARIO, loaded)
        else:
            profile = LEGACY_SCENARIO

    normalized_asset_profiles: Dict[str, Dict[str, Any]] = {}
    raw_assets = profile.get("asset_profiles", {}) if isinstance(profile.get("asset_profiles", {}), dict) else {}
    for asset_name in CANONICAL_ASSET_NAMES:
        defaults = LEGACY_SCENARIO["asset_profiles"][asset_name]
        src = raw_assets.get(asset_name, defaults)
        criticality = _clamp01(float(src.get("criticality", defaults["criticality"])))
        business_dependency = _clamp01(float(src.get("business_dependency", defaults["business_dependency"])))
        downtime_cost = _clamp01(float(src.get("downtime_cost", 0.45 + 0.55 * business_dependency)))
        patch_speed = _clamp01(float(src.get("patch_speed", 0.35 + 0.55 * (1.0 - criticality))))
        exposure = _clamp01(float(src.get("exposure", 0.30 + 0.60 * (1.0 - defaults["patch_level"][0]))))
        business_value = _clamp01(float(src.get("business_value", business_dependency)))
        normalized_asset_profiles[asset_name] = {
            "patch_level": _coerce_range(src.get("patch_level"), defaults["patch_level"][0], defaults["patch_level"][1]),
            "detection_level": _coerce_range(src.get("detection_level"), defaults["detection_level"][0], defaults["detection_level"][1]),
            "credential_risk": _coerce_range(src.get("credential_risk"), defaults["credential_risk"][0], defaults["credential_risk"][1]),
            "backup_status": _coerce_range(src.get("backup_status"), defaults["backup_status"][0], defaults["backup_status"][1]),
            "criticality": criticality,
            "business_dependency": business_dependency,
            "downtime_cost": downtime_cost,
            "patch_speed": patch_speed,
            "exposure": exposure,
            "business_value": business_value,
        }

    detection_maturity = _clamp01(float(profile.get("detection_maturity", LEGACY_SCENARIO["detection_maturity"])))
    recovery_speed = _clamp01(float(profile.get("recovery_speed", LEGACY_SCENARIO["recovery_speed"])))

    return {
        "name": str(profile.get("name", scenario_name or "legacy")),
        "description": str(profile.get("description", "")),
        "budget": str(profile.get("budget", LEGACY_SCENARIO["budget"])),
        "detection_maturity": detection_maturity,
        "recovery_speed": recovery_speed,
        "critical_assets": list(profile.get("critical_assets", LEGACY_SCENARIO["critical_assets"])),
        "attack_priorities": list(profile.get("attack_priorities", LEGACY_SCENARIO["attack_priorities"])),
        "network_topology": list(profile.get("network_topology", LEGACY_SCENARIO["network_topology"])),
        "asset_inventory": list(profile.get("asset_inventory", CANONICAL_ASSET_NAMES)),
        "asset_profiles": normalized_asset_profiles,
    }


def load_difficulty_profile(name: str) -> Dict[str, Any]:
    base = _read_yaml(PROFILE_DIR / "difficulty.yaml")
    difficulty_name = (name or "medium").strip().lower()
    raw_levels = base.get("levels", {}) if isinstance(base.get("levels", {}), dict) else {}

    level = raw_levels.get(difficulty_name, DIFFICULTY_DEFAULTS.get(difficulty_name))
    if level is None:
        difficulty_name = "medium"
        level = raw_levels.get("medium", DIFFICULTY_DEFAULTS["medium"])

    merged = _merge_dict(DIFFICULTY_DEFAULTS.get(difficulty_name, DIFFICULTY_DEFAULTS["medium"]), level)
    return {
        "name": difficulty_name,
        "attack_frequency": _clamp01(float(merged.get("attack_frequency", 0.84))),
        "exploit_success_multiplier": float(merged.get("exploit_success_multiplier", 1.0)),
        "stealth_probability": _clamp01(float(merged.get("stealth_probability", 0.35))),
        "escalation_speed": max(0.1, float(merged.get("escalation_speed", 1.0))),
        "zero_day_chance": _clamp01(float(merged.get("zero_day_chance", 0.06))),
        "insider_event_chance": _clamp01(float(merged.get("insider_event_chance", 0.04))),
    }


def load_attacker_profile(name: str) -> Dict[str, Any]:
    base = _read_yaml(PROFILE_DIR / "attackers.yaml")
    attacker_name = (name or "legacy_default").strip().lower()
    raw_attackers = base.get("attackers", {}) if isinstance(base.get("attackers", {}), dict) else {}

    profile = raw_attackers.get(attacker_name, ATTACKER_DEFAULTS.get(attacker_name))
    if profile is None:
        attacker_name = "legacy_default"
        profile = raw_attackers.get(attacker_name, ATTACKER_DEFAULTS[attacker_name])

    merged = _merge_dict(ATTACKER_DEFAULTS.get(attacker_name, ATTACKER_DEFAULTS["credential_thief"]), profile)
    attack_bias = merged.get("attack_bias", {}) if isinstance(merged.get("attack_bias", {}), dict) else {}

    normalized_bias: Dict[str, float] = {}
    for key, value in attack_bias.items():
        normalized_bias[str(key)] = max(0.0, float(value))

    return {
        "name": attacker_name,
        "description": str(merged.get("description", "")),
        "goal": str(merged.get("goal", "")).strip().lower(),
        "attack_bias": normalized_bias,
        "chain_bias": _clamp01(float(merged.get("chain_bias", 0.8))),
        "stealth_modifier": max(0.5, float(merged.get("stealth_modifier", 1.0))),
        "target_criticality_bias": max(0.5, float(merged.get("target_criticality_bias", 1.0))),
        "target_backup_bias": max(0.5, float(merged.get("target_backup_bias", 1.0))),
        "chain_templates": [
            [str(step) for step in chain]
            for chain in merged.get("chain_templates", [])
            if isinstance(chain, list) and chain
        ],
    }


def load_runtime_profiles(
    scenario: str,
    difficulty: str,
    attacker: str,
) -> RuntimeProfiles:
    return RuntimeProfiles(
        scenario=load_scenario_profile(scenario),
        difficulty=load_difficulty_profile(difficulty),
        attacker=load_attacker_profile(attacker),
    )
