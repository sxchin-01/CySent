from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class EventRecord:
    turn: int
    actor: str
    event_type: str
    target: str
    success: bool
    details: Dict[str, Any]
    summary: str


def blue_action_summary(action_name: str, red_log: Dict[str, Any]) -> str:
    attack = str(red_log.get("attack", ""))
    mappings = {
        "do_nothing": "Blue held current posture and observed network telemetry.",
        "patch_hr_systems": "Blue patched HR Systems to reduce exploit exposure.",
        "patch_web_server": "Blue hardened the Web Server with priority patching.",
        "patch_auth_server": "Blue patched the Auth Server to close escalation paths.",
        "rotate_credentials": "Blue rotated credentials after suspicious authentication pressure.",
        "isolate_suspicious_host": "Blue isolated a suspicious host to contain potential spread.",
        "increase_monitoring": "Blue increased monitoring depth across security telemetry.",
        "restore_backup": "Blue restored from backup and attempted service recovery.",
        "deploy_honeypot": "Blue deployed a honeypot to divert hostile reconnaissance.",
        "phishing_training": "Blue rolled out phishing-awareness reinforcement.",
        "investigate_top_alert": "Blue investigated highest-priority alerts for active compromise.",
        "segment_finance_database": "Blue segmented Finance Database access to reduce blast radius.",
    }
    line = mappings.get(action_name, f"Blue executed {action_name}.")
    if attack in {"credential_theft", "password_spray", "phishing_email"} and action_name == "rotate_credentials":
        return "Blue rotated credentials after suspicious login surge."
    if attack == "lateral_movement" and action_name == "segment_finance_database":
        return "Blue segmentation blocked attempted lateral movement toward finance systems."
    return line


def red_attack_summary(red_log: Dict[str, Any]) -> str:
    attack = str(red_log.get("attack", "unknown_attack"))
    target = str(red_log.get("target", "unknown_target"))
    success = bool(red_log.get("success", False))

    phrase = {
        "phishing_email": f"Phishing wave targeted {target} users.",
        "password_spray": f"Password spray targeted identity surfaces on {target}.",
        "malware_dropper": f"Malware dropper attempted foothold deployment on {target}.",
        "credential_theft": f"Credential theft operations focused on {target}.",
        "privilege_escalation": f"Privilege escalation attempt ran against {target}.",
        "lateral_movement": f"Lateral movement attempt pivoted through {target}.",
        "ransomware_attempt": f"Ransomware detonation attempt hit {target}.",
        "data_exfiltration": f"Data exfiltration channel opened from {target}.",
        "insider_misuse": f"Insider misuse behavior was observed around {target}.",
        "recon_scan": f"Recon scan profiled exposed services on {target}.",
    }.get(attack, f"Red executed {attack} against {target}.")

    if success:
        return phrase
    return f"{phrase[:-1]} but defenses disrupted execution."


def build_turn_events(
    turn: int,
    action_name: str,
    red_log: Dict[str, Any],
    chain_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    events: List[EventRecord] = []

    events.append(
        EventRecord(
            turn=turn,
            actor="blue",
            event_type="defense_action",
            target=str(red_log.get("target", "network")),
            success=True,
            details={"action": action_name},
            summary=blue_action_summary(action_name, red_log),
        )
    )

    events.append(
        EventRecord(
            turn=turn,
            actor="red",
            event_type="attack_action",
            target=str(red_log.get("target", "network")),
            success=bool(red_log.get("success", False)),
            details={k: v for k, v in red_log.items() if k != "target"},
            summary=red_attack_summary(red_log),
        )
    )

    if chain_context:
        chain_id = str(chain_context.get("chain_id", ""))
        stage_name = str(chain_context.get("stage", ""))
        stage_idx = int(chain_context.get("stage_index", 0))
        stage_total = int(chain_context.get("stage_total", 0))
        events.append(
            EventRecord(
                turn=turn,
                actor="red",
                event_type="attack_chain",
                target=str(red_log.get("target", "network")),
                success=bool(chain_context.get("advance", False)),
                details={
                    "chain_id": chain_id,
                    "stage": stage_name,
                    "stage_index": stage_idx,
                    "stage_total": stage_total,
                    "complete": bool(chain_context.get("complete", False)),
                },
                summary=(
                    f"Attack chain {chain_id} progressed to stage {stage_idx + 1}/{max(stage_total, 1)}: {stage_name}."
                    if chain_id
                    else "Red attempted to progress a multi-stage campaign."
                ),
            )
        )

    return [asdict(e) for e in events]


def summarize_events(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "No major security events were recorded this turn."
    return " ".join(str(event.get("summary", "")).strip() for event in events if event.get("summary"))
