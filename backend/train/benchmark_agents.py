from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.heuristic_agent import HeuristicAgent
from backend.agents.ppo_agent import PPOAgent
from backend.agents.random_agent import RandomAgent
from backend.agents.router import AgentRouter
from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv

PPO_AGENT = "ppo_existing_checkpoint"
FRESH_PPO_AGENT = "ppo_fresh_checkpoint"
CORE_AGENTS = {"random", "heuristic", PPO_AGENT, FRESH_PPO_AGENT}
DEFAULT_AGENTS = ["random", "heuristic", PPO_AGENT]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_MATRIX = [
    {"scenario": "bank", "difficulty": "hard", "attacker": "ransomware_gang"},
    {"scenario": "saas", "difficulty": "hard", "attacker": "silent_apt"},
    {"scenario": "hospital", "difficulty": "medium", "attacker": "insider_saboteur"},
]
DEFAULT_PPO_PATH = PROJECT_ROOT / "backend/train/artifacts/best_model/best_model.zip"
DEFAULT_FRESH_PPO_PATH = PROJECT_ROOT / "backend/train/artifacts/p2_fresh_ppo/p2_fresh_primary_seed42/best_model/best_model.zip"

METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "total_reward": {"scope": "episode", "direction": "higher", "definition": "Sum of environment rewards across the episode."},
    "breach_rate": {"scope": "episode-final", "direction": "lower", "definition": "Final compromised assets divided by total assets."},
    "successful_attacks": {"scope": "episode", "direction": "lower", "definition": "Count of RED attempts whose red_log.success is true."},
    "prevented_attacks": {"scope": "episode", "direction": "higher", "definition": "Count of scheduled RED attempts that failed; no-attack turns are excluded."},
    "prevention_rate": {"scope": "episode", "direction": "higher", "definition": "Prevented attacks divided by scheduled attack attempts; zero when there were no attempts."},
    "compromised_assets": {"scope": "episode-final", "direction": "lower", "definition": "Count of compromised assets in the final state."},
    "critical_compromised_assets": {"scope": "episode-final", "direction": "lower", "definition": "Final compromised assets listed as critical by the active scenario."},
    "mean_network_risk": {"scope": "episode", "direction": "lower", "definition": "Arithmetic mean of post-transition network risk over all turns."},
    "final_network_risk": {"scope": "episode-final", "direction": "lower", "definition": "Network risk after the final transition."},
    "peak_network_risk": {"scope": "episode", "direction": "lower", "definition": "Maximum initial or post-transition network risk."},
    "mean_uptime": {"scope": "episode", "direction": "higher", "definition": "Mean per-turn fraction of assets with uptime_status=true."},
    "final_uptime": {"scope": "episode-final", "direction": "higher", "definition": "Final fraction of assets with uptime_status=true."},
    "mean_downtime": {"scope": "episode", "direction": "lower", "definition": "One minus mean uptime."},
    "action_cost": {"scope": "episode", "direction": "lower", "definition": "Sum of action_cost reported by reward_breakdown."},
    "substituted_actions": {"scope": "episode", "direction": "lower", "definition": "Requested actions replaced by a different executed action due to environment constraints."},
    "substitution_rate": {"scope": "episode", "direction": "lower", "definition": "Fraction of requested actions replaced by a different executed action."},
    "repeated_actions": {"scope": "episode", "direction": "lower", "definition": "Consecutive requested actions equal to the preceding requested action."},
    "repeated_action_rate": {"scope": "episode", "direction": "lower", "definition": "Fraction of decisions after the first that repeat the preceding requested action."},
    "survival_turns": {"scope": "episode", "direction": "higher", "definition": "Number of transitions before termination or truncation."},
    "decision_latency_ms": {"scope": "episode", "direction": "lower", "definition": "Mean wall-clock policy decision latency per turn in milliseconds."},
    "fallback_count": {"scope": "episode", "direction": "lower", "definition": "Decisions where a router reported fallback to another agent."},
}
AGGREGATE_METRICS = list(METRIC_DEFINITIONS)


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    scenario: str
    difficulty: str
    attacker: str
    seed: int


@dataclass
class EpisodeResult:
    agent: str
    episode_index: int
    case_id: str
    scenario: str
    difficulty: str
    attacker: str
    seed: int
    total_reward: float
    breach_rate: float
    successful_attacks: int
    prevented_attacks: int
    attempted_attacks: int
    prevention_rate: float
    compromised_assets: int
    critical_compromised_assets: int
    mean_network_risk: float
    final_network_risk: float
    peak_network_risk: float
    mean_uptime: float
    final_uptime: float
    mean_downtime: float
    action_cost: float
    substituted_actions: int
    substitution_rate: float
    repeated_actions: int
    repeated_action_rate: float
    survival_turns: int
    decision_latency_ms: float
    fallback_count: int
    termination_reason: str
    terminated: bool
    truncated: bool
    requested_actions: str
    executed_actions: str
    underlying_agents: str
    qwen_decisions: int = 0
    ordinary_ppo_decisions: int = 0
    qwen_failures: int = 0
    ppo_fallbacks: int = 0
    fallback_reasons: str = "[]"


@dataclass
class FailureResult:
    agent: str
    case_id: str
    seed: int
    error_type: str
    error: str
    attempt_utc: str = ""


def _parse_agents(raw: str) -> List[str]:
    aliases = {
        "random": "random",
        "heuristic": "heuristic",
        "ppo": PPO_AGENT,
        PPO_AGENT: PPO_AGENT,
        "fresh_ppo": FRESH_PPO_AGENT,
        FRESH_PPO_AGENT: FRESH_PPO_AGENT,
        "qwen": "qwen_rl",
        "qwen_rl": "qwen_rl",
        "hf": "qwen_rl",
        "hybrid": "hybrid_router",
        "hybrid_router": "hybrid_router",
    }
    agents: List[str] = []
    for token in raw.replace(",", " ").split():
        canonical = aliases.get(token.strip().lower())
        if canonical is None:
            raise ValueError(f"Unsupported agent: {token}")
        if canonical not in agents:
            agents.append(canonical)
    return agents or list(DEFAULT_AGENTS)


def _parse_seeds(raw: str) -> List[int]:
    seeds = [int(token) for token in raw.replace(",", " ").split()]
    if not seeds:
        raise ValueError("At least one benchmark seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Benchmark seeds must be unique.")
    return seeds


def build_experiment_plan(seeds: Sequence[int], matrix: Sequence[Dict[str, str]] = DEFAULT_MATRIX) -> List[ExperimentCase]:
    return [
        ExperimentCase(
            case_id=f"c{config_index:02d}_s{seed}",
            scenario=str(config["scenario"]),
            difficulty=str(config["difficulty"]),
            attacker=str(config["attacker"]),
            seed=int(seed),
        )
        for config_index, config in enumerate(matrix)
        for seed in seeds
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _uptime(assets: Sequence[Dict[str, Any]]) -> float:
    return sum(1 for asset in assets if bool(asset.get("uptime_status", False))) / max(len(assets), 1)


def _agent_state(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario": info.get("profile", {}).get("scenario", "legacy"),
        "attacker": info.get("profile", {}).get("attacker", "legacy_default"),
        "network_risk": info.get("network_risk", 0.0),
        "risk_breakdown": info.get("risk_breakdown", {}),
        "assets": info.get("assets", []),
        "alerts": info.get("alerts", []),
        "defender": info.get("defender", {}),
        "events": info.get("events", []),
        "intelligence": info.get("intelligence", {}),
    }


class PolicySet:
    def __init__(self, agents: Sequence[str], ppo_path: Path, fresh_ppo_path: Optional[Path] = None) -> None:
        self.random = RandomAgent()
        self.heuristic = HeuristicAgent()
        self.ppo = PPOAgent(str(ppo_path)) if PPO_AGENT in agents else None
        self.fresh_ppo = PPOAgent(str(fresh_ppo_path)) if FRESH_PPO_AGENT in agents and fresh_ppo_path is not None else None
        self.qwen: Optional[AgentRouter] = None
        self.hybrid: Optional[AgentRouter] = None
        if "qwen_rl" in agents:
            self.qwen = AgentRouter(config={"default_agent": "hf_llm_agent", "mode": "full_llm", "full_llm": True})
            if not self.qwen.switch_agent("hf_llm_agent"):
                raise RuntimeError("Qwen RL is unavailable; it cannot be included in the benchmark.")
        if "hybrid_router" in agents:
            self.hybrid = AgentRouter(config={"default_agent": "ppo_agent", "mode": "hybrid", "full_llm": False})
            if not self.hybrid.is_agent_available("ppo_agent") or not self.hybrid.is_agent_available("hf_llm_agent"):
                raise RuntimeError("Hybrid Router requires both PPO and Qwen to be available.")
            self.hybrid.switch_agent("hybrid")

    def reset_episode(self, agent: str, seed: int) -> None:
        if agent == "random":
            self.random.reset(seed)
        if agent == "heuristic":
            self.heuristic.reset()
        router = self.qwen if agent == "qwen_rl" else self.hybrid if agent == "hybrid_router" else None
        if router is not None:
            router.turn_counter = 0
            router.last_fallback_reason = None

    def decide(self, agent: str, env: CySentSecurityEnv, obs: np.ndarray, info: Dict[str, Any]) -> Tuple[int, str, Optional[str]]:
        if agent == "random":
            return int(self.random.predict_action()), "random", None
        if agent == "heuristic":
            return int(self.heuristic.predict_action(_agent_state(info))), "heuristic", None
        if agent == PPO_AGENT:
            if self.ppo is None:
                raise RuntimeError("Historical PPO checkpoint was not loaded.")
            return int(self.ppo.predict_action(obs, deterministic=True)), PPO_AGENT, None
        if agent == FRESH_PPO_AGENT:
            if self.fresh_ppo is None:
                raise RuntimeError("Fresh PPO checkpoint was not loaded.")
            return int(self.fresh_ppo.predict_action(obs, deterministic=True)), FRESH_PPO_AGENT, None
        router = self.qwen if agent == "qwen_rl" else self.hybrid if agent == "hybrid_router" else None
        if router is None:
            raise ValueError(f"Unsupported agent: {agent}")
        action = int(router.predict_action(obs, _agent_state(info)))
        return action, router.last_used_agent, router.last_fallback_reason


def run_episode(*, agent: str, episode_index: int, case: ExperimentCase, max_steps: int, policies: PolicySet) -> EpisodeResult:
    env = CySentSecurityEnv(max_steps=max_steps, seed=case.seed)
    obs, info = env.reset(
        seed=case.seed,
        options={
            "scenario": case.scenario,
            "difficulty": case.difficulty,
            "attacker": case.attacker,
            "strategy_mode": "balanced",
            "action_source": agent,
            "intelligence_enabled": False,
        },
    )
    policies.reset_episode(agent, case.seed)
    rewards: List[float] = []
    risks: List[float] = [float(info.get("network_risk", 0.0))]
    uptimes: List[float] = []
    latencies: List[float] = []
    requested: List[str] = []
    executed: List[str] = []
    underlying: List[str] = []
    fallback_reasons: List[str] = []
    attempted_attacks = 0
    action_cost = 0.0
    substitutions = repeats = fallbacks = 0
    terminated = truncated = False

    while not (terminated or truncated):
        start = time.perf_counter()
        action, underlying_agent, fallback_reason = policies.decide(agent, env, obs, info)
        latencies.append((time.perf_counter() - start) * 1000.0)
        if not env.action_space.contains(action):
            raise ValueError(f"Agent {agent} produced invalid action {action}.")
        requested_name = ACTION_NAMES[action]
        repeats += int(bool(requested) and requested[-1] == requested_name)
        requested.append(requested_name)
        underlying.append(underlying_agent)
        fallbacks += int(bool(fallback_reason))
        if fallback_reason:
            fallback_reasons.append(fallback_reason)

        obs, reward, terminated, truncated, info = env.step(action)
        executed_name = str(info.get("last_action", requested_name))
        executed.append(executed_name)
        substitutions += int(executed_name != requested_name)
        rewards.append(float(reward))
        risks.append(float(info.get("network_risk", 1.0)))
        uptimes.append(_uptime(info.get("assets", [])))
        action_cost += float(info.get("reward_breakdown", {}).get("action_cost", 0.0))
        attempted_attacks += int(str(info.get("red_log", {}).get("attack", "no_attack")) != "no_attack")

    assets = list(info.get("assets", []))
    metrics = info.get("metrics", {})
    critical_names = set(info.get("profile", {}).get("critical_assets", []))
    compromised = [asset for asset in assets if bool(asset.get("compromised", False))]
    prevented = int(metrics.get("prevented_attacks", 0))
    final_uptime = _uptime(assets)
    mean_uptime = float(np.mean(uptimes) if uptimes else final_uptime)
    return EpisodeResult(
        agent=agent,
        episode_index=episode_index,
        case_id=case.case_id,
        scenario=case.scenario,
        difficulty=case.difficulty,
        attacker=case.attacker,
        seed=case.seed,
        total_reward=float(sum(rewards)),
        breach_rate=float(len(compromised) / max(len(assets), 1)),
        successful_attacks=int(metrics.get("successful_attacks", 0)),
        prevented_attacks=prevented,
        attempted_attacks=attempted_attacks,
        prevention_rate=float(prevented / attempted_attacks) if attempted_attacks else 0.0,
        compromised_assets=len(compromised),
        critical_compromised_assets=sum(1 for asset in compromised if asset.get("name") in critical_names),
        mean_network_risk=float(np.mean(risks[1:]) if len(risks) > 1 else risks[0]),
        final_network_risk=float(risks[-1]),
        peak_network_risk=float(max(risks)),
        mean_uptime=mean_uptime,
        final_uptime=final_uptime,
        mean_downtime=float(1.0 - mean_uptime),
        action_cost=float(action_cost),
        substituted_actions=substitutions,
        substitution_rate=float(substitutions / len(rewards)) if rewards else 0.0,
        repeated_actions=repeats,
        repeated_action_rate=float(repeats / max(len(rewards) - 1, 1)) if rewards else 0.0,
        survival_turns=len(rewards),
        decision_latency_ms=float(np.mean(latencies) if latencies else 0.0),
        fallback_count=fallbacks,
        termination_reason=str(info.get("termination_reason", "active")) if terminated else "max_steps",
        terminated=bool(terminated),
        truncated=bool(truncated),
        requested_actions=json.dumps(requested, separators=(",", ":")),
        executed_actions=json.dumps(executed, separators=(",", ":")),
        underlying_agents=json.dumps(underlying, separators=(",", ":")),
        qwen_decisions=sum(source == "hf_llm_agent" for source in underlying),
        ordinary_ppo_decisions=max(sum(source == "ppo_agent" for source in underlying) - fallbacks, 0)
        if agent == "hybrid_router" else 0,
        qwen_failures=fallbacks if agent == "hybrid_router" else 0,
        ppo_fallbacks=fallbacks if agent == "hybrid_router" else 0,
        fallback_reasons=json.dumps(fallback_reasons, separators=(",", ":")),
    )


def aggregate_results(rows: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for agent in dict.fromkeys(row.agent for row in rows):
        agent_rows = [row for row in rows if row.agent == agent]
        summary: Dict[str, Any] = {"agent": agent, "sample_count": len(agent_rows)}
        for metric in AGGREGATE_METRICS:
            values = np.asarray([float(getattr(row, metric)) for row in agent_rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(summary)
    return summaries


def _action_distribution(rows: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for agent in dict.fromkeys(row.agent for row in rows):
        agent_rows = [row for row in rows if row.agent == agent]
        for action_kind, field in (("requested", "requested_actions"), ("executed", "executed_actions")):
            counts: Counter[str] = Counter()
            for row in agent_rows:
                counts.update(json.loads(getattr(row, field)))
            total = sum(counts.values())
            for action_id, action_name in ACTION_NAMES.items():
                output.append({
                    "agent": agent,
                    "action_kind": action_kind,
                    "action_id": action_id,
                    "action_name": action_name,
                    "count": counts[action_name],
                    "rate": counts[action_name] / total if total else 0.0,
                })
    return output


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, lambda handle: handle.write(value))


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2) + "\n")


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    materialized = list(rows)

    def write(handle: Any) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in materialized:
            csv_writer.writerow({name: row.get(name) for name in fieldnames})

    _atomic_write(path, write)


def _read_episode_rows(path: Path) -> List[EpisodeResult]:
    if not path.exists():
        return []
    integer_fields = {
        "episode_index", "seed", "successful_attacks", "prevented_attacks", "attempted_attacks",
        "compromised_assets", "critical_compromised_assets", "substituted_actions", "repeated_actions",
        "survival_turns", "fallback_count", "qwen_decisions", "ordinary_ppo_decisions", "qwen_failures",
        "ppo_fallbacks",
    }
    float_fields = {
        "total_reward", "breach_rate", "prevention_rate", "mean_network_risk", "final_network_risk",
        "peak_network_risk", "mean_uptime", "final_uptime", "mean_downtime", "action_cost",
        "substitution_rate", "repeated_action_rate", "decision_latency_ms",
    }
    boolean_fields = {"terminated", "truncated"}
    defaults = {name: field.default for name, field in EpisodeResult.__dataclass_fields__.items()}
    rows: List[EpisodeResult] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            values: Dict[str, Any] = {}
            for name in EpisodeResult.__dataclass_fields__:
                value: Any = raw.get(name, defaults.get(name))
                if name in integer_fields:
                    value = int(value)
                elif name in float_fields:
                    value = float(value)
                elif name in boolean_fields:
                    value = str(value).lower() == "true"
                values[name] = value
            rows.append(EpisodeResult(**values))
    return rows


def _read_failure_rows(path: Path) -> List[FailureResult]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [FailureResult(**{name: raw.get(name, "") for name in FailureResult.__dataclass_fields__}) for raw in csv.DictReader(handle)]


def _write_plots(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    agents = [str(row["agent"]) for row in summaries]
    specs = [
        ("total_reward", "Episode Reward", None),
        ("breach_rate", "Final Breach Rate", (0.0, 1.0)),
        ("mean_network_risk", "Mean Network Risk", (0.0, 1.0)),
        ("final_network_risk", "Final Network Risk", (0.0, 1.0)),
        ("mean_uptime", "Mean Uptime", (0.0, 1.0)),
        ("action_cost", "Episode Action Cost", None),
    ]
    colors = ["#2563eb", "#059669", "#b45309", "#7c3aed", "#be123c"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (metric, title, limits) in zip(axes.flat, specs):
        means = [float(row[f"{metric}_mean"]) for row in summaries]
        stds = [float(row[f"{metric}_std"]) for row in summaries]
        axis.bar(agents, means, yerr=stds, capsize=4, color=colors[: len(agents)])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        if limits:
            axis.set_ylim(*limits)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        fig.savefig(temp_path, dpi=150, format="png")
        os.replace(temp_path, path)
    finally:
        plt.close(fig)
        temp_path.unlink(missing_ok=True)


def _fmt(row: Dict[str, Any], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def aggregate_action_rates(rows: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for agent in dict.fromkeys(row.agent for row in rows):
        agent_rows = [row for row in rows if row.agent == agent]
        decisions = sum(row.survival_turns for row in agent_rows)
        repeat_opportunities = sum(max(row.survival_turns - 1, 0) for row in agent_rows)
        output.append({
            "agent": agent,
            "episode_mean_substitution_rate": float(np.mean([row.substitution_rate for row in agent_rows])),
            "pooled_substitution_rate": float(sum(row.substituted_actions for row in agent_rows) / max(decisions, 1)),
            "episode_mean_repeated_action_rate": float(np.mean([row.repeated_action_rate for row in agent_rows])),
            "pooled_repeated_action_rate": float(sum(row.repeated_actions for row in agent_rows) / max(repeat_opportunities, 1)),
            "decisions": decisions,
            "repeat_opportunities": repeat_opportunities,
        })
    return output


def _write_report(path: Path, metadata: Dict[str, Any], summaries: Sequence[Dict[str, Any]], failures: Sequence[FailureResult], rows: Sequence[EpisodeResult]) -> None:
    lines = [
        "# CySent P1 Baseline Benchmark", "",
        "This report contains only post-P0 executions from this output directory.", "",
        "## Experiment", "",
        f"- Run UTC: {metadata['run_utc']}",
        f"- Git commit: {metadata['git']['commit']}",
        f"- Dirty working tree: {metadata['git']['dirty']}",
        f"- Seeds: {', '.join(str(seed) for seed in metadata['seeds'])}",
        f"- Maximum episode length: {metadata['max_steps']}",
        f"- Raw completed episodes: {metadata['completed_episode_count']}", "",
        "## Aggregate Results", "",
        "| Agent | Reward | Breach | Mean Risk | Final Risk | Uptime | Action Cost | Survival | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['agent']} | {_fmt(row, 'total_reward')} | {_fmt(row, 'breach_rate')} | "
            f"{_fmt(row, 'mean_network_risk')} | {_fmt(row, 'final_network_risk')} | "
            f"{_fmt(row, 'mean_uptime')} | {_fmt(row, 'action_cost')} | "
            f"{_fmt(row, 'survival_turns')} | {row['sample_count']} |"
        )
    lines.extend([
        "", "## Action Rate Aggregation", "",
        "Episode mean gives every episode equal weight. Pooled rate gives every decision (or repeat opportunity) equal weight.", "",
        "| Agent | Episode-Mean Substitution | Pooled Substitution | Episode-Mean Repeat | Pooled Repeat | Decisions |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for rate in aggregate_action_rates(rows):
        lines.append(
            f"| {rate['agent']} | {rate['episode_mean_substitution_rate']:.4%} | "
            f"{rate['pooled_substitution_rate']:.4%} | "
            f"{rate['episode_mean_repeated_action_rate']:.4%} | "
            f"{rate['pooled_repeated_action_rate']:.4%} | {rate['decisions']} |"
        )
    lines.extend(["", "## Metric Definitions", ""])
    for name, definition in METRIC_DEFINITIONS.items():
        lines.append(f"- `{name}` ({definition['direction']} is better): {definition['definition']}")
    lines.extend(["", "## Reliability", ""])
    if failures:
        lines.extend(f"- {f.agent} / {f.case_id}: {f.error_type}: {f.error}" for f in failures)
    else:
        lines.append("- No agent failures or timeouts were recorded.")
    lines.extend([
        "", "## Limitations", "",
        "- The historical PPO checkpoint is local, uncommitted, and has incomplete exact training-source provenance.",
        "- Population standard deviation is reported over this fixed benchmark matrix; the sample is not a claim of broad external validity.",
        "- Wasteful-action count is not reported because the environment does not expose that reward-internal classification reliably.",
    ])
    _write_text(path, "\n".join(lines) + "\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_model_metadata(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    allowed = (
        "model_id", "requested_revision", "resolved_revision", "dtype", "quantization", "device",
        "gpu_name", "cuda_version", "python_version", "torch_version", "numpy_version",
        "transformers_version", "accelerate_version", "huggingface_hub_version", "stable_baselines3_version",
    )
    normalized = {name: raw[name] for name in allowed if name in raw}
    for name, value in normalized.items():
        if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
            raise ValueError(f"Model metadata field {name!r} is not JSON compatible.")
    revision = str(normalized.get("resolved_revision", ""))
    if revision and (len(revision) != 40 or any(character not in "0123456789abcdefABCDEF" for character in revision)):
        raise ValueError("resolved_revision must be an immutable 40-character Hugging Face commit SHA.")
    return normalized


def build_metadata(*, agents: Sequence[str], seeds: Sequence[int], plan: Sequence[ExperimentCase], ppo_path: Path, max_steps: int, completed_count: int, failure_count: int, fresh_ppo_path: Optional[Path] = None, model_metadata: Optional[Dict[str, Any]] = None, resumable: bool = False) -> Dict[str, Any]:
    dirty_output = _git("status", "--short")
    source_paths = [
        PROJECT_ROOT / "backend/env/security_env.py",
        PROJECT_ROOT / "backend/env/reward.py",
        PROJECT_ROOT / "backend/env/threat_engine.py",
        PROJECT_ROOT / "backend/agents/random_agent.py",
        PROJECT_ROOT / "backend/agents/heuristic_agent.py",
        PROJECT_ROOT / "backend/agents/hf_agent.py",
        PROJECT_ROOT / "backend/agents/router.py",
        PROJECT_ROOT / "backend/train/benchmark_agents.py",
        PROJECT_ROOT / "configs/v1_locked.yaml",
    ]
    return {
        "schema_version": 2 if resumable else 1,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(dirty_output and dirty_output != "unknown"),
            "status": dirty_output.splitlines() if dirty_output not in {"", "unknown"} else [],
        },
        "agents": list(agents),
        "seeds": list(seeds),
        "max_steps": max_steps,
        "matrix": [asdict(case) for case in plan],
        "expected_episode_count": len(agents) * len(plan),
        "completed_episode_count": completed_count,
        "failure_count": failure_count,
        "environment": {
            "observation_shape": [67],
            "action_count": len(ACTION_NAMES),
            "action_names": ACTION_NAMES,
            "source_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths},
        },
        "ppo": {
            "label": "Existing/Historical PPO Checkpoint",
            "agent_id": PPO_AGENT,
            "path": _display_path(ppo_path),
            "sha256": _sha256(ppo_path),
            "committed": bool(_git("ls-files", _display_path(ppo_path))),
            "vecnormalize_required": False,
            "provenance": "Incomplete; compatibility verified, exact P0-corrected training source not claimed.",
        },
        "fresh_ppo": {
            "label": "Fresh PPO Primary Best Checkpoint",
            "agent_id": FRESH_PPO_AGENT,
            "path": _display_path(fresh_ppo_path),
            "sha256": _sha256(fresh_ppo_path),
            "committed": bool(_git("ls-files", _display_path(fresh_ppo_path))),
        } if fresh_ppo_path is not None else None,
        "qwen_model": _normalize_model_metadata(model_metadata),
        "status": "incomplete" if resumable else (
            "complete" if completed_count == len(agents) * len(plan) else "failed"
        ),
        "metric_definitions": METRIC_DEFINITIONS,
        "aggregation": "Arithmetic mean and population standard deviation across completed episodes; no outlier removal.",
        "historical_results_included": False,
    }


def _resume_contract(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": metadata["schema_version"],
        "git_commit": metadata["git"]["commit"],
        "agents": metadata["agents"],
        "seeds": metadata["seeds"],
        "max_steps": metadata["max_steps"],
        "matrix": metadata["matrix"],
        "source_sha256": metadata["environment"]["source_sha256"],
        "ppo": {key: metadata["ppo"][key] for key in ("agent_id", "path", "sha256")},
        "fresh_ppo": metadata["fresh_ppo"],
        "qwen_model": metadata["qwen_model"],
    }


def _contract_digest(contract: Dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _episode_key(agent: str, case_id: str, seed: int) -> Tuple[str, str, int]:
    return agent, case_id, int(seed)


def _persist_progress(outdir: Path, rows: Sequence[EpisodeResult], failures: Sequence[FailureResult], metadata: Dict[str, Any]) -> None:
    completed = len(rows)
    expected = int(metadata["expected_episode_count"])
    metadata["completed_episode_count"] = completed
    metadata["failure_count"] = len(failures)
    metadata["status"] = "complete" if completed == expected else "incomplete"
    metadata["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _write_csv(outdir / "episodes.csv", (asdict(row) for row in rows), list(EpisodeResult.__dataclass_fields__))
    _write_csv(outdir / "failures.csv", (asdict(row) for row in failures), list(FailureResult.__dataclass_fields__))
    _write_json(outdir / "metadata.json", metadata)


def run_benchmark(*, agents: Sequence[str], seeds: Sequence[int], outdir: Path, max_steps: int = 150, matrix: Sequence[Dict[str, str]] = DEFAULT_MATRIX, ppo_path: Path = DEFAULT_PPO_PATH, fresh_ppo_path: Optional[Path] = None, resume: bool = False, model_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    agents = list(agents)
    seeds = list(seeds)
    plan = build_experiment_plan(seeds, matrix)
    if PPO_AGENT in agents and not ppo_path.exists():
        raise FileNotFoundError(f"Historical PPO checkpoint not found: {ppo_path}")
    if FRESH_PPO_AGENT in agents and (fresh_ppo_path is None or not fresh_ppo_path.exists()):
        raise FileNotFoundError(f"Fresh PPO checkpoint not found: {fresh_ppo_path}")
    qwen_requested = bool({"qwen_rl", "hybrid_router"}.intersection(agents))
    normalized_model_metadata = _normalize_model_metadata(model_metadata)
    if resume and qwen_requested:
        required = {"model_id", "resolved_revision", "dtype", "quantization", "device"}
        missing = sorted(required.difference(normalized_model_metadata or {}))
        if missing:
            raise ValueError(f"Resumable Qwen/Hybrid runs require model metadata fields: {', '.join(missing)}")

    outdir = outdir if outdir.is_absolute() else PROJECT_ROOT / outdir
    rows: List[EpisodeResult] = []
    failures: List[FailureResult] = []
    metadata: Optional[Dict[str, Any]] = None
    if resume:
        outdir.mkdir(parents=True, exist_ok=True)
        candidate = build_metadata(
            agents=agents,
            seeds=seeds,
            plan=plan,
            ppo_path=ppo_path,
            max_steps=max_steps,
            completed_count=0,
            failure_count=0,
            fresh_ppo_path=fresh_ppo_path if FRESH_PPO_AGENT in agents else None,
            model_metadata=normalized_model_metadata,
            resumable=True,
        )
        contract = _resume_contract(candidate)
        candidate["resume_contract"] = contract
        candidate["resume_contract_sha256"] = _contract_digest(contract)
        metadata_path = outdir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("resume_contract_sha256") != candidate["resume_contract_sha256"] or metadata.get("resume_contract") != contract:
                raise RuntimeError("Refusing incompatible resume: benchmark configuration, model revision, checkpoint, commit, or source hashes changed.")
            rows = _read_episode_rows(outdir / "episodes.csv")
            failures = _read_failure_rows(outdir / "failures.csv")
        else:
            existing = [path for path in outdir.iterdir()]
            if existing:
                raise FileExistsError(f"Refusing to overwrite non-resumable output directory: {outdir}")
            metadata = candidate
        completed_keys = [_episode_key(row.agent, row.case_id, row.seed) for row in rows]
        if len(completed_keys) != len(set(completed_keys)):
            raise RuntimeError("Refusing resume because episodes.csv contains duplicate completed episode keys.")
        _persist_progress(outdir, rows, failures, metadata)

    policies = PolicySet(agents, ppo_path, fresh_ppo_path)
    completed = {_episode_key(row.agent, row.case_id, row.seed) for row in rows}
    for agent in agents:
        for episode_index, case in enumerate(plan):
            key = _episode_key(agent, case.case_id, case.seed)
            if key in completed:
                continue
            try:
                row = run_episode(agent=agent, episode_index=episode_index, case=case, max_steps=max_steps, policies=policies)
                if _episode_key(row.agent, row.case_id, row.seed) != key:
                    raise RuntimeError("Episode result identity does not match the requested agent/case/seed key.")
                rows.append(row)
                completed.add(key)
            except Exception as exc:
                failures.append(FailureResult(
                    agent=agent,
                    case_id=case.case_id,
                    seed=case.seed,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    attempt_utc=datetime.now(timezone.utc).isoformat(),
                ))
            finally:
                if resume and metadata is not None:
                    _persist_progress(outdir, rows, failures, metadata)

    summaries = aggregate_results(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = build_metadata(
            agents=agents,
            seeds=seeds,
            plan=plan,
            ppo_path=ppo_path,
            max_steps=max_steps,
            completed_count=len(rows),
            failure_count=len(failures),
            fresh_ppo_path=fresh_ppo_path if FRESH_PPO_AGENT in agents else None,
            model_metadata=normalized_model_metadata,
        )
    episode_fields = list(EpisodeResult.__dataclass_fields__)
    summary_fields = ["agent", "sample_count"] + [f"{metric}_{suffix}" for metric in AGGREGATE_METRICS for suffix in ("mean", "std")]
    _write_csv(outdir / "episodes.csv", (asdict(row) for row in rows), episode_fields)
    _write_csv(outdir / "summary.csv", summaries, summary_fields)
    _write_csv(outdir / "action_distribution.csv", _action_distribution(rows), ["agent", "action_kind", "action_id", "action_name", "count", "rate"])
    _write_csv(outdir / "failures.csv", (asdict(row) for row in failures), list(FailureResult.__dataclass_fields__))
    _write_json(outdir / "metadata.json", metadata)
    _write_report(outdir / "report.md", metadata, summaries, failures, rows)
    if summaries:
        _write_plots(outdir / "comparison.png", summaries)
    return {
        "status": "ok" if len(rows) == len(agents) * len(plan) else "failed",
        "output_directory": str(outdir),
        "completed_episode_count": len(rows),
        "expected_episode_count": len(agents) * len(plan),
        "failure_count": len(failures),
        "summary": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible CySent P1 baseline benchmark.")
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--ppo-path", default=str(DEFAULT_PPO_PATH.relative_to(PROJECT_ROOT)))
    parser.add_argument("--fresh-ppo-path", default=str(DEFAULT_FRESH_PPO_PATH.relative_to(PROJECT_ROOT)))
    parser.add_argument("--outdir", default="outputs/benchmarks/p1_baseline_v2")
    parser.add_argument("--resume", action="store_true", help="Persist each episode atomically and resume a compatible run.")
    parser.add_argument("--model-metadata", help="JSON file with immutable local Qwen model/runtime identity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_metadata = None
    if args.model_metadata:
        model_metadata = json.loads(Path(args.model_metadata).read_text(encoding="utf-8"))
    result = run_benchmark(
        agents=_parse_agents(args.agents),
        seeds=_parse_seeds(args.seeds),
        outdir=Path(args.outdir),
        max_steps=int(args.max_steps),
        ppo_path=PROJECT_ROOT / args.ppo_path,
        fresh_ppo_path=PROJECT_ROOT / args.fresh_ppo_path,
        resume=bool(args.resume),
        model_metadata=model_metadata,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
