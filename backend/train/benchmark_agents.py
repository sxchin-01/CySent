from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Allow `python backend/train/benchmark_agents.py ...` from repository root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.random_agent import RandomAgent
from backend.agents.router import AgentRouter
from backend.env.security_env import CySentSecurityEnv


CRITICAL_FAILURE_REASONS = {"critical_breach", "downtime_cascade"}


def _load_local_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries from .env into process env when missing.

    HF-related keys are always refreshed from .env to match production precedence.
    """
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        if key.startswith("HF_") or key.startswith("HUGGINGFACE"):
            os.environ[key] = value
            continue

        if os.getenv(key) is None:
            os.environ[key] = value


def _parse_agents(raw: str) -> List[str]:
    aliases = {
        "hf": "hf_llm_agent",
        "hf_llm": "hf_llm_agent",
        "hf_llm_agent": "hf_llm_agent",
        "ppo": "ppo",
        "random": "random",
    }
    out: List[str] = []
    for token in raw.replace(",", " ").split():
        key = aliases.get(token.strip().lower())
        if key and key not in out:
            out.append(key)
    return out or ["ppo", "hf_llm_agent", "random"]


def _available_scenarios() -> List[str]:
    scenarios_dir = PROJECT_ROOT / "backend/env/scenarios"
    if not scenarios_dir.exists():
        return ["legacy"]

    names = sorted(p.stem for p in scenarios_dir.glob("*.yaml"))
    return names or ["legacy"]


def _parse_scenarios(raw: str) -> List[str]:
    available = _available_scenarios()
    if raw.strip().lower() == "all":
        return available

    requested = [s.strip().lower() for s in raw.replace(",", " ").split() if s.strip()]
    if not requested:
        return available

    valid = [s for s in requested if s in set(available)]
    if not valid:
        raise ValueError(f"No valid scenarios in '{raw}'. Available: {', '.join(available)}")
    return valid


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _safe_num(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


@dataclass
class EpisodeResult:
    agent: str
    episode_index: int
    scenario: str
    seed: int
    total_reward: float
    win: int
    success: int
    episode_length: int
    avg_network_risk: float
    final_network_risk: float
    critical_failure: int
    termination_reason: str
    stabilization_step: Optional[int]
    avg_action_latency_ms: float


def _first_stabilization_step(risks: Sequence[float], threshold: float, window: int) -> Optional[int]:
    if len(risks) < window:
        return None

    for idx in range(0, len(risks) - window + 1):
        chunk = risks[idx : idx + window]
        if all(float(r) <= threshold for r in chunk):
            return idx + 1  # 1-based step index
    return None


def _select_action(
    agent: str,
    obs: np.ndarray,
    info: Dict[str, Any],
    ppo_router: AgentRouter,
    hf_router: AgentRouter,
    random_agent: RandomAgent,
) -> Tuple[int, float]:
    state = {
        "scenario": info.get("profile", {}).get("scenario", "legacy"),
        "attacker": info.get("profile", {}).get("attacker", "legacy_default"),
        "network_risk": info.get("network_risk", 0.0),
        "risk_breakdown": info.get("risk_breakdown", {}),
        "assets": info.get("assets", []),
        "events": info.get("events", []),
        "intelligence": info.get("intelligence", {}),
    }

    start = time.perf_counter()
    if agent == "random":
        action = int(random_agent.predict_action())
    elif agent == "ppo":
        action = int(ppo_router.predict_action(obs, state))
    elif agent == "hf_llm_agent":
        action = int(hf_router.predict_action(obs, state))
    else:
        raise ValueError(f"Unsupported agent: {agent}")
    latency_ms = (time.perf_counter() - start) * 1000.0
    return action, float(latency_ms)


def _run_episode(
    env: CySentSecurityEnv,
    *,
    agent: str,
    episode_index: int,
    scenario: str,
    seed: int,
    max_steps: int,
    ppo_router: AgentRouter,
    hf_router: AgentRouter,
    random_agent: RandomAgent,
    stabilize_threshold: float,
    stabilize_window: int,
) -> EpisodeResult:
    obs, info = env.reset(
        seed=seed,
        options={
            "scenario": scenario,
            "action_source": "hf_llm_agent" if agent == "hf_llm_agent" else "ppo_agent",
            "intelligence_enabled": True,
        },
    )

    done = False
    trunc = False
    step_count = 0
    reward_sum = 0.0
    risks: List[float] = []
    latencies_ms: List[float] = []
    termination_reason = "max_steps"

    while not (done or trunc):
        action, latency_ms = _select_action(
            agent=agent,
            obs=obs,
            info=info,
            ppo_router=ppo_router,
            hf_router=hf_router,
            random_agent=random_agent,
        )
        latencies_ms.append(latency_ms)

        obs, reward, done, trunc, info = env.step(action)
        step_count += 1
        reward_sum += float(reward)
        risks.append(float(info.get("network_risk", 1.0)))

        if done:
            termination_reason = str(info.get("termination_reason", "terminated"))
        elif trunc:
            termination_reason = "max_steps"

        if step_count >= max_steps:
            break

    stabilization_step = _first_stabilization_step(
        risks=risks,
        threshold=stabilize_threshold,
        window=stabilize_window,
    )

    critical_failure = int(termination_reason in CRITICAL_FAILURE_REASONS)
    # Win/success definition: no critical failure and achieved stabilization within episode.
    success = int(critical_failure == 0 and stabilization_step is not None)

    return EpisodeResult(
        agent=agent,
        episode_index=episode_index,
        scenario=scenario,
        seed=seed,
        total_reward=float(reward_sum),
        win=success,
        success=success,
        episode_length=int(step_count),
        avg_network_risk=float(np.mean(risks) if risks else 1.0),
        final_network_risk=float(risks[-1] if risks else 1.0),
        critical_failure=critical_failure,
        termination_reason=termination_reason,
        stabilization_step=stabilization_step,
        avg_action_latency_ms=float(np.mean(latencies_ms) if latencies_ms else 0.0),
    )


def _aggregate(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    rewards = [r.total_reward for r in results]
    wins = [float(r.win) for r in results]
    lengths = [float(r.episode_length) for r in results]
    avg_risks = [r.avg_network_risk for r in results]
    final_risks = [r.final_network_risk for r in results]
    failures = [float(r.critical_failure) for r in results]
    latencies = [r.avg_action_latency_ms for r in results]
    stabilize_steps = [float(r.stabilization_step) for r in results if r.stabilization_step is not None]

    return {
        "episodes": len(results),
        "average_total_reward": _safe_mean(rewards),
        "success_win_rate": _safe_mean(wins),
        "average_episode_length": _safe_mean(lengths),
        "average_network_risk": _safe_mean(avg_risks),
        "average_final_network_risk": _safe_mean(final_risks),
        "critical_failures_count": int(np.sum(np.asarray(failures, dtype=np.float64))),
        "critical_failure_rate": _safe_mean(failures),
        "mean_time_to_stabilize": _safe_mean(stabilize_steps),
        "stabilized_episode_count": int(len(stabilize_steps)),
        "average_action_latency_ms": _safe_mean(latencies),
        "std_total_reward": _safe_std(rewards),
        "std_network_risk": _safe_std(avg_risks),
    }


def _write_results_csv(path: Path, rows: Sequence[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent",
        "episode_index",
        "scenario",
        "seed",
        "total_reward",
        "win",
        "success",
        "episode_length",
        "avg_network_risk",
        "final_network_risk",
        "critical_failure",
        "termination_reason",
        "stabilization_step",
        "avg_action_latency_ms",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "agent": row.agent,
                    "episode_index": row.episode_index,
                    "scenario": row.scenario,
                    "seed": row.seed,
                    "total_reward": row.total_reward,
                    "win": row.win,
                    "success": row.success,
                    "episode_length": row.episode_length,
                    "avg_network_risk": row.avg_network_risk,
                    "final_network_risk": row.final_network_risk,
                    "critical_failure": row.critical_failure,
                    "termination_reason": row.termination_reason,
                    "stabilization_step": row.stabilization_step if row.stabilization_step is not None else "",
                    "avg_action_latency_ms": row.avg_action_latency_ms,
                }
            )


def _write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_report_md(path: Path, summary: Dict[str, Any], selected_hf_path: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# CySent Benchmark Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    cfg = summary.get("config", {})
    lines.append(f"- Episodes: {cfg.get('episodes')}")
    lines.append(f"- Base Seed: {cfg.get('seed')}")
    lines.append(f"- Scenarios: {', '.join(cfg.get('scenarios', []))}")
    lines.append(f"- Agents: {', '.join(cfg.get('agents', []))}")
    lines.append("")
    lines.append("## HF Model Path Selection")
    lines.append("")
    lines.append(f"- Selected backend: {selected_hf_path.get('backend')}")
    lines.append(f"- Endpoint URL: {selected_hf_path.get('endpoint_url') or 'n/a'}")
    lines.append(f"- Merged model id: {selected_hf_path.get('merged_model_id') or 'n/a'}")
    lines.append(f"- Model id: {selected_hf_path.get('model_id') or 'n/a'}")
    lines.append(f"- Adapter path/id: {selected_hf_path.get('adapter_path') or 'n/a'}")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Agent | Avg Reward | Win Rate | Avg Ep Length | Avg Risk | Critical Failures | Mean Stabilize Step | Avg Action Latency (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    by_agent = summary.get("by_agent", {})
    for agent in summary.get("config", {}).get("agents", []):
        row = by_agent.get(agent, {})
        lines.append(
            "| "
            f"{agent} | "
            f"{_safe_num(row.get('average_total_reward'))} | "
            f"{_safe_num(row.get('success_win_rate'))} | "
            f"{_safe_num(row.get('average_episode_length'))} | "
            f"{_safe_num(row.get('average_network_risk'))} | "
            f"{row.get('critical_failures_count', 'n/a')} | "
            f"{_safe_num(row.get('mean_time_to_stabilize'))} | "
            f"{_safe_num(row.get('average_action_latency_ms'))} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    agents = summary.get("config", {}).get("agents", [])
    by_agent = summary.get("by_agent", {})
    rewards = [float(by_agent.get(a, {}).get("average_total_reward") or 0.0) for a in agents]
    wins = [float(by_agent.get(a, {}).get("success_win_rate") or 0.0) for a in agents]

    x = np.arange(len(agents))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, rewards, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(agents)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agents, rotation=10)
    axes[0].set_title("Average Total Reward")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, wins, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(agents)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agents, rotation=10)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Success / Win Rate")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _extract_hf_selection(router: AgentRouter) -> Dict[str, Any]:
    hf = router.hf_agent
    if hf is None:
        return {
            "backend": "unavailable",
            "endpoint_url": None,
            "merged_model_id": None,
            "model_id": None,
            "adapter_path": None,
        }

    return {
        "backend": getattr(hf, "_active_backend", "unknown"),
        "endpoint_url": getattr(hf, "endpoint_url", None),
        "merged_model_id": getattr(hf, "merged_model_id", None),
        "model_id": getattr(hf, "model_id", None),
        "adapter_path": getattr(hf, "adapter_path", None),
    }


def run_benchmark(
    *,
    episodes: int,
    seed: int,
    scenarios: List[str],
    agents: List[str],
    outdir: Path,
    max_steps: int,
    stabilize_threshold: float,
    stabilize_window: int,
) -> Dict[str, Any]:
    _load_local_env_file(path=str(PROJECT_ROOT / ".env"))

    ppo_router = AgentRouter(config={"default_agent": "ppo_agent", "mode": "ppo_only", "full_llm": False})
    hf_router = AgentRouter(config={"default_agent": "hf_llm_agent", "mode": "full_llm", "full_llm": True})
    random_agent = RandomAgent()

    # Enforce requested agent availability early.
    if "ppo" in agents and not ppo_router.is_agent_available("ppo_agent"):
        raise RuntimeError("PPO agent is unavailable. Ensure PPO model exists at configured production path.")
    hf_requested_but_unavailable = "hf_llm_agent" in agents and not hf_router.is_agent_available("hf_llm_agent")

    env = CySentSecurityEnv(max_steps=max_steps, seed=seed)

    # Same seeds/scenario schedule for all agents.
    episode_plan: List[Tuple[int, str, int]] = []
    for ep_idx in range(episodes):
        scenario = scenarios[ep_idx % len(scenarios)]
        ep_seed = int(seed + ep_idx)
        episode_plan.append((ep_idx, scenario, ep_seed))

    raw_rows: List[EpisodeResult] = []
    by_agent: Dict[str, Dict[str, Any]] = {}

    for agent in agents:
        if agent == "hf_llm_agent" and hf_requested_but_unavailable:
            by_agent[agent] = {
                "status": "failed",
                "error": "HF LLM agent is unavailable in production loading path (router/hf_agent).",
                "episodes": 0,
                "episodes_requested": len(episode_plan),
            }
            continue

        agent_rows: List[EpisodeResult] = []
        agent_error: Optional[str] = None
        try:
            for ep_idx, scenario, ep_seed in episode_plan:
                row = _run_episode(
                    env=env,
                    agent=agent,
                    episode_index=ep_idx,
                    scenario=scenario,
                    seed=ep_seed,
                    max_steps=max_steps,
                    ppo_router=ppo_router,
                    hf_router=hf_router,
                    random_agent=random_agent,
                    stabilize_threshold=stabilize_threshold,
                    stabilize_window=stabilize_window,
                )
                raw_rows.append(row)
                agent_rows.append(row)
        except Exception as exc:
            agent_error = f"{type(exc).__name__}: {exc}"

        if agent_error is not None:
            by_agent[agent] = {
                "status": "failed",
                "error": agent_error,
                "episodes": len(agent_rows),
                "episodes_requested": len(episode_plan),
            }
        else:
            stats = _aggregate(agent_rows)
            stats["status"] = "ok"
            by_agent[agent] = stats

    hf_selected = _extract_hf_selection(hf_router)

    summary = {
        "config": {
            "episodes": episodes,
            "seed": seed,
            "scenarios": scenarios,
            "agents": agents,
            "max_steps": max_steps,
            "stabilize_threshold": stabilize_threshold,
            "stabilize_window": stabilize_window,
            "fairness": {
                "identical_episode_plan_per_agent": True,
                "episode_plan": [
                    {"episode_index": ep_idx, "scenario": scenario, "seed": ep_seed}
                    for ep_idx, scenario, ep_seed in episode_plan
                ],
            },
        },
        "hf_model_path_selection": hf_selected,
        "by_agent": by_agent,
    }

    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir

    outdir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(outdir / "benchmark_results.csv", raw_rows)
    _write_summary_json(outdir / "benchmark_summary.json", summary)
    _write_report_md(outdir / "benchmark_report.md", summary, hf_selected)
    _write_plot(outdir / "benchmark_plot.png", summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PPO vs HF vs Random in CySent environment")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (quick=10, standard=50, full=100)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--scenarios", type=str, default="all", help="all|bank|government|hospital|manufacturing|saas")
    parser.add_argument("--outdir", type=str, default="outputs/benchmarks", help="Output directory")
    parser.add_argument("--agents", type=str, default="ppo,hf,random", help="Comma-separated agents: ppo,hf,random")
    parser.add_argument("--max-steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--stabilize-threshold", type=float, default=0.35, help="Risk threshold for stabilization")
    parser.add_argument("--stabilize-window", type=int, default=3, help="Consecutive low-risk window size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = _parse_agents(args.agents)
    scenarios = _parse_scenarios(args.scenarios)

    summary = run_benchmark(
        episodes=int(args.episodes),
        seed=int(args.seed),
        scenarios=scenarios,
        agents=agents,
        outdir=Path(args.outdir),
        max_steps=int(args.max_steps),
        stabilize_threshold=float(args.stabilize_threshold),
        stabilize_window=int(args.stabilize_window),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
