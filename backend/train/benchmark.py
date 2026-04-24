from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from backend.agents.router import AgentRouter
from backend.env.risk import breach_rate, uptime_ratio
from backend.env.security_env import CySentSecurityEnv
from backend.train.evaluate import evaluate


def _normalize_agent_name(agent: str) -> str:
    normalized = agent.strip().lower()
    if normalized == "hf_llm":
        return "hf_llm_agent"
    return normalized


def _load_experiment_registry(artifacts_root: Path) -> List[Dict[str, Any]]:
    path = artifacts_root / "experiment_runs.json"
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    return []


def _discover_latest_run_dirs(artifacts_root: Path, latest: int) -> List[str]:
    if latest <= 0:
        return []

    rows = _load_experiment_registry(artifacts_root)
    out: List[str] = []
    seen: set[str] = set()
    for row in reversed(rows):
        run_dir = str(row.get("run_dir", "")).strip()
        if not run_dir or run_dir in seen:
            continue
        if Path(run_dir).exists():
            out.append(run_dir)
            seen.add(run_dir)
        if len(out) >= latest:
            break
    return list(reversed(out))


def _derive_model_path(run_dir: str) -> str:
    return str(Path(run_dir) / "model.zip")


def _build_specs(
    run_dirs: List[str],
    labels: List[str],
    latest: int,
    artifacts_root: Path,
    baseline_model: str,
    tuned_model: str,
    cloud_model: str,
) -> List[Dict[str, str]]:
    resolved_run_dirs = list(run_dirs)
    if not resolved_run_dirs:
        resolved_run_dirs = _discover_latest_run_dirs(artifacts_root=artifacts_root, latest=latest)

    specs: List[Dict[str, str]] = []
    for idx, run_dir in enumerate(resolved_run_dirs):
        label = labels[idx] if idx < len(labels) else Path(run_dir).name
        specs.append(
            {
                "label": label,
                "run_dir": run_dir,
                "model_path": _derive_model_path(run_dir),
            }
        )

    # Backward-compatible model-path mode when run directories are not provided.
    if not specs:
        legacy = [
            ("ppo_baseline", baseline_model),
            ("ppo_tuned", tuned_model),
            ("cloud_trained", cloud_model),
        ]
        for label, model_path in legacy:
            if not model_path:
                continue
            specs.append(
                {
                    "label": label,
                    "run_dir": str(Path(model_path).parent),
                    "model_path": model_path,
                }
            )

    return specs


def _evaluate_spec(spec: Dict[str, str], episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    model_path = spec["model_path"]
    run_dir = spec.get("run_dir")
    label = spec["label"]

    if not Path(model_path).exists():
        return {
            "label": label,
            "status": "missing_model",
            "model_path": model_path,
            "run_dir": run_dir,
        }

    try:
        summary = evaluate(
            model_path=model_path,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            run_dir=run_dir,
        )
    except Exception as exc:
        return {
            "label": label,
            "status": "eval_failed",
            "error": str(exc),
            "model_path": model_path,
            "run_dir": run_dir,
        }

    trained = summary["trained_policy"]
    random_row = summary["random_policy"]
    validation = summary["validation"]
    score_row = summary["security_score"]

    return {
        "label": label,
        "status": "ok",
        "run_name": str(validation.get("run_name") or label),
        "seed": int(validation.get("seed") or seed),
        "run_dir": str(validation.get("run_dir") or run_dir or ""),
        "model_path": model_path,
        "reward": float(trained.get("avg_episode_reward", np.nan)),
        "breach": float(trained.get("avg_breach_rate", np.nan)),
        "uptime": float(trained.get("avg_uptime", np.nan)),
        "risk": float(trained.get("avg_final_risk", np.nan)),
        "score": float(score_row.get("trained", np.nan)),
        "random_reward": float(random_row.get("avg_episode_reward", np.nan)),
        "random_breach": float(random_row.get("avg_breach_rate", np.nan)),
        "random_uptime": float(random_row.get("avg_uptime", np.nan)),
        "random_risk": float(random_row.get("avg_final_risk", np.nan)),
        "random_score": float(score_row.get("random", np.nan)),
        "delta_vs_random": float(score_row.get("delta", np.nan)),
    }


def _select_baseline(rows: List[Dict[str, Any]], baseline_ref: str) -> Dict[str, Any] | None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return None

    if baseline_ref:
        for row in ok_rows:
            if row.get("label") == baseline_ref or row.get("run_name") == baseline_ref:
                return row

    return ok_rows[0]


def _apply_decisions(
    rows: List[Dict[str, Any]],
    baseline_ref: str,
    freeze_min_delta: float,
    promote_min_delta: float,
    max_breach_regression: float,
    max_risk_regression: float,
) -> Dict[str, Any]:
    baseline = _select_baseline(rows, baseline_ref=baseline_ref)

    if baseline is None:
        for row in rows:
            row["decision"] = "discard"
            row["decision_reason"] = "No successful run available for baseline selection."
        return {"baseline": None, "rows": rows}

    baseline_score = float(baseline["score"])
    baseline_breach = float(baseline["breach"])
    baseline_risk = float(baseline["risk"])

    for row in rows:
        if row.get("status") != "ok":
            row["decision"] = "discard"
            row["decision_reason"] = f"Run status is {row.get('status')}"
            continue

        if row is baseline:
            row["decision"] = "freeze_baseline"
            if float(row["delta_vs_random"]) >= freeze_min_delta:
                row["decision_reason"] = (
                    f"Baseline beats random by {row['delta_vs_random']:.2f} score points (>= {freeze_min_delta:.2f})."
                )
            else:
                row["decision_reason"] = (
                    f"Baseline advantage over random is {row['delta_vs_random']:.2f} (< {freeze_min_delta:.2f}); keep but review."
                )
            continue

        score_gain = float(row["score"]) - baseline_score
        breach_regression = float(row["breach"]) - baseline_breach
        risk_regression = float(row["risk"]) - baseline_risk

        if (
            score_gain >= promote_min_delta
            and breach_regression <= max_breach_regression
            and risk_regression <= max_risk_regression
        ):
            row["decision"] = "promote"
            row["decision_reason"] = (
                f"Score +{score_gain:.2f} vs baseline with acceptable breach/risk drift "
                f"({breach_regression:+.4f}, {risk_regression:+.4f})."
            )
        else:
            row["decision"] = "discard"
            row["decision_reason"] = (
                f"Insufficient gain or safety regression: score {score_gain:+.2f}, "
                f"breach {breach_regression:+.4f}, risk {risk_regression:+.4f}."
            )

    rows.sort(key=lambda r: float(r.get("score", -1e9)) if r.get("status") == "ok" else -1e9, reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_by_score"] = idx

    return {
        "baseline": {
            "label": baseline.get("label"),
            "run_name": baseline.get("run_name"),
            "score": baseline_score,
            "breach": baseline_breach,
            "risk": baseline_risk,
        },
        "rows": rows,
    }


def _write_table(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    table_json = output_dir / "benchmark_table.json"
    table_csv = output_dir / "benchmark_table.csv"
    table_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "rank_by_score",
        "label",
        "run_name",
        "seed",
        "reward",
        "breach",
        "uptime",
        "risk",
        "score",
        "delta_vs_random",
        "decision",
        "decision_reason",
        "status",
        "run_dir",
        "model_path",
    ]
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _bar_chart(path: Path, labels: List[str], values: List[float], title: str, y_lim: tuple[float, float] | None = None) -> None:
    fig = plt.figure(figsize=(9, 4.8))
    x = np.arange(len(labels))
    plt.bar(x, values, color=["#991b1b", "#0f766e", "#166534", "#0369a1"])
    plt.xticks(x, labels, rotation=15, ha="right")
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_benchmark(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return

    labels = [str(r.get("run_name") or r.get("label")) for r in ok_rows]
    reward_vals = [float(r.get("reward", np.nan)) for r in ok_rows]
    breach_vals = [float(r.get("breach", np.nan)) for r in ok_rows]
    uptime_vals = [float(r.get("uptime", np.nan)) for r in ok_rows]
    risk_vals = [float(r.get("risk", np.nan)) for r in ok_rows]
    score_vals = [float(r.get("score", np.nan)) for r in ok_rows]

    _bar_chart(output_dir / "benchmark_reward.png", labels, reward_vals, "Avg Episode Reward")
    _bar_chart(output_dir / "benchmark_breach.png", labels, breach_vals, "Avg Breach Rate", y_lim=(0.0, 1.0))
    _bar_chart(output_dir / "benchmark_uptime.png", labels, uptime_vals, "Avg Uptime", y_lim=(0.0, 1.0))
    _bar_chart(output_dir / "benchmark_risk.png", labels, risk_vals, "Avg Final Risk", y_lim=(0.0, 1.0))
    _bar_chart(output_dir / "benchmark_score.png", labels, score_vals, "Security Score", y_lim=(0.0, 100.0))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _security_score(reward: float, breach: float, uptime: float, risk: float, survival: float, max_steps: int) -> float:
    breach_score = 100.0 * (1.0 - _clamp(breach, 0.0, 1.0))
    uptime_score = 100.0 * _clamp(uptime, 0.0, 1.0)
    risk_score = 100.0 * (1.0 - _clamp(risk, 0.0, 1.0))
    survival_score = 100.0 * _clamp(survival / max(float(max_steps), 1.0), 0.0, 1.0)
    reward_score = 100.0 * _clamp((reward + 50.0) / 200.0, 0.0, 1.0)
    return float(
        0.35 * breach_score
        + 0.25 * uptime_score
        + 0.20 * risk_score
        + 0.10 * survival_score
        + 0.10 * reward_score
    )


def _stress_suite(stress: str) -> List[Dict[str, str]]:
    base = [
        {"scenario": "bank", "difficulty": "hard", "attacker": "ransomware_gang"},
        {"scenario": "saas", "difficulty": "hard", "attacker": "silent_apt"},
        {"scenario": "hospital", "difficulty": "medium", "attacker": "insider_saboteur"},
    ]
    if stress == "all-hard":
        return [
            {"scenario": "bank", "difficulty": "hard", "attacker": "ransomware_gang"},
            {"scenario": "saas", "difficulty": "hard", "attacker": "silent_apt"},
            {"scenario": "hospital", "difficulty": "hard", "attacker": "insider_saboteur"},
            {"scenario": "government", "difficulty": "hard", "attacker": "credential_thief"},
            {"scenario": "manufacturing", "difficulty": "hard", "attacker": "botnet"},
        ]
    return base


def _run_agent_episode(
    env: CySentSecurityEnv,
    *,
    agent: str,
    model: PPO | None,
    router: AgentRouter | None,
    max_steps: int,
    seed: int,
    options: Dict[str, Any],
) -> Dict[str, float]:
    obs, info = env.reset(seed=seed, options=options)
    done = False
    trunc = False
    turns = 0
    rewards: List[float] = []

    while not (done or trunc):
        if agent == "random":
            action = env.action_space.sample()
        elif agent == "ppo":
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)
        else:
            assert router is not None
            state = {
                "scenario": options.get("scenario"),
                "attacker": options.get("attacker"),
                "network_risk": info.get("network_risk", 0.0),
                "risk_breakdown": info.get("risk_breakdown", {}),
                "assets": info.get("assets", []),
                "events": info.get("events", []),
                "intelligence": info.get("intelligence", {}),
            }
            action = router.predict_action(obs, state)

        obs, reward, done, trunc, info = env.step(int(action))
        turns += 1
        rewards.append(float(reward))
        if turns >= max_steps:
            break

    assets = info.get("assets", [])
    episode_reward = float(np.sum(rewards))
    breach = float(breach_rate(assets)) if assets else 1.0
    uptime = float(uptime_ratio(assets)) if assets else 0.0
    risk = float(info.get("network_risk", 1.0))
    score = _security_score(episode_reward, breach, uptime, risk, float(turns), max_steps=max_steps)

    return {
        "reward": episode_reward,
        "breach": breach,
        "uptime": uptime,
        "risk": risk,
        "score": score,
        "survival_turns": float(turns),
    }


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
        "var": float(arr.var()) if arr.size else float("nan"),
    }


def _plot_agent_benchmark(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return

    labels = [str(r["agent"]) for r in ok_rows]
    scores = [float(r.get("security_score", 0.0)) for r in ok_rows]
    risks = [float(r.get("risk", 1.0)) for r in ok_rows]
    breaches = [float(r.get("breach", 1.0)) for r in ok_rows]

    _bar_chart(output_dir / "leaderboard_bars.png", labels, scores, "Leaderboard Security Score", y_lim=(0.0, 100.0))
    _bar_chart(output_dir / "risk_trend.png", labels, risks, "Risk Trend (Lower is Better)", y_lim=(0.0, 1.0))
    _bar_chart(output_dir / "breach_compare.png", labels, breaches, "Breach Compare (Lower is Better)", y_lim=(0.0, 1.0))


def _write_agent_tables(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    table_json = output_dir / "benchmark_table.json"
    table_csv = output_dir / "benchmark_table.csv"
    table_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "rank",
        "agent",
        "reward",
        "breach",
        "uptime",
        "risk",
        "security_score",
        "reward_std",
        "reward_var",
        "breach_std",
        "breach_var",
        "uptime_std",
        "uptime_var",
        "risk_std",
        "risk_var",
        "security_score_std",
        "security_score_var",
        "decision",
        "decision_reason",
        "status",
        "seeds",
        "runs",
    ]
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _build_agent_benchmark(
    *,
    agents: List[str],
    seeds: int,
    episodes: int,
    max_steps: int,
    seed: int,
    stress: str,
    output_path: Path,
    tuned_model: str,
) -> Dict[str, Any]:
    requested_agents = [_normalize_agent_name(a) for a in agents if a.strip()]
    if not requested_agents:
        requested_agents = ["ppo", "hf_llm_agent", "random"]

    scenarios = _stress_suite(stress)
    seed_schedule = [seed + i for i in range(seeds)]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ppo_model: PPO | None = None
    if "ppo" in requested_agents and Path(tuned_model).exists():
        ppo_model = PPO.load(tuned_model)

    router = AgentRouter() if "hf_llm_agent" in requested_agents else None
    hf_available = bool(router and router.is_agent_available("hf_llm_agent"))
    if router and hf_available:
        router.switch_agent("hf_llm_agent")

    rows: List[Dict[str, Any]] = []
    for agent in requested_agents:
        if agent == "ppo" and ppo_model is None:
            rows.append({"agent": "ppo", "status": "missing_model", "decision": "discard", "decision_reason": f"Missing model at {tuned_model}"})
            continue
        if agent == "hf_llm_agent" and not hf_available:
            rows.append({"agent": "hf_llm_agent", "status": "unavailable", "decision": "discard", "decision_reason": "HF adapter/model unavailable; skipped gracefully."})
            continue

        reward_vals: List[float] = []
        breach_vals: List[float] = []
        uptime_vals: List[float] = []
        risk_vals: List[float] = []
        score_vals: List[float] = []

        for scenario_cfg in scenarios:
            env = CySentSecurityEnv(max_steps=max_steps, seed=seed)
            for s in seed_schedule:
                for ep in range(episodes):
                    episode_seed = s * 1000 + ep
                    result = _run_agent_episode(
                        env,
                        agent=agent,
                        model=ppo_model,
                        router=router,
                        max_steps=max_steps,
                        seed=episode_seed,
                        options={
                            "scenario": scenario_cfg["scenario"],
                            "difficulty": scenario_cfg["difficulty"],
                            "attacker": scenario_cfg["attacker"],
                            "action_source": "hf_llm_agent" if agent == "hf_llm_agent" else "ppo_agent",
                            "intelligence_enabled": True,
                        },
                    )
                    reward_vals.append(result["reward"])
                    breach_vals.append(result["breach"])
                    uptime_vals.append(result["uptime"])
                    risk_vals.append(result["risk"])
                    score_vals.append(result["score"])

        reward_stats = _stats(reward_vals)
        breach_stats = _stats(breach_vals)
        uptime_stats = _stats(uptime_vals)
        risk_stats = _stats(risk_vals)
        score_stats = _stats(score_vals)

        rows.append(
            {
                "agent": agent,
                "status": "ok",
                "reward": reward_stats["mean"],
                "breach": breach_stats["mean"],
                "uptime": uptime_stats["mean"],
                "risk": risk_stats["mean"],
                "security_score": score_stats["mean"],
                "reward_std": reward_stats["std"],
                "reward_var": reward_stats["var"],
                "breach_std": breach_stats["std"],
                "breach_var": breach_stats["var"],
                "uptime_std": uptime_stats["std"],
                "uptime_var": uptime_stats["var"],
                "risk_std": risk_stats["std"],
                "risk_var": risk_stats["var"],
                "security_score_std": score_stats["std"],
                "security_score_var": score_stats["var"],
                "seeds": seeds,
                "runs": len(score_vals),
            }
        )

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ppo_row = next((r for r in ok_rows if r.get("agent") == "ppo"), None)
    for row in rows:
        if row.get("status") != "ok":
            row["decision"] = row.get("decision", "discard")
            row["decision_reason"] = row.get("decision_reason", f"Agent status={row.get('status')}")
            continue
        if ppo_row is None or row is ppo_row:
            row["decision"] = "hold"
            row["decision_reason"] = "Reference PPO baseline."
            continue
        score_gain = float(row["security_score"]) - float(ppo_row["security_score"])
        breach_reg = float(row["breach"]) - float(ppo_row["breach"])
        risk_reg = float(row["risk"]) - float(ppo_row["risk"])
        if score_gain >= 1.5 and breach_reg <= 0.01 and risk_reg <= 0.02:
            row["decision"] = "promote"
            row["decision_reason"] = f"Score +{score_gain:.2f}, no safety regression (breach {breach_reg:+.4f}, risk {risk_reg:+.4f})."
        elif score_gain >= 0.0 and breach_reg <= 0.02 and risk_reg <= 0.03:
            row["decision"] = "hold"
            row["decision_reason"] = f"Comparable score with acceptable safety drift (breach {breach_reg:+.4f}, risk {risk_reg:+.4f})."
        else:
            row["decision"] = "discard"
            row["decision_reason"] = f"Insufficient gain or safety regression (score {score_gain:+.2f}, breach {breach_reg:+.4f}, risk {risk_reg:+.4f})."

    rows.sort(key=lambda r: float(r.get("security_score", -1e9)) if r.get("status") == "ok" else -1e9, reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    _write_agent_tables(rows, output_path.parent)
    _plot_agent_benchmark(rows, output_path.parent)

    summary = {
        "config": {
            "agents": requested_agents,
            "seeds": seeds,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "stress": stress,
            "scenarios": scenarios,
            "fair_compare": {
                "same_seed_schedule": True,
                "same_scenarios": True,
                "same_max_steps": True,
                "same_episode_count": True,
            },
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_benchmark(
    episodes: int,
    max_steps: int,
    seed: int,
    output: str = "backend/train/artifacts/benchmark/benchmark_summary.json",
    run_dirs: List[str] | None = None,
    labels: List[str] | None = None,
    latest: int = 0,
    baseline: str = "",
    freeze_min_delta: float = 3.0,
    promote_min_delta: float = 1.5,
    max_breach_regression: float = 0.01,
    max_risk_regression: float = 0.02,
    baseline_model: str = "backend/train/artifacts/cysent_ppo.zip",
    tuned_model: str = "backend/train/artifacts/best_model/best_model.zip",
    cloud_model: str = "",
    agents: List[str] | None = None,
    seeds: int = 20,
    stress: str = "default",
) -> Dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = output_path.parent.parent if output_path.parent.name == "benchmark" else output_path.parent

    run_dirs = run_dirs or []
    labels = labels or []
    agents = agents or []

    # Agent leaderboard mode (Step6) when explicitly requested.
    if agents:
        return _build_agent_benchmark(
            agents=agents,
            seeds=seeds,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            stress=stress,
            output_path=output_path,
            tuned_model=tuned_model,
        )

    specs = _build_specs(
        run_dirs=run_dirs,
        labels=labels,
        latest=latest,
        artifacts_root=artifacts_root,
        baseline_model=baseline_model,
        tuned_model=tuned_model,
        cloud_model=cloud_model,
    )

    rows = [_evaluate_spec(spec=s, episodes=episodes, max_steps=max_steps, seed=seed) for s in specs]
    decisions = _apply_decisions(
        rows=rows,
        baseline_ref=baseline,
        freeze_min_delta=freeze_min_delta,
        promote_min_delta=promote_min_delta,
        max_breach_regression=max_breach_regression,
        max_risk_regression=max_risk_regression,
    )

    result = {
        "config": {
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "baseline": baseline,
            "freeze_min_delta": freeze_min_delta,
            "promote_min_delta": promote_min_delta,
            "max_breach_regression": max_breach_regression,
            "max_risk_regression": max_risk_regression,
        },
        "baseline": decisions["baseline"],
        "rows": decisions["rows"],
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_table(decisions["rows"], output_path.parent)
    _plot_benchmark(decisions["rows"], output_path.parent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reproducible CySent runs")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=str, default="", help="Comma-separated agent list: ppo,hf_llm_agent,random")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seed runs per agent/scenario in agent mode")
    parser.add_argument("--stress", type=str, default="default", choices=["default", "all-hard"], help="Stress preset suite")
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory containing model.zip and config.json")
    parser.add_argument("--label", action="append", default=[], help="Optional label for each --run-dir")
    parser.add_argument("--latest", type=int, default=0, help="Auto-discover N latest run dirs from experiment_runs.json")
    parser.add_argument("--baseline", type=str, default="", help="Baseline label/run_name for promotion decisions")
    parser.add_argument("--freeze-min-delta", type=float, default=3.0)
    parser.add_argument("--promote-min-delta", type=float, default=1.5)
    parser.add_argument("--max-breach-regression", type=float, default=0.01)
    parser.add_argument("--max-risk-regression", type=float, default=0.02)
    # Backward-compatible legacy model path options.
    parser.add_argument("--baseline-model", type=str, default="backend/train/artifacts/cysent_ppo.zip")
    parser.add_argument("--tuned-model", type=str, default="backend/train/artifacts/best_model/best_model.zip")
    parser.add_argument("--cloud-model", type=str, default="")
    parser.add_argument("--output", type=str, default="backend/train/artifacts/benchmark/benchmark_summary.json")
    args = parser.parse_args()

    raw_agents = args.agents.replace(",", " ")
    agents = [a.strip() for a in raw_agents.split() if a.strip()]

    summary = build_benchmark(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
        run_dirs=args.run_dir,
        labels=args.label,
        latest=args.latest,
        baseline=args.baseline,
        freeze_min_delta=args.freeze_min_delta,
        promote_min_delta=args.promote_min_delta,
        max_breach_regression=args.max_breach_regression,
        max_risk_regression=args.max_risk_regression,
        baseline_model=args.baseline_model,
        tuned_model=args.tuned_model,
        cloud_model=args.cloud_model,
        agents=agents,
        seeds=args.seeds,
        stress=args.stress,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
