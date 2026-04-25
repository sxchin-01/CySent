"""Live HF LLM RL training against CySent environment.

REINFORCE with baseline, using LoRA on a small instruct model.
Runs the real env step loop: observe → prompt LLM → parse action → env.step → reward.
Designed for minimal HF credit burn on a single GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv

VALID_ACTIONS = {name: idx for idx, name in ACTION_NAMES.items()}
ACTION_LIST = ", ".join(VALID_ACTIONS.keys())


def _ensure_dynamic_cache_compat() -> None:
    """Backfill DynamicCache.from_legacy_cache for older transformers builds."""
    if hasattr(DynamicCache, "from_legacy_cache"):
        return

    @classmethod
    def _from_legacy_cache(cls, past_key_values: Any) -> Any:
        cache = cls()
        if past_key_values is None:
            return cache
        try:
            for layer_idx, layer_past in enumerate(past_key_values):
                if not isinstance(layer_past, (tuple, list)) or len(layer_past) < 2:
                    continue
                key_states, value_states = layer_past[0], layer_past[1]
                cache.update(key_states, value_states, layer_idx)
        except Exception:
            # Fallback to empty cache; generation will continue without legacy state.
            return cls()
        return cache

    setattr(DynamicCache, "from_legacy_cache", _from_legacy_cache)


def _build_prompt(info: Dict[str, Any], scenario: str, attacker: str) -> str:
    risk = float(info.get("network_risk", 0.0))
    assets = info.get("assets", [])
    compromised = [a["name"] for a in assets if a.get("compromised")]
    infected = [a["name"] for a in assets if a.get("infected")]
    rb = info.get("risk_breakdown", {})

    return (
        f"You are CySent BLUE defender on a {scenario} network vs {attacker}.\n"
        f"Risk: {risk:.2f} | Compromised: {compromised or 'None'} | Infected: {infected or 'None'}\n"
        f"Credential exposure: {float(rb.get('credential_exposure', 0)):.2f} | "
        f"Patch debt: {float(rb.get('patch_debt', 0)):.2f}\n"
        f"Choose exactly one action: {ACTION_LIST}\n"
        f"Action:"
    )


def _parse_action(text: str) -> int:
    text_lower = text.lower().replace("-", "_").replace(" ", "_")
    for name, idx in VALID_ACTIONS.items():
        if name in text_lower:
            return idx
    return 0


def _generate_action(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    temperature: float = 0.7,
) -> Tuple[int, torch.Tensor]:
    """Generate one action token sequence and return (action_id, log_prob)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad():
        pass

    outputs = model.generate(
        **inputs,
        max_new_tokens=12,
        do_sample=True,
        temperature=max(temperature, 0.1),
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    scores = outputs.scores

    log_probs = []
    for step_idx, token_id in enumerate(generated_ids):
        if step_idx >= len(scores):
            break
        logits = scores[step_idx][0]
        lp = torch.log_softmax(logits, dim=-1)
        log_probs.append(lp[token_id])

    total_log_prob = sum(log_probs) if log_probs else torch.tensor(0.0, device=device)

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    action_id = _parse_action(decoded)

    return action_id, total_log_prob


def _run_episode(
    model: Any,
    tokenizer: Any,
    env: CySentSecurityEnv,
    device: torch.device,
    scenario: str,
    attacker: str,
    difficulty: str,
    seed: int,
    max_steps: int,
    temperature: float,
) -> Tuple[List[torch.Tensor], List[float], Dict[str, Any]]:
    """Run one episode, collect log_probs and rewards."""
    obs, info = env.reset(seed=seed, options={
        "scenario": scenario,
        "difficulty": difficulty,
        "attacker": attacker,
        "action_source": "hf_llm_agent",
        "strategy_mode": "balanced",
        "intelligence_enabled": True,
    })

    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    actions_taken: List[str] = []
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        prompt = _build_prompt(info, scenario, attacker)
        action_id, log_prob = _generate_action(model, tokenizer, prompt, device, temperature)

        obs, reward, done, truncated, info = env.step(action_id)

        log_probs.append(log_prob)
        rewards.append(float(reward))
        actions_taken.append(ACTION_NAMES[action_id])
        steps += 1

    episode_stats = {
        "steps": steps,
        "total_reward": sum(rewards),
        "avg_reward": sum(rewards) / max(steps, 1),
        "final_risk": float(info.get("network_risk", 1.0)),
        "breaches": sum(1 for a in info.get("assets", []) if a.get("compromised")),
        "actions": actions_taken,
        "terminated": done,
    }
    return log_probs, rewards, episode_stats


def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def _detect_lora_targets(model: Any) -> List[str]:
    """Auto-detect LoRA target modules based on model architecture."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    if "qkv_proj" in names:
        return ["qkv_proj", "o_proj"]
    return ["q_proj", "v_proj"]


def train_hf_rl(
    model_id: str,
    output_dir: str,
    episodes: int = 20,
    max_steps: int = 40,
    lr: float = 5e-5,
    gamma: float = 0.97,
    temperature: float = 0.7,
    lora_r: int = 8,
    lora_alpha: int = 16,
    seed: int = 42,
    scenarios: List[str] | None = None,
    attackers: List[str] | None = None,
    difficulty: str = "medium",
    grad_accum: int = 4,
    early_stop_reward: float = 3.0,
    save_every: int = 5,
) -> Dict[str, Any]:
    _ensure_dynamic_cache_compat()
    scenarios = scenarios or ["bank", "saas"]
    attackers = attackers or ["ransomware_gang", "credential_thief"]
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_hf_rl] Device: {device}")
    print(f"[train_hf_rl] Model: {model_id}")
    print(f"[train_hf_rl] Episodes: {episodes}, max_steps: {max_steps}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    targets = _detect_lora_targets(base_model)
    print(f"[train_hf_rl] LoRA targets: {targets}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=targets,
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train_hf_rl] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    env = CySentSecurityEnv(max_steps=max_steps, seed=seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    baseline_reward = 0.0
    best_avg_reward = -999.0
    t0 = time.time()

    for ep in range(episodes):
        scenario = scenarios[int(rng.integers(0, len(scenarios)))]
        attacker = attackers[int(rng.integers(0, len(attackers)))]
        ep_seed = seed + ep * 137

        model.train()
        log_probs, rewards, stats = _run_episode(
            model, tokenizer, env, device,
            scenario, attacker, difficulty,
            ep_seed, max_steps, temperature,
        )

        returns = _compute_returns(rewards, gamma)
        baseline_reward = 0.9 * baseline_reward + 0.1 * stats["total_reward"]

        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for lp, G in zip(log_probs, returns):
            advantage = G - baseline_reward
            policy_loss = policy_loss + (-lp * advantage)

        if log_probs:
            policy_loss = policy_loss / len(log_probs)

        (policy_loss / grad_accum).backward()

        if (ep + 1) % grad_accum == 0 or ep == episodes - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        stats["episode"] = ep
        stats["policy_loss"] = float(policy_loss.item())
        stats["baseline"] = baseline_reward
        stats["scenario"] = scenario
        stats["attacker"] = attacker
        stats["elapsed_s"] = time.time() - t0
        history.append(stats)

        print(
            f"  ep {ep:3d} | reward {stats['total_reward']:+7.2f} | "
            f"risk {stats['final_risk']:.3f} | breaches {stats['breaches']} | "
            f"steps {stats['steps']:3d} | loss {stats['policy_loss']:.4f} | "
            f"{scenario}/{attacker}"
        )

        if (ep + 1) % save_every == 0:
            ckpt = out_path / f"checkpoint_ep{ep+1}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))

        avg_recent = np.mean([h["total_reward"] for h in history[-5:]]) if len(history) >= 5 else stats["total_reward"]
        if avg_recent > best_avg_reward:
            best_avg_reward = avg_recent

        if len(history) >= 8 and avg_recent >= early_stop_reward:
            print(f"[train_hf_rl] Early stop: avg reward {avg_recent:.2f} >= {early_stop_reward}")
            break

    model.save_pretrained(str(out_path / "final_adapter"))
    tokenizer.save_pretrained(str(out_path / "final_adapter"))

    (out_path / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8",
    )

    elapsed = time.time() - t0
    summary = {
        "model_id": model_id,
        "episodes_run": len(history),
        "total_time_s": round(elapsed, 1),
        "best_avg_reward": round(best_avg_reward, 3),
        "final_avg_reward": round(np.mean([h["total_reward"] for h in history[-5:]]), 3) if len(history) >= 5 else round(history[-1]["total_reward"], 3),
        "final_avg_risk": round(np.mean([h["final_risk"] for h in history[-5:]]), 3) if len(history) >= 5 else round(history[-1]["final_risk"], 3),
        "adapter_path": str(out_path / "final_adapter"),
        "trainable_params": trainable,
        "device": str(device),
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[train_hf_rl] Done in {elapsed:.0f}s. Adapter: {out_path / 'final_adapter'}")
    return summary


def _load_hf_adapter(
    model_id: str,
    adapter_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, Any]:
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def _run_hf_episode(
    model: Any,
    tokenizer: Any,
    env: CySentSecurityEnv,
    device: torch.device,
    scenario: str,
    attacker: str,
    difficulty: str,
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed, options={
        "scenario": scenario, "difficulty": difficulty,
        "attacker": attacker, "action_source": "hf_llm_agent",
        "strategy_mode": "balanced", "intelligence_enabled": True,
    })
    done, trunc, total_r, steps = False, False, 0.0, 0
    while not (done or trunc) and steps < max_steps:
        prompt = _build_prompt(info, scenario, attacker)
        action_id, _ = _generate_action(model, tokenizer, prompt, device, temperature=0.0)
        obs, reward, done, trunc, info = env.step(action_id)
        total_r += reward
        steps += 1
    return {
        "reward": total_r,
        "risk": float(info.get("network_risk", 1.0)),
        "breaches": sum(1 for a in info.get("assets", []) if a.get("compromised")),
        "steps": steps,
    }


def benchmark_hf_adapter(
    model_id: str,
    adapter_path: str,
    episodes: int = 10,
    max_steps: int = 40,
    seed: int = 42,
    scenarios: List[str] | None = None,
    attackers: List[str] | None = None,
    difficulty: str = "medium",
    extra_adapters: List[Tuple[str, str, str]] | None = None,
) -> Dict[str, Any]:
    """Benchmark HF adapter(s) vs random vs PPO (if available).

    extra_adapters: list of (label, base_model_id, adapter_path) tuples for
    additional LLM agents to include in the comparison.
    """
    scenarios = scenarios or ["bank", "saas"]
    attackers = attackers or ["ransomware_gang", "credential_thief"]
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    hf_models: Dict[str, Tuple[Any, Any]] = {}
    if adapter_path and Path(adapter_path).exists():
        m, t = _load_hf_adapter(model_id, adapter_path, device, dtype)
        hf_models["hf_rl_adapter"] = (m, t)
        del m, t
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for label, base_id, a_path in (extra_adapters or []):
        if Path(a_path).exists():
            m, t = _load_hf_adapter(base_id, a_path, device, dtype)
            hf_models[label] = (m, t)
            del m, t
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ppo_model = None
    try:
        from stable_baselines3 import PPO
        ppo_path = Path("backend/train/artifacts/best_model/best_model.zip")
        if not ppo_path.exists():
            ppo_path = Path("backend/train/artifacts/cysent_ppo.zip")
        if ppo_path.exists():
            ppo_model = PPO.load(str(ppo_path))
    except Exception:
        pass

    env = CySentSecurityEnv(max_steps=max_steps, seed=seed)
    agents: Dict[str, List[Dict[str, Any]]] = {name: [] for name in hf_models}
    agents["random"] = []
    if ppo_model is not None:
        agents["ppo"] = []

    for ep in range(episodes):
        scenario = scenarios[int(rng.integers(0, len(scenarios)))]
        attacker = attackers[int(rng.integers(0, len(attackers)))]
        ep_seed = seed + ep * 53

        for label, (model, tokenizer) in hf_models.items():
            agents[label].append(_run_hf_episode(
                model, tokenizer, env, device,
                scenario, attacker, difficulty, ep_seed, max_steps,
            ))

        obs, info = env.reset(seed=ep_seed, options={
            "scenario": scenario, "difficulty": difficulty,
            "attacker": attacker, "action_source": "ppo_agent",
            "strategy_mode": "balanced", "intelligence_enabled": True,
        })
        done, trunc, total_r, steps = False, False, 0.0, 0
        while not (done or trunc) and steps < max_steps:
            obs, reward, done, trunc, info = env.step(env.action_space.sample())
            total_r += reward
            steps += 1
        agents["random"].append({
            "reward": total_r, "risk": float(info.get("network_risk", 1.0)),
            "breaches": sum(1 for a in info.get("assets", []) if a.get("compromised")),
            "steps": steps,
        })

        if ppo_model is not None:
            obs, info = env.reset(seed=ep_seed, options={
                "scenario": scenario, "difficulty": difficulty,
                "attacker": attacker, "action_source": "ppo_agent",
                "strategy_mode": "balanced", "intelligence_enabled": True,
            })
            done, trunc, total_r, steps = False, False, 0.0, 0
            while not (done or trunc) and steps < max_steps:
                action, _ = ppo_model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(int(action))
                total_r += reward
                steps += 1
            agents["ppo"].append({
                "reward": total_r, "risk": float(info.get("network_risk", 1.0)),
                "breaches": sum(1 for a in info.get("assets", []) if a.get("compromised")),
                "steps": steps,
            })

    results: Dict[str, Any] = {}
    for agent_name, runs in agents.items():
        if not runs:
            continue
        results[agent_name] = {
            "avg_reward": round(float(np.mean([r["reward"] for r in runs])), 3),
            "avg_risk": round(float(np.mean([r["risk"] for r in runs])), 3),
            "avg_breaches": round(float(np.mean([r["breaches"] for r in runs])), 3),
            "avg_steps": round(float(np.mean([r["steps"] for r in runs])), 1),
            "episodes": len(runs),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="HF LLM live RL training against CySent env")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_p = sub.add_parser("train", help="Train HF LLM with REINFORCE on live CySent env")
    train_p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    train_p.add_argument("--output-dir", type=str, default="outputs/hf_rl_adapter")
    train_p.add_argument("--episodes", type=int, default=20)
    train_p.add_argument("--max-steps", type=int, default=40)
    train_p.add_argument("--lr", type=float, default=5e-5)
    train_p.add_argument("--gamma", type=float, default=0.97)
    train_p.add_argument("--temperature", type=float, default=0.7)
    train_p.add_argument("--lora-r", type=int, default=8)
    train_p.add_argument("--lora-alpha", type=int, default=16)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--grad-accum", type=int, default=4)
    train_p.add_argument("--early-stop-reward", type=float, default=3.0)
    train_p.add_argument("--save-every", type=int, default=5)

    # --- benchmark ---
    bench_p = sub.add_parser("benchmark", help="Benchmark HF adapter(s) vs random/PPO")
    bench_p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    bench_p.add_argument("--adapter-path", type=str, default="outputs/hf_rl_adapter/final_adapter")
    bench_p.add_argument("--episodes", type=int, default=10)
    bench_p.add_argument("--max-steps", type=int, default=40)
    bench_p.add_argument("--seed", type=int, default=42)
    bench_p.add_argument(
        "--extra-adapter", action="append", default=[],
        help="Additional adapter: label:base_model_id:adapter_path",
    )
    bench_p.add_argument("--output", type=str, default="")

    args = parser.parse_args()

    if args.command == "train":
        summary = train_hf_rl(
            model_id=args.model_id,
            output_dir=args.output_dir,
            episodes=args.episodes,
            max_steps=args.max_steps,
            lr=args.lr,
            gamma=args.gamma,
            temperature=args.temperature,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            grad_accum=args.grad_accum,
            early_stop_reward=args.early_stop_reward,
            save_every=args.save_every,
        )
        print("\n" + json.dumps(summary, indent=2))

    elif args.command == "benchmark":
        extra: List[Tuple[str, str, str]] = []
        for spec in args.extra_adapter:
            parts = spec.split(":", 2)
            if len(parts) == 3:
                extra.append((parts[0], parts[1], parts[2]))
            else:
                print(f"[warn] Ignoring malformed --extra-adapter: {spec}")

        results = benchmark_hf_adapter(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            extra_adapters=extra,
        )

        print("\n=== Benchmark Results ===")
        header = f"{'Agent':<24} {'Avg Reward':>12} {'Avg Risk':>10} {'Avg Breaches':>14} {'Avg Steps':>10}"
        print(header)
        print("-" * len(header))
        for agent, stats in results.items():
            print(
                f"{agent:<24} {stats['avg_reward']:>+12.3f} {stats['avg_risk']:>10.3f} "
                f"{stats['avg_breaches']:>14.3f} {stats['avg_steps']:>10.1f}"
            )

        out_path = Path(args.output) if args.output else Path(args.adapter_path).parent / "benchmark_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
