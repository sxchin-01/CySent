"""Live RL training of Qwen against the CySent environment.

REINFORCE with exponential-moving-average baseline, LoRA on Qwen2.5-3B-Instruct.
Runs the real env loop: observe -> prompt LLM -> parse action -> env.step -> reward.
Designed for a single free-tier Colab GPU (T4 / L4).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv

ACTION_LIST = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]
NUM_ACTIONS = len(ACTION_LIST)
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_LIST)}


def _build_prompt(info: Dict[str, Any]) -> str:
    risk = float(info.get("network_risk", 0.0))
    rb = info.get("risk_breakdown", {})
    red = info.get("red_log", {})
    assets = info.get("assets", [])

    compromised = [a["name"] for a in assets if a.get("compromised")]
    infected = [a["name"] for a in assets if a.get("infected")]
    attack = str(red.get("attack", "unknown"))
    target = str(red.get("target", "unknown"))

    top_risks = sorted(
        ((k, v) for k, v in rb.items() if k != "network_risk" and isinstance(v, (int, float))),
        key=lambda x: x[1], reverse=True,
    )[:3]
    risk_str = ", ".join(f"{k}={v:.2f}" for k, v in top_risks) if top_risks else "none"

    return (
        f"You are an expert cybersecurity defender.\n"
        f"Network risk: {risk:.3f} | Attack: {attack} -> {target}\n"
        f"Top risks: {risk_str}\n"
        f"Compromised: {compromised or 'none'} | Infected: {infected or 'none'}\n"
        f"Choose ONE action from: {', '.join(ACTION_LIST)}\n"
        f"Answer with ONLY the action name."
    )


def _parse_action(text: str) -> Optional[int]:
    text = text.strip().lower().split("\n")[0]
    for name in ACTION_LIST:
        if name in text:
            return ACTION_TO_ID[name]
    digits = re.findall(r"\d+", text)
    if digits:
        idx = int(digits[0])
        if 0 <= idx < NUM_ACTIONS:
            return idx
    return None


def _detect_lora_targets(model: torch.nn.Module) -> List[str]:
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, _ in model.named_modules():
        short = name.split(".")[-1]
        if short in candidates:
            found.add(short)
    return sorted(found) if found else ["q_proj", "v_proj"]


def load_model(
    model_id: str,
    adapter_path: Optional[str],
    lora_r: int,
    lora_alpha: int,
    token: Optional[str] = None,
):
    print(f"[LiveRL] Loading base model: {model_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and Path(adapter_path).exists():
        print(f"[LiveRL] Loading SFT adapter from: {adapter_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print("[LiveRL] SFT adapter merged into base weights.")
        except Exception as e:
            print(f"[LiveRL] Could not merge SFT adapter ({e}), continuing with base model.")

    targets = _detect_lora_targets(model)
    print(f"[LiveRL] LoRA targets: {targets}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    return model, tokenizer, device


def collect_trajectory(
    model, tokenizer, device, env: CySentSecurityEnv, max_turns: int = 150,
) -> Tuple[List[torch.Tensor], List[float], Dict[str, Any]]:
    obs, info = env.reset()
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    stats = {"actions": [], "parsed_ok": 0, "fallback": 0, "turns": 0}

    for t in range(max_turns):
        prompt = _build_prompt(info)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        next_logits = outputs.logits[:, -1, :]

        action_token_ids = []
        for name in ACTION_LIST:
            toks = tokenizer.encode(name, add_special_tokens=False)
            action_token_ids.append(toks[0] if toks else 0)

        action_logits = next_logits[0, action_token_ids]
        action_dist = torch.distributions.Categorical(logits=action_logits)
        sampled = action_dist.sample()
        log_prob = action_dist.log_prob(sampled)

        action_id = int(sampled.item())
        log_probs.append(log_prob)
        stats["actions"].append(ACTION_LIST[action_id])
        stats["parsed_ok"] += 1

        obs, reward, terminated, truncated, info = env.step(action_id)
        rewards.append(float(reward))
        stats["turns"] += 1

        if terminated or truncated:
            break

    return log_probs, rewards, stats


def train(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
    token: Optional[str] = None,
    total_steps: int = 500,
    max_turns: int = 100,
    lr: float = 1e-5,
    gamma: float = 0.99,
    baseline_beta: float = 0.9,
    lora_r: int = 16,
    lora_alpha: int = 32,
    checkpoint_every: int = 100,
    output_dir: str = "outputs/cysent_qwen_rl",
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer, device = load_model(model_id, adapter_path, lora_r, lora_alpha, token)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = CySentSecurityEnv(max_steps=max_turns, seed=seed)
    baseline = 0.0
    history: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  CySent Live RL — REINFORCE + Baseline")
    print(f"  Model: {model_id}")
    print(f"  SFT adapter: {adapter_path or 'none'}")
    print(f"  Steps: {total_steps} | Max turns/ep: {max_turns}")
    print(f"  LR: {lr} | Gamma: {gamma} | LoRA r={lora_r}")
    print(f"{'='*60}\n")

    t0 = time.time()
    step = 0

    while step < total_steps:
        env_seed = seed + step
        env = CySentSecurityEnv(max_steps=max_turns, seed=env_seed)

        log_probs, rewards, stats = collect_trajectory(model, tokenizer, device, env, max_turns)

        if not log_probs:
            step += 1
            continue

        ep_reward = sum(rewards)
        baseline = baseline_beta * baseline + (1 - baseline_beta) * ep_reward

        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
        advantages = returns_t - baseline

        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std

        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for lp, adv in zip(log_probs, advantages):
            policy_loss = policy_loss + (-lp * adv.detach())
        policy_loss = policy_loss / len(log_probs)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        elapsed = time.time() - t0

        record = {
            "step": step, "reward": round(ep_reward, 3),
            "baseline": round(baseline, 3), "loss": round(policy_loss.item(), 4),
            "turns": stats["turns"], "elapsed": round(elapsed, 1),
        }
        history.append(record)

        if step % 10 == 0 or step == 1:
            print(
                f"[Step {step:>4}/{total_steps}]  reward={ep_reward:>8.2f}  "
                f"baseline={baseline:>8.2f}  loss={policy_loss.item():>7.4f}  "
                f"turns={stats['turns']:>3}  elapsed={elapsed:.0f}s"
            )

        if step % checkpoint_every == 0:
            ckpt = out / f"checkpoint_step{step}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            print(f"  -> Checkpoint saved: {ckpt}")

    final_path = out / "final_adapter"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nFinal adapter saved: {final_path}")

    history_path = out / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Training history: {history_path}")

    if history:
        first_10 = np.mean([h["reward"] for h in history[:10]])
        last_10 = np.mean([h["reward"] for h in history[-10:]])
        print(f"\nAvg reward first 10 eps: {first_10:.2f}")
        print(f"Avg reward last  10 eps: {last_10:.2f}")
        print(f"Improvement: {last_10 - first_10:+.2f}")

    return str(final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live RL training of Qwen on CySent")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Path to SFT adapter (warm start)")
    parser.add_argument("--token", default="", help="HF token for gated models")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/cysent_qwen_rl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        model_id=args.model_id,
        adapter_path=args.adapter_path or None,
        token=args.token or None,
        total_steps=args.steps,
        max_turns=args.max_turns,
        lr=args.lr,
        gamma=args.gamma,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        seed=args.seed,
    )
