"""Live RL training of Qwen against the CySent environment.

REINFORCE with exponential-moving-average baseline, LoRA on Qwen2.5-3B-Instruct.
Runs the real env loop: observe -> prompt LLM -> parse action -> env.step -> reward.
Designed for a single free-tier Colab GPU (T4 / L4).
"""
from __future__ import annotations

import argparse
import json
import os
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

# Filled on first use; avoids re-tokenizing 12 actions every env step.
_ACTION_TOKEN_IDS_CACHE: Optional[List[int]] = None


def _action_token_ids(tokenizer: Any) -> List[int]:
    global _ACTION_TOKEN_IDS_CACHE
    if _ACTION_TOKEN_IDS_CACHE is None:
        _ACTION_TOKEN_IDS_CACHE = []
        for name in ACTION_LIST:
            toks = tokenizer.encode(name, add_special_tokens=False)
            _ACTION_TOKEN_IDS_CACHE.append(toks[0] if toks else 0)
    return _ACTION_TOKEN_IDS_CACHE


def _max_prompt_len() -> int:
    return max(64, int(os.environ.get("CYSENT_RL_MAX_PROMPT_LEN", "256")))


def _policy_microbatch_size() -> int:
    """Call torch.cuda.empty_cache() every N backward passes (same episode); does not split optimizer steps."""
    return max(1, int(os.environ.get("CYSENT_RL_POLICY_MICROBATCH", "8")))


def _make_optimizer(params, lr: float, weight_decay: float):
    try:
        import bitsandbytes as bnb  # type: ignore

        if torch.cuda.is_available():
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    except Exception:
        pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


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


def _should_use_qlora() -> bool:
    if not torch.cuda.is_available():
        return False
    override = os.environ.get("CYSENT_RL_QLORA", "").strip().lower()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return True
    total = torch.cuda.get_device_properties(0).total_memory
    return total < 22 * 1024**3


def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_id: str,
    adapter_path: Optional[str],
    lora_r: int,
    lora_alpha: int,
    token: Optional[str] = None,
):
    global _ACTION_TOKEN_IDS_CACHE
    _ACTION_TOKEN_IDS_CACHE = None

    use_qlora = _should_use_qlora()
    print(f"[LiveRL] Loading base model: {model_id} (QLoRA={use_qlora})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = None

    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            base = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=token,
            )
        except Exception as exc:
            vram = torch.cuda.get_device_properties(0).total_memory
            if vram < 22 * 1024**3:
                raise RuntimeError(
                    "QLoRA (4-bit) is required on this GPU (~16GB) but failed to load. "
                    "Install: pip install bitsandbytes  (and restart runtime). "
                    f"Original error: {exc}"
                ) from exc
            print(f"[LiveRL] QLoRA load failed ({exc}), falling back to fp16.")
            use_qlora = False

    if not use_qlora:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True, token=token,
        )
        if adapter_path and Path(adapter_path).exists():
            print(f"[LiveRL] Loading SFT adapter from: {adapter_path}")
            try:
                from peft import PeftModel
                base = PeftModel.from_pretrained(base, adapter_path)
                base = base.merge_and_unload()
                print("[LiveRL] SFT adapter merged into base weights.")
            except Exception as e:
                print(f"[LiveRL] Could not merge SFT adapter ({e}), continuing with base model.")
        base = base.to(device)
    elif adapter_path and Path(adapter_path).exists():
        print(
            "[LiveRL] QLoRA mode: SFT merge skipped (use fp16 path or merge offline). "
            "Training LoRA on 4-bit base."
        )

    try:
        from peft import prepare_model_for_kbit_training
        if use_qlora:
            base = prepare_model_for_kbit_training(base)
    except Exception:
        pass

    targets = _detect_lora_targets(base)
    print(f"[LiveRL] LoRA targets: {targets}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=targets,
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()

    model.train()
    use_gc = os.environ.get("CYSENT_RL_GRAD_CHECKPOINT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if use_gc:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception as exc:
            print(f"[LiveRL] Gradient checkpointing skipped: {exc}")

    if use_qlora:
        device = _model_input_device(model)
    return model, tokenizer, device


def collect_trajectory(
    model,
    tokenizer,
    device: torch.device,
    env: CySentSecurityEnv,
    max_turns: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rollout without autograd — avoids keeping ~100 forward graphs in VRAM."""
    dev = _model_input_device(model)
    at_ids = torch.tensor(_action_token_ids(tokenizer), device=dev, dtype=torch.long)
    transitions: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"actions": [], "parsed_ok": 0, "fallback": 0, "turns": 0}

    was_training = model.training
    model.eval()
    obs, info = env.reset()
    mlen = _max_prompt_len()
    with torch.no_grad():
        for _ in range(max_turns):
            prompt = _build_prompt(info)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=mlen)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            outputs = model(**inputs, use_cache=False)
            next_logits = outputs.logits[:, -1, :]
            action_logits = next_logits[0, at_ids]
            action_dist = torch.distributions.Categorical(logits=action_logits)
            sampled = action_dist.sample()
            action_id = int(sampled.item())

            transitions.append({"prompt": prompt, "action_id": action_id})
            stats["actions"].append(ACTION_LIST[action_id])
            stats["parsed_ok"] += 1

            obs, reward, terminated, truncated, info = env.step(action_id)
            transitions[-1]["reward"] = float(reward)
            stats["turns"] += 1

            if terminated or truncated:
                break

    if was_training:
        model.train()
    return transitions, stats


def forward_log_prob_of_action(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    action_id: int,
) -> torch.Tensor:
    """log π(a|s) for one (prompt, action) — use inside training backward."""
    dev = _model_input_device(model)
    at_ids = _action_token_ids(tokenizer)
    mlen = _max_prompt_len()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=mlen)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    outputs = model(**inputs, use_cache=False)
    next_logits = outputs.logits[:, -1, :]
    idx = torch.tensor(at_ids, device=dev, dtype=torch.long)
    action_logits = next_logits[0, idx]
    log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
    return log_probs[action_id]


def train(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: Optional[str] = None,
    token: Optional[str] = None,
    total_steps: int = 300,
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
    optimizer = _make_optimizer(
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
    print(
        f"  Steps: {total_steps} | Max turns/ep: {max_turns} | max_prompt_len={_max_prompt_len()} "
        f"| policy_microbatch={_policy_microbatch_size()}"
    )
    print(f"  LR: {lr} | Gamma: {gamma} | LoRA r={lora_r}")
    print(f"{'='*60}\n")

    t0 = time.time()
    step = 0

    while step < total_steps:
        env_seed = seed + step
        env = CySentSecurityEnv(max_steps=max_turns, seed=env_seed)

        transitions, stats = collect_trajectory(model, tokenizer, device, env, max_turns)

        if not transitions:
            step += 1
            continue

        rewards = [tr["reward"] for tr in transitions]
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

        n_t = len(transitions)
        policy_loss_val = 0.0
        mb = _policy_microbatch_size()
        optimizer.zero_grad()
        for i, (tr, adv) in enumerate(zip(transitions, advantages)):
            lp = forward_log_prob_of_action(
                model, tokenizer, device, tr["prompt"], int(tr["action_id"]),
            )
            contrib = (-lp * adv) / n_t
            policy_loss_val += float(contrib.detach().item())
            contrib.backward()
            if mb > 1 and (i + 1) % mb == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        del transitions, advantages, returns_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        step += 1
        elapsed = time.time() - t0

        record = {
            "step": step, "reward": round(ep_reward, 3),
            "baseline": round(baseline, 3), "loss": round(policy_loss_val, 4),
            "turns": stats["turns"], "elapsed": round(elapsed, 1),
        }
        history.append(record)

        if step % 10 == 0 or step == 1:
            print(
                f"[Step {step:>4}/{total_steps}]  reward={ep_reward:>8.2f}  "
                f"baseline={baseline:>8.2f}  loss={policy_loss_val:>7.4f}  "
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
    parser.add_argument("--steps", type=int, default=300)
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
