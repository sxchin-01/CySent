# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.40.0",
#   "peft>=0.11.0",
#   "accelerate",
#   "bitsandbytes>=0.46.1",
#   "gymnasium",
#   "numpy",
#   "huggingface_hub",
#   "pyyaml",
# ]
# ///
"""
CySent Qwen Live RL — Hugging Face Jobs runner.

Usage (HF Jobs CLI):
  # Test run (~50 steps, ~$0.35)
  hf jobs uv run --flavor t4-small --secrets HF_TOKEN scripts/train_on_hf.py --test

  # Full run (~100 steps)
  hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 12h scripts/train_on_hf.py

  # Full run without SFT warm start
  hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 12h scripts/train_on_hf.py --no-warmstart
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CySent Qwen Live RL on HF Jobs")
    parser.add_argument("--steps", type=int, default=100, help="Training steps (episodes)")
    parser.add_argument("--test", action="store_true", help="Quick test: 50 steps only")
    parser.add_argument("--max-turns", type=int, default=100, help="Max env turns per episode")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--upload-repo", default="sxchin01/CySent-Qwen-RL",
                        help="HF Hub repo to upload trained adapter")
    parser.add_argument("--sft-adapter", default="sxchin01/CySent-adapter",
                        help="HF Hub repo with SFT adapter for warm start")
    parser.add_argument("--no-warmstart", action="store_true",
                        help="Skip SFT adapter warm start, train from base model")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    steps = 50 if args.test else args.steps
    token = os.environ.get("HF_TOKEN")

    # ── 1. Clone CySent repo ────────────────────────────────────
    REPO = "/tmp/CySent"
    marker = Path(REPO) / "backend" / "env" / "security_env.py"
    if not marker.exists():
        print(f"Cloning CySent repo to {REPO} ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/sxchin-01/CySent.git", REPO],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise RuntimeError("git clone failed. Is the repo public?")
        print("Clone OK.")
    else:
        print("CySent repo already present.")

    sys.path.insert(0, REPO)

    # ── 2. Download SFT adapter for warm start ──────────────────
    sft_local = None
    if not args.no_warmstart:
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading SFT adapter: {args.sft_adapter} ...")
            sft_local = snapshot_download(
                args.sft_adapter, token=token, cache_dir="/tmp/sft_cache",
            )
            print(f"SFT adapter downloaded to: {sft_local}")
        except Exception as e:
            print(f"SFT adapter download failed ({e}), training from base model.")
            sft_local = None

    # ── 3. Run live RL training ─────────────────────────────────
    output_dir = "/tmp/cysent_qwen_rl"
    from backend.train.train_qwen_rl import train

    final_path = train(
        model_id=args.model_id,
        adapter_path=sft_local,
        token=token,
        total_steps=steps,
        max_turns=args.max_turns,
        lr=args.lr,
        gamma=0.99,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        checkpoint_every=100,
        output_dir=output_dir,
        seed=args.seed,
    )

    # ── 4. Upload adapter to HF Hub ─────────────────────────────
    if token and args.upload_repo:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        print(f"\nUploading adapter to {args.upload_repo} ...")
        api.create_repo(args.upload_repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=final_path,
            repo_id=args.upload_repo,
            repo_type="model",
            commit_message=f"Qwen 2.5 3B live RL adapter (REINFORCE, {steps} steps)",
        )

        history_path = Path(output_dir) / "training_history.json"
        if history_path.exists():
            api.upload_file(
                path_or_fileobj=str(history_path),
                path_in_repo="training_history.json",
                repo_id=args.upload_repo,
                repo_type="model",
            )

        print(f"Adapter uploaded: https://huggingface.co/{args.upload_repo}")
    else:
        print("Skipping upload (no HF_TOKEN or --upload-repo).")
        print(f"Adapter saved locally at: {final_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
