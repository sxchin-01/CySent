from __future__ import annotations

import argparse
import re
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_ACTIONS: List[str] = [
    "do_nothing",
    "patch_hr_systems",
    "patch_web_server",
    "patch_auth_server",
    "rotate_credentials",
    "isolate_suspicious_host",
    "increase_monitoring",
    "restore_backup",
    "deploy_honeypot",
    "phishing_training",
    "investigate_top_alert",
    "segment_finance_database",
]

ALIASES = {
    "isolate_host": "isolate_suspicious_host",
}


def parse_action(text: str) -> str:
    lower = text.lower()
    for alias, mapped in ALIASES.items():
        if alias in lower:
            return mapped
    for action in VALID_ACTIONS:
        if action in lower:
            return action
    return "do_nothing"


def build_prompt(state_text: str) -> str:
    actions = ", ".join(VALID_ACTIONS)
    return (
        "You are CySent BLUE defender. Return exactly one action label from this list: "
        f"{actions}.\n"
        f"State: {state_text}\n"
        "Action:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference for CySent adapter/model")
    parser.add_argument("--base-model", type=str, required=True, help="Base model id or local path")
    parser.add_argument("--adapter", type=str, default="", help="Optional LoRA adapter path")
    parser.add_argument(
        "--state",
        type=str,
        default="risk=0.72, attack=phishing_email, target=auth server, compromised=1, credential_exposure=0.81",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    prompt = build_prompt(args.state)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    predicted = parse_action(decoded)

    print("--- Prompt ---")
    print(prompt)
    print("--- Raw Output ---")
    print(decoded)
    print("--- Parsed CySent Action ---")
    print(predicted)


if __name__ == "__main__":
    main()
