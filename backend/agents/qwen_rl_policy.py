from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - Qwen RL requires torch at runtime
    torch = None

from backend.agents.hf_agent import HFAgent
from backend.env.security_env import ACTION_NAMES


HISTORICAL_ACTION_LIST = [ACTION_NAMES[index] for index in sorted(ACTION_NAMES)]
HISTORICAL_MAX_PROMPT_LENGTH = 256
EXPECTED_QWEN25_ACTION_TOKEN_IDS = [
    2982, 3400, 3400, 3400, 16213, 285, 78501, 30804, 35794, 759, 42014, 23169,
]
QWEN_RL_SOURCE_ID = "qwen_rl_policy"
QWEN_RL_POLICY_MODE = "historical_first_token_seeded_categorical_v1"
CONDITIONAL_PROBABILITY_LABEL = "probabilities conditional on the historical 12-action constrained policy"
HISTORICAL_COLLISION_LIMITATION = (
    "The historical Qwen RL policy scored CySent actions using the first token of each canonical action name. "
    "Three patch actions share the same first token under the Qwen2.5 tokenizer, so the learned policy assigns "
    "identical scores to those actions. Historical training sampled them as separate categories. Evaluation "
    "preserves this behavior using seeded categorical sampling rather than resolving the collision post hoc."
)


def build_historical_rl_prompt(info: Dict[str, Any]) -> str:
    risk = float(info.get("network_risk", 0.0))
    risk_breakdown = info.get("risk_breakdown", {})
    red_log = info.get("red_log", {})
    assets = info.get("assets", [])

    compromised = [asset["name"] for asset in assets if asset.get("compromised")]
    infected = [asset["name"] for asset in assets if asset.get("infected")]
    attack = str(red_log.get("attack", "unknown"))
    target = str(red_log.get("target", "unknown"))
    top_risks = sorted(
        (
            (name, value)
            for name, value in risk_breakdown.items()
            if name != "network_risk" and isinstance(value, (int, float))
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    risk_text = ", ".join(f"{name}={value:.2f}" for name, value in top_risks) if top_risks else "none"

    return (
        "You are an expert cybersecurity defender.\n"
        f"Network risk: {risk:.3f} | Attack: {attack} -> {target}\n"
        f"Top risks: {risk_text}\n"
        f"Compromised: {compromised or 'none'} | Infected: {infected or 'none'}\n"
        f"Choose ONE action from: {', '.join(HISTORICAL_ACTION_LIST)}\n"
        "Answer with ONLY the action name."
    )


def historical_action_token_ids(tokenizer: Any) -> List[int]:
    token_ids: List[int] = []
    for action_name in HISTORICAL_ACTION_LIST:
        encoded = tokenizer.encode(action_name, add_special_tokens=False)
        token_ids.append(int(encoded[0]) if encoded else 0)
    return token_ids


def collision_groups(action_token_ids: Sequence[int]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[str]] = defaultdict(list)
    for action_name, token_id in zip(HISTORICAL_ACTION_LIST, action_token_ids):
        grouped[int(token_id)].append(action_name)
    return [
        {"token_id": token_id, "actions": actions}
        for token_id, actions in grouped.items()
        if len(actions) > 1
    ]


QWEN_RL_POLICY_CONTRACT = {
    "mode": QWEN_RL_POLICY_MODE,
    "source_id": QWEN_RL_SOURCE_ID,
    "prompt": "historical_train_qwen_rl",
    "max_prompt_length": HISTORICAL_MAX_PROMPT_LENGTH,
    "action_order": HISTORICAL_ACTION_LIST,
    "action_token_ids": EXPECTED_QWEN25_ACTION_TOKEN_IDS,
    "selection": "seeded categorical sample over 12 gathered first-token logits",
    "numerics": {
        "model_inference": "unchanged model dtype (FP16 on the verified T4 run)",
        "raw_policy_scores": "model constrained logits in their original dtype",
        "categorical_normalization": "softmax over the 12 constrained logits after conversion to float32",
        "historical_training": "torch.distributions.Categorical(logits=action_logits)",
        "reproduction_scope": "distribution-faithful; not bit-for-bit historical floating-point sampling",
    },
    "rng": "agent-local CPU torch.Generator reset from benchmark episode seed",
    "conditional_probability_label": CONDITIONAL_PROBABILITY_LABEL,
    "collision_groups": collision_groups(EXPECTED_QWEN25_ACTION_TOKEN_IDS),
    "known_limitation": HISTORICAL_COLLISION_LIMITATION,
}


class QwenRLPolicyAgent(HFAgent):
    """Historical Qwen RL policy: local next-token scoring plus seeded categorical sampling."""

    source_id = QWEN_RL_SOURCE_ID
    policy_mode = QWEN_RL_POLICY_MODE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if torch is None:
            raise ImportError("torch is required for the Qwen RL constrained policy.")
        super().__init__(*args, **kwargs)
        self._sampling_generator = torch.Generator(device="cpu")
        self._episode_seed = 0
        self._sampling_generator.manual_seed(self._episode_seed)
        self._action_token_ids: Optional[List[int]] = None
        self.last_decision: Optional[Dict[str, Any]] = None

    def reset(self, seed: Optional[int] = None) -> None:
        self._episode_seed = int(seed if seed is not None else 0)
        self._sampling_generator.manual_seed(self._episode_seed)
        self.last_decision = None

    def _validated_action_token_ids(self) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Qwen RL tokenizer is not initialized.")
        if self._action_token_ids is None:
            token_ids = historical_action_token_ids(self.tokenizer)
            if token_ids != EXPECTED_QWEN25_ACTION_TOKEN_IDS:
                raise RuntimeError(
                    "Qwen RL tokenizer action-token IDs do not match the verified Qwen2.5 historical contract."
                )
            self._action_token_ids = token_ids
        return list(self._action_token_ids)

    def _ensure_local_policy_model(self) -> None:
        if self.client is not None:
            raise RuntimeError("Qwen RL constrained policy requires a local model; hosted generation is unsupported.")
        if not self._using_local_model:
            if self._local_load_attempted:
                raise RuntimeError("Qwen RL local model initialization previously failed.")
            self._local_load_attempted = True
            self._initialize_local_model()
            self._active_backend = "local"

    def _predict_action_impl(self, state: Dict[str, Any]) -> int:
        self.last_decision = None
        self._ensure_local_policy_model()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen RL local model is not initialized.")

        prompt = build_historical_rl_prompt(state)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=HISTORICAL_MAX_PROMPT_LENGTH,
        )
        try:
            model_device = next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            model_device = getattr(self.model, "device", None)
        if model_device is None:
            raise RuntimeError("Qwen RL model input device could not be determined.")
        inputs = {name: value.to(model_device) for name, value in inputs.items()}
        with torch.no_grad() if torch is not None else nullcontext():
            outputs = self.model(**inputs, use_cache=False)
        final_vocabulary_logits = outputs.logits[:, -1, :]
        action_token_ids = self._validated_action_token_ids()
        gather_index = torch.tensor(action_token_ids, device=final_vocabulary_logits.device, dtype=torch.long)
        constrained_logits = final_vocabulary_logits[0, gather_index]
        if constrained_logits.numel() != len(HISTORICAL_ACTION_LIST):
            raise RuntimeError("Qwen RL constrained policy did not produce exactly 12 action logits.")
        if not bool(torch.isfinite(constrained_logits).all()):
            raise RuntimeError("Qwen RL constrained policy produced non-finite action logits.")
        sampling_logits = constrained_logits.float()
        constrained_probabilities = torch.softmax(sampling_logits, dim=-1)
        if not bool(torch.isfinite(constrained_probabilities).all()):
            raise RuntimeError("Qwen RL constrained policy produced non-finite action probabilities.")

        cpu_probabilities = constrained_probabilities.detach().to(device="cpu")
        selected_action_id = int(torch.multinomial(
            cpu_probabilities,
            num_samples=1,
            replacement=True,
            generator=self._sampling_generator,
        ).item())

        logits = [float(value) for value in constrained_logits.detach().to(device="cpu", dtype=torch.float64).tolist()]
        probabilities = [float(value) for value in cpu_probabilities.tolist()]
        top_logit = max(logits)
        top_action_ids = [index for index, value in enumerate(logits) if value == top_logit]
        distinct_scores = sorted(set(logits), reverse=True)
        collision = next(
            (group for group in collision_groups(action_token_ids) if HISTORICAL_ACTION_LIST[selected_action_id] in group["actions"]),
            None,
        )
        self.last_decision = {
            "policy_mode": self.policy_mode,
            "source": self.source_id,
            "episode_seed": self._episode_seed,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "selected_action_id": selected_action_id,
            "selected_action_name": HISTORICAL_ACTION_LIST[selected_action_id],
            "action_token_ids": action_token_ids,
            "constrained_logits": logits,
            "constrained_probabilities": probabilities,
            "raw_constrained_logits_dtype": str(constrained_logits.dtype),
            "categorical_normalization_dtype": str(constrained_probabilities.dtype),
            "conditional_probability_label": CONDITIONAL_PROBABILITY_LABEL,
            "selected_action_constrained_probability": probabilities[selected_action_id],
            "top_action_ids": top_action_ids,
            "top_action_names": [HISTORICAL_ACTION_LIST[index] for index in top_action_ids],
            "top_constrained_probability": max(probabilities),
            "top_two_distinct_score_margin": (
                distinct_scores[0] - distinct_scores[1] if len(distinct_scores) > 1 else None
            ),
            "selected_action_in_collision_group": collision is not None,
            "selected_collision_group": collision,
            "argmax_action_id_diagnostic": top_action_ids[0],
            "argmax_action_name_diagnostic": HISTORICAL_ACTION_LIST[top_action_ids[0]],
            "argmax_tied_action_ids": top_action_ids,
        }
        return selected_action_id

    def deployment_label(self) -> str:
        return "Qwen RL Constrained Policy"
