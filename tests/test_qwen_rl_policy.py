from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from backend.agents.qwen_rl_policy import (
    CONDITIONAL_PROBABILITY_LABEL,
    EXPECTED_QWEN25_ACTION_TOKEN_IDS,
    HISTORICAL_ACTION_LIST,
    HISTORICAL_MAX_PROMPT_LENGTH,
    QWEN_RL_POLICY_MODE,
    QWEN_RL_SOURCE_ID,
    QwenRLPolicyAgent,
    build_historical_rl_prompt,
    collision_groups,
    historical_action_token_ids,
)
from backend.agents.router import AgentRouter
from backend.env.security_env import ACTION_NAMES
from backend.train import benchmark_agents, train_qwen_rl


class _Tokenizer:
    def __init__(self) -> None:
        self.last_prompt = None
        self.last_kwargs = None

    def encode(self, name, add_special_tokens=False):
        index = HISTORICAL_ACTION_LIST.index(name)
        return [EXPECTED_QWEN25_ACTION_TOKEN_IDS[index], 90000 + index]

    def __call__(self, prompt, **kwargs):
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _Model:
    device = torch.device("cpu")

    def __init__(self, action_scores=None, dtype=torch.float32) -> None:
        self.action_scores = action_scores or [0.0] * len(HISTORICAL_ACTION_LIST)
        self.dtype = dtype

    def __call__(self, **kwargs):
        logits = torch.full(
            (1, 3, max(EXPECTED_QWEN25_ACTION_TOKEN_IDS) + 1), -100.0, dtype=self.dtype,
        )
        for token_id, score in zip(EXPECTED_QWEN25_ACTION_TOKEN_IDS, self.action_scores):
            logits[0, -1, token_id] = score
        return SimpleNamespace(logits=logits)


def _policy(action_scores=None, dtype=torch.float32) -> QwenRLPolicyAgent:
    agent = object.__new__(QwenRLPolicyAgent)
    agent.model = _Model(action_scores, dtype=dtype)
    agent.tokenizer = _Tokenizer()
    agent.adapter_path = "local"
    agent.client = None
    agent._using_local_model = True
    agent._local_load_attempted = True
    agent._active_backend = "local"
    agent._sampling_generator = torch.Generator(device="cpu")
    agent._episode_seed = 0
    agent._sampling_generator.manual_seed(0)
    agent._action_token_ids = None
    agent.last_decision = None
    return agent


class QwenRLContractTests(unittest.TestCase):
    def test_historical_action_order_and_exact_collision_group(self) -> None:
        self.assertEqual(HISTORICAL_ACTION_LIST, [ACTION_NAMES[index] for index in range(12)])
        self.assertEqual(historical_action_token_ids(_Tokenizer()), EXPECTED_QWEN25_ACTION_TOKEN_IDS)
        self.assertEqual(collision_groups(EXPECTED_QWEN25_ACTION_TOKEN_IDS), [{
            "token_id": 3400,
            "actions": ["patch_hr_systems", "patch_web_server", "patch_auth_server"],
        }])

    def test_duplicate_patch_logits_remain_separate_equal_categories(self) -> None:
        scores = [-2.0, 3.0, -7.0, 9.0] + [-2.0] * 8
        agent = _policy(scores)
        agent.reset(42)
        agent._predict_action_impl(self._state())
        diagnostics = agent.last_decision

        self.assertEqual(diagnostics["constrained_logits"][1:4], [9.0, 9.0, 9.0])
        patch_probabilities = diagnostics["constrained_probabilities"][1:4]
        self.assertEqual(patch_probabilities[0], patch_probabilities[1])
        self.assertEqual(patch_probabilities[1], patch_probabilities[2])
        self.assertAlmostEqual(sum(diagnostics["constrained_probabilities"]), 1.0, places=6)
        self.assertEqual(diagnostics["conditional_probability_label"], CONDITIONAL_PROBABILITY_LABEL)
        self.assertEqual(diagnostics["top_action_ids"], [1, 2, 3])

    def test_fp16_logits_are_normalized_in_fp32_without_small_probability_underflow(self) -> None:
        scores = [
            7.19140625, 22.28125, 22.28125, 22.28125, 0.86572265625, 0.381591796875,
            -6.1328125, -3.7734375, 3.3359375, -2.9375, -1.4931640625, 4.88671875,
        ]
        agent = _policy(scores, dtype=torch.float16)
        global_before = torch.random.get_rng_state().clone()
        agent.reset(42)

        agent._predict_action_impl(self._state())
        diagnostics = agent.last_decision
        probabilities = diagnostics["constrained_probabilities"]

        self.assertEqual(len(probabilities), 12)
        self.assertTrue(all(torch.isfinite(torch.tensor(probabilities))))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertEqual(probabilities[1], probabilities[2])
        self.assertEqual(probabilities[2], probabilities[3])
        self.assertGreater(probabilities[6], 0.0)
        self.assertGreater(probabilities[7], 0.0)
        self.assertEqual(diagnostics["raw_constrained_logits_dtype"], "torch.float16")
        self.assertEqual(diagnostics["categorical_normalization_dtype"], "torch.float32")
        self.assertEqual(diagnostics["policy_mode"], QWEN_RL_POLICY_MODE)
        self.assertEqual(diagnostics["source"], QWEN_RL_SOURCE_ID)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))

    def test_seeded_sampling_reproduces_sequences_without_global_rng_use(self) -> None:
        agent = _policy()
        global_before = torch.random.get_rng_state().clone()

        agent.reset(42)
        first = [agent._predict_action_impl(self._state()) for _ in range(20)]
        agent.reset(42)
        second = [agent._predict_action_impl(self._state()) for _ in range(20)]
        agent.reset(43)
        third = [agent._predict_action_impl(self._state()) for _ in range(20)]

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))

    def test_scoring_maps_sampled_category_to_canonical_action_id(self) -> None:
        scores = [-100.0] * 12
        scores[11] = 100.0
        agent = _policy(scores)
        agent.reset(7)

        action = agent._predict_action_impl(self._state())

        self.assertEqual(action, 11)
        self.assertEqual(agent.last_decision["selected_action_name"], "segment_finance_database")
        self.assertEqual(agent.last_decision["source"], QWEN_RL_SOURCE_ID)
        self.assertEqual(agent.last_decision["policy_mode"], QWEN_RL_POLICY_MODE)

    def test_prompt_and_tokenizer_settings_match_historical_training(self) -> None:
        state = self._state()
        expected = (
            "You are an expert cybersecurity defender.\n"
            "Network risk: 0.456 | Attack: credential_theft -> auth_server\n"
            "Top risks: credential=0.90, malware=0.70, lateral=0.40\n"
            "Compromised: ['db'] | Infected: ['web']\n"
            f"Choose ONE action from: {', '.join(HISTORICAL_ACTION_LIST)}\n"
            "Answer with ONLY the action name."
        )
        self.assertEqual(build_historical_rl_prompt(state), expected)
        self.assertEqual(train_qwen_rl._build_prompt(state), expected)
        self.assertEqual(HISTORICAL_MAX_PROMPT_LENGTH, 256)

        agent = _policy()
        agent.reset(1)
        agent._predict_action_impl(state)
        self.assertEqual(agent.tokenizer.last_prompt, expected)
        self.assertTrue(agent.tokenizer.last_kwargs["truncation"])
        self.assertEqual(agent.tokenizer.last_kwargs["max_length"], 256)

    def test_benchmark_qwen_and_hybrid_request_constrained_policy_mode(self) -> None:
        configs = []

        class Router:
            def __init__(self, config):
                configs.append(config)

            def switch_agent(self, name):
                return True

            def is_agent_available(self, name):
                return True

        with patch.object(benchmark_agents, "AgentRouter", Router):
            benchmark_agents.PolicySet(["qwen_rl", "hybrid_router"], benchmark_agents.DEFAULT_PPO_PATH)

        self.assertEqual([config["hf_policy_mode"] for config in configs], ["qwen_rl", "qwen_rl"])

    def test_router_qwen_branch_reports_constrained_source_and_resets_seed(self) -> None:
        class FakePolicy:
            source_id = QWEN_RL_SOURCE_ID

            def __init__(self):
                self.seeds = []

            def is_available(self):
                return True

            def reset(self, seed):
                self.seeds.append(seed)

            def predict_action(self, state):
                return 8

        with patch.object(AgentRouter, "_initialize_agents", return_value=None):
            router = AgentRouter(config={"default_agent": "ppo_agent", "mode": "hybrid"})
        router.hf_agent = FakePolicy()
        router.ppo_agent = SimpleNamespace(is_available=lambda: True, predict_action=lambda *args, **kwargs: 0)
        router.reset_episode(44)

        self.assertEqual(router.predict_action(np.zeros(2), {"network_risk": 0.8}), 8)
        self.assertEqual(router.last_used_agent, QWEN_RL_SOURCE_ID)
        self.assertEqual(router.hf_agent.seeds, [44])

    @staticmethod
    def _state():
        return {
            "network_risk": 0.456,
            "risk_breakdown": {"malware": 0.7, "credential": 0.9, "network_risk": 0.456, "lateral": 0.4},
            "red_log": {"attack": "credential_theft", "target": "auth_server"},
            "assets": [
                {"name": "db", "compromised": True, "infected": False},
                {"name": "web", "compromised": False, "infected": True},
            ],
        }


if __name__ == "__main__":
    unittest.main()
