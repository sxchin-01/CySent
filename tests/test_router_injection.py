from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from backend.agents.qwen_rl_policy import QWEN_RL_SOURCE_ID
from backend.agents.identities import HISTORICAL_PPO_AGENT
from backend.agents.router import AgentRouter


class _InjectedAgent:
    def __init__(self, action: int, *, source_id: str | None = None, error: Exception | None = None) -> None:
        self.action = action
        self.error = error
        self.calls = 0
        self.seeds = []
        if source_id is not None:
            self.source_id = source_id

    def is_available(self) -> bool:
        return True

    def predict_action(self, *args, **kwargs) -> int:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.action

    def reset(self, seed) -> None:
        self.seeds.append(seed)

    def deployment_label(self) -> str:
        return "Injected Test Agent"


class AgentRouterInjectionTests(unittest.TestCase):
    def test_injected_agents_skip_default_construction_and_preserve_identity(self) -> None:
        ppo = _InjectedAgent(3)
        qwen = _InjectedAgent(7, source_id=QWEN_RL_SOURCE_ID)
        with patch("backend.agents.router.PPOAgent") as ppo_class, patch(
            "backend.agents.router.HFAgent"
        ) as hf_class, patch("backend.agents.router.QwenRLPolicyAgent") as qwen_class:
            router = AgentRouter(
                config={"default_agent": "ppo_agent", "mode": "hybrid", "hf_policy_mode": "qwen_rl"},
                ppo_agent=ppo,
                hf_agent=qwen,
            )

        ppo_class.assert_not_called()
        hf_class.assert_not_called()
        qwen_class.assert_not_called()
        self.assertIs(router.ppo_agent, ppo)
        self.assertIs(router.hf_agent, qwen)

    def test_normal_construction_remains_backward_compatible(self) -> None:
        ppo = _InjectedAgent(3)
        hf = _InjectedAgent(7)
        with patch("backend.agents.router.PPOAgent", return_value=ppo) as ppo_class, patch(
            "backend.agents.router.HFAgent", return_value=hf
        ) as hf_class:
            router = AgentRouter(config={
                "default_agent": "ppo_agent", "mode": "hybrid", "hf_policy_mode": "generative_legacy",
                "hf_timeout": 10.0,
            })

        ppo_class.assert_called_once_with(None, artifact_id="historical_ppo", device="cpu")
        hf_class.assert_called_once_with(adapter_path=None, timeout=10.0)
        self.assertIs(router.ppo_agent, ppo)
        self.assertIs(router.hf_agent, hf)

    def test_injected_hybrid_routing_reset_and_fallback_are_unchanged(self) -> None:
        ppo = _InjectedAgent(3)
        qwen = _InjectedAgent(7, source_id=QWEN_RL_SOURCE_ID)
        router = AgentRouter(
            config={
                "default_agent": "ppo_agent",
                "mode": "hybrid",
                "hybrid_threshold": 10,
                "full_llm": False,
                "hf_policy_mode": "qwen_rl",
            },
            ppo_agent=ppo,
            hf_agent=qwen,
        )
        observation = np.zeros(2)
        low_state = {"network_risk": 0.2}

        router.reset_episode(42)
        self.assertEqual(router.predict_action(observation, low_state), 3)
        self.assertEqual(router.last_used_agent, HISTORICAL_PPO_AGENT)
        for _ in range(8):
            router.predict_action(observation, low_state)
        self.assertEqual(router.predict_action(observation, low_state), 7)
        self.assertEqual(router.last_used_agent, QWEN_RL_SOURCE_ID)

        router.reset_episode(43)
        self.assertEqual(router.predict_action(observation, {"network_risk": 0.8}), 7)
        self.assertEqual(router.last_used_agent, QWEN_RL_SOURCE_ID)
        self.assertEqual(qwen.seeds, [42, 43])

        failing_qwen = _InjectedAgent(7, source_id=QWEN_RL_SOURCE_ID, error=RuntimeError("injected failure"))
        fallback_router = AgentRouter(
            config={"default_agent": "ppo_agent", "mode": "hybrid", "hybrid_threshold": 10},
            ppo_agent=ppo,
            hf_agent=failing_qwen,
        )
        fallback_router.reset_episode(44)
        self.assertEqual(fallback_router.predict_action(observation, {"network_risk": 0.8}), 3)
        self.assertEqual(fallback_router.last_used_agent, HISTORICAL_PPO_AGENT)
        self.assertIn("HF prediction failed", fallback_router.last_fallback_reason or "")


if __name__ == "__main__":
    unittest.main()
