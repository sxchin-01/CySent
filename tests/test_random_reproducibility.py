from __future__ import annotations

import random
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from backend.agents.random_agent import RandomAgent
from backend.agents.router import AgentRouter
from backend.api import main as api
from backend.env.security_env import CySentSecurityEnv
from backend.train.benchmark_agents import PolicySet


class RandomAgentReproducibilityTests(unittest.TestCase):
    def test_same_seed_produces_same_sequence(self) -> None:
        first = RandomAgent(seed=42)
        second = RandomAgent(seed=42)

        self.assertEqual(
            [first.predict_action() for _ in range(32)],
            [second.predict_action() for _ in range(32)],
        )

    def test_different_seeds_can_produce_different_sequences(self) -> None:
        first = RandomAgent(seed=42)
        second = RandomAgent(seed=43)

        self.assertNotEqual(
            [first.predict_action() for _ in range(32)],
            [second.predict_action() for _ in range(32)],
        )

    def test_global_random_calls_do_not_change_agent_sequence(self) -> None:
        first = RandomAgent(seed=42)
        expected = [first.predict_action() for _ in range(32)]

        random.seed(999)
        for _ in range(100):
            random.random()

        second = RandomAgent(seed=42)
        self.assertEqual(expected, [second.predict_action() for _ in range(32)])

    def test_api_autonomous_random_episode_is_reproducible_after_reset(self) -> None:
        def run(seed: int) -> list[int]:
            env = CySentSecurityEnv(max_steps=20)
            _, info = env.reset(seed=seed, options={"intelligence_enabled": False})
            with patch.object(AgentRouter, "_initialize_agents", return_value=None):
                router = AgentRouter(config={"default_agent": "random_agent", "mode": "ppo_only"})
            runtime = SimpleNamespace(
                env=env,
                agent_router=router,
                state_lock=threading.RLock(),
                last_info=info,
                current_episode_id="random-test",
                episode_done=False,
                replays={},
            )
            runtime.snapshot_state = lambda: {
                "episode_id": runtime.current_episode_id,
                "step": runtime.last_info.get("step", 0),
                "network_risk": runtime.last_info.get("network_risk", 0.0),
                "assets": runtime.last_info.get("assets", []),
                "profile": runtime.last_info.get("profile", {}),
            }
            actions = []
            with patch.object(api, "_get_runtime", return_value=runtime):
                api.reset(api.ResetRequest(seed=seed, action_source="random", intelligence_enabled=False), Mock())
                for _ in range(16):
                    response = api.step(Mock())
                    actions.append(response["selected_action"])
                    if response["terminated"] or response["truncated"]:
                        break
            return actions

        sequence_a = run(42)
        sequence_b = run(42)
        sequence_c = run(43)
        self.assertEqual(sequence_a, sequence_b)
        self.assertNotEqual(sequence_a, sequence_c)

    def test_benchmark_random_uses_random_agent_abstraction(self) -> None:
        policies = PolicySet(["random"], Path("unused-for-random.zip"))
        policies.random = Mock(spec=RandomAgent)
        policies.random.predict_action.return_value = 6

        policies.reset_episode("random", 42)
        action, identity, fallback = policies.decide(
            "random",
            CySentSecurityEnv(max_steps=1),
            np.zeros(67, dtype=np.float32),
            {},
        )

        policies.random.reset.assert_called_once_with(42)
        policies.random.predict_action.assert_called_once_with()
        self.assertEqual((action, identity, fallback), (6, "random", None))


if __name__ == "__main__":
    unittest.main()
