from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.agents.identities import (
    CANONICAL_RESEARCH_AGENTS,
    FRESH_PPO_AGENT,
    HEURISTIC_AGENT,
    HISTORICAL_PPO_AGENT,
    HYBRID_AGENT,
    QWEN_RL_AGENT,
    RANDOM_AGENT,
)
from backend.agents.qwen_rl_policy import QWEN_RL_POLICY_MODE, QWEN_RL_SOURCE_ID, QwenRLPolicyAgent
from backend.agents.ppo_agent import CUSTOM_PPO_IDENTITY, PPOAgent
from backend.agents.router import AgentRouter
from backend.artifacts import (
    ArtifactVerificationError,
    load_artifact_manifest,
    verify_local_artifact,
    verify_qwen_snapshot,
)
from backend.train import benchmark_agents


class _Agent:
    def __init__(self, action: int, *, source_id: str | None = None, error: Exception | None = None) -> None:
        self.action = action
        self.error = error
        if source_id:
            self.source_id = source_id

    def is_available(self) -> bool:
        return True

    def predict_action(self, *args, **kwargs) -> int:
        if self.error:
            raise self.error
        return self.action

    def reset(self, seed=None) -> None:
        pass

    def deployment_label(self) -> str:
        return "test"


class DeploymentContractTests(unittest.TestCase):
    def test_canonical_identities_match_authoritative_benchmark(self) -> None:
        self.assertEqual(CANONICAL_RESEARCH_AGENTS, (
            RANDOM_AGENT, HEURISTIC_AGENT, HISTORICAL_PPO_AGENT,
            FRESH_PPO_AGENT, QWEN_RL_AGENT, HYBRID_AGENT,
        ))
        self.assertEqual(benchmark_agents.PPO_AGENT, HISTORICAL_PPO_AGENT)
        self.assertEqual(benchmark_agents.FRESH_PPO_AGENT, FRESH_PPO_AGENT)

    def test_manifest_freezes_artifact_and_policy_identity(self) -> None:
        manifest = load_artifact_manifest()
        self.assertEqual(manifest["historical_qwen_policy"], QWEN_RL_POLICY_MODE)
        self.assertEqual(
            manifest["artifacts"]["historical_ppo"]["sha256"],
            "4D4955993DD2D98CC3B8D3319C1CDA92F10EFBA95407B3D78D9763E5D968D1AF",
        )
        self.assertEqual(
            manifest["artifacts"]["fresh_ppo"]["sha256"],
            "7BA9122F3AE4BD67E539EC3BEE95EB587F63F22091605DDEF044190586DC0197",
        )
        self.assertEqual(
            manifest["artifacts"]["qwen_merged"]["revision"],
            "fb75512b037bb37de575916afd900c03ab860cb5",
        )

    def test_missing_or_mismatched_artifact_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.zip"
            with self.assertRaisesRegex(ArtifactVerificationError, "unavailable"):
                verify_local_artifact("historical_ppo", missing)
            wrong = Path(temp_dir) / "wrong.zip"
            wrong.write_bytes(b"not the frozen checkpoint")
            with self.assertRaisesRegex(ArtifactVerificationError, "SHA256 mismatch"):
                verify_local_artifact("historical_ppo", wrong)
            with self.assertRaisesRegex(ArtifactVerificationError, "SHA256 mismatch"):
                verify_local_artifact("fresh_ppo", wrong)

            with patch("backend.agents.ppo_agent.PPO.load") as load:
                with self.assertRaisesRegex(ArtifactVerificationError, "SHA256 mismatch"):
                    PPOAgent(str(wrong), artifact_id="historical_ppo")
                load.assert_not_called()

    def test_custom_ppo_is_distinct_and_default_device_semantics_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "custom.zip"
            custom.write_bytes(b"custom checkpoint fixture")
            with patch("backend.agents.ppo_agent.PPO.load", return_value=object()) as load:
                agent = PPOAgent(str(custom))
            self.assertEqual(agent.identity, CUSTOM_PPO_IDENTITY)
            load.assert_called_once_with(str(custom))

        with patch.object(benchmark_agents, "PPOAgent", return_value=object()) as agent_class:
            benchmark_agents.PolicySet([HISTORICAL_PPO_AGENT], Path("arbitrary-location.zip"))
        agent_class.assert_called_once_with("arbitrary-location.zip", artifact_id="historical_ppo")

    def test_qwen_rl_missing_local_artifact_is_unavailable(self) -> None:
        agent = object.__new__(QwenRLPolicyAgent)
        agent.client = None
        agent.model = None
        agent.tokenizer = None
        agent._using_local_model = False
        agent.adapter_path = "definitely/missing/qwen"
        self.assertFalse(agent.is_available())

    def test_qwen_requires_exact_immutable_hf_snapshot_provenance(self) -> None:
        revision = "fb75512b037bb37de575916afd900c03ab860cb5"
        with tempfile.TemporaryDirectory() as temp_dir:
            arbitrary = Path(temp_dir) / "qwen-model"
            arbitrary.mkdir()
            with self.assertRaisesRegex(ArtifactVerificationError, "provenance is unverified"):
                verify_qwen_snapshot(arbitrary)

            snapshot = (
                Path(temp_dir)
                / "models--sxchin01--CySent-Qwen-RL-merged"
                / "snapshots"
                / revision
            )
            snapshot.mkdir(parents=True)
            verified = verify_qwen_snapshot(snapshot)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["revision"], revision)

            agent = object.__new__(QwenRLPolicyAgent)
            agent.client = None
            agent.model = None
            agent.tokenizer = None
            agent._using_local_model = False
            agent.adapter_path = str(snapshot)
            self.assertTrue(agent.is_available())

    def test_hybrid_fallback_identity_and_reason_are_truthful(self) -> None:
        router = AgentRouter(
            config={"default_agent": HISTORICAL_PPO_AGENT, "mode": "hybrid", "hybrid_threshold": 10},
            ppo_agent=_Agent(3),
            hf_agent=_Agent(7, source_id=QWEN_RL_SOURCE_ID, error=RuntimeError("failure")),
        )
        self.assertTrue(router.switch_agent(HYBRID_AGENT))
        action = router.predict_action(np.zeros(2), {"network_risk": 0.8})
        self.assertEqual(action, 3)
        self.assertEqual(router.last_used_agent, HISTORICAL_PPO_AGENT)
        self.assertTrue(router.last_fallback_reason)

    def test_cpu_random_heuristic_path_needs_no_model_and_preserves_global_rng(self) -> None:
        global_state = random.getstate()
        policies = benchmark_agents.PolicySet([RANDOM_AGENT, HEURISTIC_AGENT], Path("missing-not-used.zip"))
        plan = benchmark_agents.build_experiment_plan([42])[:1]
        random_row = benchmark_agents.run_episode(
            agent=RANDOM_AGENT, episode_index=0, case=plan[0], max_steps=3, policies=policies,
        )
        heuristic_row = benchmark_agents.run_episode(
            agent=HEURISTIC_AGENT, episode_index=0, case=plan[0], max_steps=3, policies=policies,
        )
        self.assertEqual((random_row.agent, heuristic_row.agent), (RANDOM_AGENT, HEURISTIC_AGENT))
        self.assertEqual(random.getstate(), global_state)


if __name__ == "__main__":
    unittest.main()
