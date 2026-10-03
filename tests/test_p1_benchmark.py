from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.agents.heuristic_agent import HeuristicAgent
from backend.env.security_env import CySentSecurityEnv
from backend.train import benchmark_agents as benchmark


class HeuristicAgentTests(unittest.TestCase):
    def base_state(self):
        env = CySentSecurityEnv(max_steps=5)
        _, info = env.reset(seed=42)
        return benchmark._agent_state(info)

    def test_representative_rules_are_deterministic(self) -> None:
        state = self.base_state()
        for asset in state["assets"]:
            if asset["name"] == "Finance Database":
                asset["compromised"] = True
        first = HeuristicAgent().predict_action(state)
        second = HeuristicAgent().predict_action(state)
        self.assertEqual(first, second)
        self.assertEqual(first, 5)

        alert_state = self.base_state()
        alert_state["alerts"] = [{"severity_score": 0.9}]
        self.assertEqual(HeuristicAgent().predict_action(alert_state), 10)

        credential_state = self.base_state()
        credential_state["risk_breakdown"]["credential_exposure"] = 0.8
        self.assertEqual(HeuristicAgent().predict_action(credential_state), 4)

    def test_heuristic_uses_no_red_log_or_hidden_threat_state(self) -> None:
        state = self.base_state()
        state["red_log"] = {"attack": "ransomware_attempt", "success": True}
        state["hidden_threat_pressure"] = 1.0
        action_with_hidden = HeuristicAgent().predict_action(state)
        state.pop("red_log")
        state.pop("hidden_threat_pressure")
        action_without_hidden = HeuristicAgent().predict_action(state)
        self.assertEqual(action_with_hidden, action_without_hidden)


class BenchmarkCorrectnessTests(unittest.TestCase):
    def test_plan_propagates_each_seed_to_each_real_configuration(self) -> None:
        plan = benchmark.build_experiment_plan([7, 11])
        self.assertEqual(len(plan), len(benchmark.DEFAULT_MATRIX) * 2)
        for config in benchmark.DEFAULT_MATRIX:
            matching = [case for case in plan if case.scenario == config["scenario"]]
            self.assertEqual([case.seed for case in matching], [7, 11])
            self.assertTrue(all(case.difficulty == config["difficulty"] for case in matching))
            self.assertTrue(all(case.attacker == config["attacker"] for case in matching))

    def test_random_is_reproducible_for_same_case(self) -> None:
        case = benchmark.ExperimentCase("test", "bank", "hard", "ransomware_gang", 19)
        policies = benchmark.PolicySet(["random"], benchmark.DEFAULT_PPO_PATH)
        first = benchmark.run_episode(agent="random", episode_index=0, case=case, max_steps=8, policies=policies)
        second = benchmark.run_episode(agent="random", episode_index=0, case=case, max_steps=8, policies=policies)
        self.assertEqual(first.requested_actions, second.requested_actions)
        self.assertEqual(first.executed_actions, second.executed_actions)
        self.assertEqual(first.total_reward, second.total_reward)
        self.assertEqual(first.final_network_risk, second.final_network_risk)
        self.assertEqual(first.attempted_attacks, first.successful_attacks + first.prevented_attacks)

    def test_episode_reset_prevents_state_leakage(self) -> None:
        case = benchmark.ExperimentCase("test", "saas", "hard", "silent_apt", 23)
        policies = benchmark.PolicySet(["heuristic"], benchmark.DEFAULT_PPO_PATH)
        first = benchmark.run_episode(agent="heuristic", episode_index=0, case=case, max_steps=6, policies=policies)
        second = benchmark.run_episode(agent="heuristic", episode_index=1, case=case, max_steps=6, policies=policies)
        self.assertEqual(first.requested_actions, second.requested_actions)
        self.assertEqual(first.total_reward, second.total_reward)
        self.assertEqual(first.successful_attacks, second.successful_attacks)

    def test_aggregation_reports_population_mean_std_and_count(self) -> None:
        case_a = benchmark.ExperimentCase("a", "bank", "hard", "ransomware_gang", 3)
        case_b = benchmark.ExperimentCase("b", "bank", "hard", "ransomware_gang", 4)
        policies = benchmark.PolicySet(["random"], benchmark.DEFAULT_PPO_PATH)
        rows = [
            benchmark.run_episode(agent="random", episode_index=0, case=case_a, max_steps=4, policies=policies),
            benchmark.run_episode(agent="random", episode_index=1, case=case_b, max_steps=4, policies=policies),
        ]
        summary = benchmark.aggregate_results(rows)[0]
        expected_mean = (rows[0].total_reward + rows[1].total_reward) / 2.0
        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["total_reward_mean"], expected_mean)
        self.assertGreaterEqual(summary["total_reward_std"], 0.0)

    def test_ppo_identity_and_metadata_are_exact(self) -> None:
        expected_hash = "4D4955993DD2D98CC3B8D3319C1CDA92F10EFBA95407B3D78D9763E5D968D1AF"
        plan = benchmark.build_experiment_plan([42], benchmark.DEFAULT_MATRIX[:1])
        metadata = benchmark.build_metadata(
            agents=[benchmark.PPO_AGENT],
            seeds=[42],
            plan=plan,
            ppo_path=benchmark.DEFAULT_PPO_PATH,
            max_steps=5,
            completed_count=0,
            failure_count=0,
        )
        self.assertEqual(metadata["ppo"]["agent_id"], "ppo_existing_checkpoint")
        self.assertEqual(metadata["ppo"]["sha256"], expected_hash)
        self.assertFalse(metadata["ppo"]["committed"])

        policies = benchmark.PolicySet([benchmark.PPO_AGENT], benchmark.DEFAULT_PPO_PATH)
        row = benchmark.run_episode(
            agent=benchmark.PPO_AGENT,
            episode_index=0,
            case=plan[0],
            max_steps=3,
            policies=policies,
        )
        self.assertEqual(set(json.loads(row.underlying_agents)), {benchmark.PPO_AGENT})

    def test_failures_are_recorded_and_not_counted_as_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(benchmark.PolicySet, "decide", side_effect=RuntimeError("agent failed")):
                result = benchmark.run_benchmark(
                    agents=["heuristic"],
                    seeds=[1],
                    matrix=benchmark.DEFAULT_MATRIX[:1],
                    max_steps=2,
                    outdir=Path(temp_dir),
                )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_episode_count"], 0)
            self.assertEqual(result["failure_count"], 1)
            with (Path(temp_dir) / "failures.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_raw_episode_count_matches_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = benchmark.run_benchmark(
                agents=["random", "heuristic"],
                seeds=[5, 6],
                matrix=benchmark.DEFAULT_MATRIX[:1],
                max_steps=3,
                outdir=Path(temp_dir),
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["completed_episode_count"], 4)
            with (Path(temp_dir) / "episodes.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
