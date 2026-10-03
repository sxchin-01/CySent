from __future__ import annotations

import copy
import math
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.agents.router import AgentRouter
from backend.agents.ppo_agent import PPOAgent
from backend.api import main as api
from backend.env.security_env import ACTION_NAMES, CySentSecurityEnv
from backend.env.threat_engine import ThreatEngine


def disable_red(env: CySentSecurityEnv) -> None:
    env.threat_engine.choose_attack = Mock(
        return_value={
            "scheduled": False,
            "target_idx": None,
            "attack_type": "no_attack",
            "chain": {},
            "notes": "test pause",
        }
    )


class EnvironmentCorrectnessTests(unittest.TestCase):
    def test_reset_seed_and_observation_are_reproducible(self) -> None:
        env = CySentSecurityEnv(max_steps=8)
        obs_a, info_a = env.reset(seed=123)
        obs_b, info_b = env.reset(seed=123)

        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertTrue(env.observation_space.contains(obs_a))
        self.assertEqual(obs_a.dtype, np.float32)
        self.assertTrue(np.isfinite(obs_a).all())
        self.assertEqual(info_a["assets"], info_b["assets"])

        env.reset(seed=77)
        samples_a = [env.action_space.sample() for _ in range(8)]
        env.reset(seed=77)
        samples_b = [env.action_space.sample() for _ in range(8)]
        self.assertEqual(samples_a, samples_b)

    def test_every_canonical_blue_action_executes_and_reports_finite_state(self) -> None:
        self.assertEqual(set(ACTION_NAMES), set(range(12)))

        for action, action_name in ACTION_NAMES.items():
            with self.subTest(action=action_name):
                env = CySentSecurityEnv(max_steps=10)
                env.reset(seed=100 + action)
                disable_red(env)

                before = {asset["name"]: dict(asset) for asset in env.assets}
                if action == 5:
                    env._find_asset("HR Systems")["infected"] = True
                elif action == 7:
                    target = env._find_asset("HR Systems")
                    target.update({"uptime_status": False, "infected": True, "compromised": True})
                elif action == 10:
                    env._find_asset("Auth Server").update({"infected": True, "compromised": True})
                    env.current_alerts = [{
                        "id": "test-alert",
                        "target": "Auth Server",
                        "severity": "critical",
                        "severity_score": 1.0,
                        "true_positive": True,
                    }]

                obs, reward, _, _, info = env.step(action)
                self.assertEqual(info["last_action"], action_name)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertTrue(math.isfinite(reward))
                self.assertEqual(info["reward_breakdown"]["reward"], reward)

                if action in {1, 2, 3}:
                    targets = {1: "HR Systems", 2: "Web Server", 3: "Auth Server"}
                    target_name = targets[action]
                    while env.pending_effects:
                        env.step(0)
                    self.assertGreater(env._find_asset(target_name)["patch_level"], before[target_name]["patch_level"])
                elif action == 4:
                    self.assertLess(env._find_asset("Auth Server")["credential_risk"], before["Auth Server"]["credential_risk"])
                elif action == 5:
                    self.assertTrue(any(asset["isolated"] for asset in env.assets))
                elif action == 6:
                    self.assertGreater(env._find_asset("Auth Server")["detection_level"], before["Auth Server"]["detection_level"])
                elif action == 7:
                    self.assertTrue(env._find_asset("HR Systems")["uptime_status"])
                    self.assertFalse(env._find_asset("HR Systems")["compromised"])
                elif action == 8:
                    self.assertGreater(env.honeypot_timer, 0)
                elif action == 9:
                    self.assertLess(env._find_asset("Employee Email")["credential_risk"], before["Employee Email"]["credential_risk"])
                elif action == 10:
                    self.assertFalse(env._find_asset("Auth Server")["compromised"])
                elif action == 11:
                    env.step(0)
                    self.assertTrue(env.segmented_finance)
                    self.assertTrue(env._find_asset("Finance Database")["isolated"])

    def test_termination_truncation_and_attack_metrics(self) -> None:
        terminal_env = CySentSecurityEnv(max_steps=10)
        terminal_env.reset(seed=9)
        disable_red(terminal_env)
        terminal_env._find_asset("Auth Server")["compromised"] = True
        terminal_env._find_asset("Finance Database")["compromised"] = True
        _, _, terminated, truncated, info = terminal_env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "critical_breach")
        self.assertEqual(info["metrics"]["prevented_attacks"], 0)

        truncated_env = CySentSecurityEnv(max_steps=1)
        truncated_env.reset(seed=9)
        disable_red(truncated_env)
        _, _, terminated, truncated, _ = truncated_env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_delayed_effect_is_included_in_reward_transition(self) -> None:
        env = CySentSecurityEnv(max_steps=6)
        env.reset(seed=5)
        disable_red(env)
        env.pending_effects = [{"due_turn": 1, "effect": "segment_finance", "target": "Finance Database"}]
        _, reward, _, _, info = env.step(0)

        self.assertTrue(env.segmented_finance)
        self.assertGreater(info["reward_breakdown"]["risk_delta"], 0.0)
        self.assertTrue(math.isfinite(reward))


class ThreatEngineCorrectnessTests(unittest.TestCase):
    def test_seeded_attack_progression_is_reproducible_and_mutates_state(self) -> None:
        env_a = CySentSecurityEnv(max_steps=10)
        env_b = CySentSecurityEnv(max_steps=10)
        env_a.reset(seed=44, options={"attacker": "ransomware_gang", "difficulty": "hard"})
        env_b.reset(seed=44, options={"attacker": "ransomware_gang", "difficulty": "hard"})
        initial_assets = copy.deepcopy(env_a.assets)

        logs_a = []
        logs_b = []
        for _ in range(5):
            logs_a.append(env_a.step(0)[4]["red_log"])
            logs_b.append(env_b.step(0)[4]["red_log"])

        self.assertEqual(logs_a, logs_b)
        self.assertTrue(any(log["attack"] != "no_attack" for log in logs_a))
        self.assertNotEqual(initial_assets, env_a.assets)

    def test_profiles_change_seeded_red_and_scenario_behavior(self) -> None:
        ransomware = CySentSecurityEnv(max_steps=12)
        apt = CySentSecurityEnv(max_steps=12)
        bank_obs, _ = ransomware.reset(
            seed=91,
            options={"scenario": "bank", "difficulty": "hard", "attacker": "ransomware_gang"},
        )
        saas_obs, _ = apt.reset(
            seed=91,
            options={"scenario": "saas", "difficulty": "hard", "attacker": "silent_apt"},
        )
        self.assertFalse(np.array_equal(bank_obs, saas_obs))

        ransomware_attacks = [ransomware.step(0)[4]["red_log"]["attack"] for _ in range(8)]
        apt_attacks = [apt.step(0)[4]["red_log"]["attack"] for _ in range(8)]
        self.assertNotEqual(ransomware_attacks, apt_attacks)

    def test_episode_reset_clears_campaign_identity(self) -> None:
        engine = ThreatEngine(seed=1)
        engine._campaign_counter = 8
        engine._active_chain = {"chain_id": "stale"}
        engine.reset_episode_state()
        self.assertEqual(engine._campaign_counter, 0)
        self.assertIsNone(engine._active_chain)


class _FakeAgent:
    def __init__(self, action: int, *, available: bool = True, error: Exception | None = None) -> None:
        self.action = action
        self.available = available
        self.error = error
        self.calls = 0

    def is_available(self) -> bool:
        return self.available

    def predict_action(self, *args, **kwargs) -> int:
        self.calls += 1
        if self.error:
            raise self.error
        return self.action

    async def predict_action_async(self, *args, **kwargs) -> int:
        return self.predict_action(*args, **kwargs)

    def deployment_label(self) -> str:
        return "Test HF Defender"


class RoutingCorrectnessTests(unittest.TestCase):
    def make_router(self) -> AgentRouter:
        with patch.object(AgentRouter, "_initialize_agents", return_value=None):
            router = AgentRouter(config={"default_agent": "ppo_agent", "mode": "hybrid", "hybrid_threshold": 10})
        router.ppo_agent = _FakeAgent(3)
        router.hf_agent = _FakeAgent(7)
        return router

    def test_explicit_ppo_qwen_and_hybrid_are_truthful(self) -> None:
        router = self.make_router()
        self.assertTrue(router.switch_agent("ppo_agent"))
        self.assertEqual(router.predict_action(np.zeros(2), {"network_risk": 0.9}), 3)
        self.assertEqual(router.ppo_agent.calls, 1)
        self.assertEqual(router.hf_agent.calls, 0)

        self.assertTrue(router.switch_agent("hf_llm_agent"))
        self.assertEqual(router.predict_action(np.zeros(2), {"network_risk": 0.1}), 7)
        self.assertEqual(router.last_used_agent, "hf_llm_agent")

        self.assertTrue(router.switch_agent("hybrid"))
        self.assertEqual(router.predict_action(np.zeros(2), {"network_risk": 0.9}), 7)

    def test_failed_agents_are_surfaced_and_hybrid_fallback_is_observable(self) -> None:
        router = self.make_router()
        router.hf_agent = _FakeAgent(7, error=RuntimeError("provider down"))
        router.switch_agent("hf_llm_agent")
        with self.assertRaises(RuntimeError):
            router.predict_action(np.zeros(2), {"network_risk": 0.9})

        router.switch_agent("hybrid")
        self.assertEqual(router.predict_action(np.zeros(2), {"network_risk": 0.9}), 3)
        self.assertIn("HF prediction failed", router.last_fallback_reason or "")

        router.ppo_agent = None
        router.hf_agent = None
        with self.assertRaisesRegex(RuntimeError, "PPO agent is unavailable"):
            router.predict_action(np.zeros(2), {"network_risk": 0.1})


class PPOArtifactSmokeTests(unittest.TestCase):
    def test_expected_checkpoint_loads_and_completes_deterministic_smoke_episode(self) -> None:
        model_path = Path("backend/train/artifacts/best_model/best_model.zip")
        if not model_path.exists():
            self.skipTest(f"PPO checkpoint is not present at {model_path}")

        agent = PPOAgent(str(model_path))
        env = CySentSecurityEnv(max_steps=20, seed=42)
        obs, _ = env.reset(seed=42)
        self.assertEqual(agent.model.observation_space.shape, env.observation_space.shape)
        self.assertEqual(agent.model.action_space.n, env.action_space.n)

        terminated = truncated = False
        turns = 0
        while not (terminated or truncated):
            action = agent.predict_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            self.assertTrue(math.isfinite(reward))
            turns += 1

        self.assertGreater(turns, 0)
        self.assertLessEqual(turns, 20)


class ApiCorrectnessTests(unittest.TestCase):
    def make_runtime(self, max_steps: int = 5):
        env = CySentSecurityEnv(max_steps=max_steps, seed=12)
        _, info = env.reset(seed=12)
        disable_red(env)
        router = Mock()
        router.last_fallback_reason = None
        router.get_active_agent_name.return_value = "Test PPO Defender"
        rt = SimpleNamespace(
            env=env,
            agent_router=router,
            state_lock=threading.RLock(),
            last_info=info,
            current_episode_id="episode-test",
            episode_done=False,
            replays={},
        )
        rt.snapshot_state = lambda: {
            "episode_id": rt.current_episode_id,
            "step": rt.last_info.get("step", 0),
            "network_risk": rt.last_info.get("network_risk", 0.0),
            "risk_breakdown": rt.last_info.get("risk_breakdown", {}),
            "assets": rt.last_info.get("assets", []),
            "profile": rt.last_info.get("profile", {}),
            "intelligence": rt.last_info.get("intelligence", {}),
            "events": rt.last_info.get("events", []),
        }
        return rt

    def test_reset_endpoint_resets_state_and_rejects_unknown_agent(self) -> None:
        rt = self.make_runtime()
        rt.env.step(0)
        rt.agent_router.switch_agent.return_value = True
        with patch.object(api, "_get_runtime", return_value=rt):
            result = api.reset(api.ResetRequest(seed=33, action_source="random"), Mock())
            self.assertEqual(result["step"], 0)
            self.assertFalse(rt.episode_done)
            rt.agent_router.switch_agent.assert_called_once_with("random")

            with self.assertRaises(HTTPException) as invalid:
                api.reset(api.ResetRequest(seed=33, action_source="not-an-agent"), Mock())
            self.assertEqual(invalid.exception.status_code, 400)

    def test_manual_step_bypasses_router_and_validates_action_identity(self) -> None:
        rt = self.make_runtime()
        with patch.object(api, "_get_runtime", return_value=rt):
            result = api.step_manual(api.StepRequest(action=4, action_name="rotate_credentials"), Mock())
            self.assertEqual(result["action_mode"], "manual")
            self.assertEqual(result["action_name"], "rotate_credentials")
            self.assertEqual(result["active_agent"], "Manual Defender")
            rt.agent_router.predict_action.assert_not_called()

            with self.assertRaises(HTTPException) as mismatch:
                api.step_manual(api.StepRequest(action=4, action_name="restore_backup"), Mock())
            self.assertEqual(mismatch.exception.status_code, 400)

        with self.assertRaises(ValidationError):
            api.StepRequest(action=12)

    def test_fastapi_manual_route_smoke(self) -> None:
        rt = self.make_runtime()
        with patch.object(api, "_get_runtime", return_value=rt):
            with TestClient(api.app) as client:
                state_response = client.get("/state")
                step_response = client.post(
                    "/step/manual",
                    json={"action": 6, "action_name": "increase_monitoring"},
                )
                invalid_response = client.post("/step/manual", json={"action": 99})

        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(step_response.status_code, 200)
        self.assertEqual(step_response.json()["action_mode"], "manual")
        self.assertEqual(step_response.json()["action_name"], "increase_monitoring")
        self.assertEqual(invalid_response.status_code, 422)

    def test_autonomous_step_invokes_router_and_failure_preserves_state(self) -> None:
        rt = self.make_runtime()
        rt.agent_router.predict_action.return_value = 6
        with patch.object(api, "_get_runtime", return_value=rt):
            result = api.step(Mock())
        self.assertEqual(result["action_mode"], "autonomous")
        self.assertEqual(result["action_name"], "increase_monitoring")
        rt.agent_router.predict_action.assert_called_once()

        failed_rt = self.make_runtime()
        failed_rt.agent_router.predict_action.side_effect = RuntimeError("model failed")
        with patch.object(api, "_get_runtime", return_value=failed_rt):
            with self.assertRaises(HTTPException) as failure:
                api.step(Mock())
        self.assertEqual(failure.exception.status_code, 502)
        self.assertEqual(failed_rt.env.current_step, 0)

    def test_terminal_episode_remains_consistent_until_reset(self) -> None:
        rt = self.make_runtime(max_steps=1)
        result = api._execute_action(rt, 0, active_agent="Manual Defender", action_mode="manual")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["episode_id"], "episode-test")
        self.assertTrue(rt.episode_done)
        self.assertEqual(rt.last_info["step"], 1)
        with self.assertRaises(HTTPException) as complete:
            api._execute_action(rt, 0, active_agent="Manual Defender", action_mode="manual")
        self.assertEqual(complete.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
