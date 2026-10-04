from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from backend.agents.hf_agent import HFAgent
from backend.agents.ppo_agent import PPOAgent
from backend.agents.qwen_rl_policy import QWEN_RL_SOURCE_ID, QwenRLPolicyAgent
from backend.agents.random_agent import RandomAgent


VALID_AGENT_NAMES = {"ppo_agent", "hf_llm_agent", "hybrid", "random", "random_agent"}
_AGENT_NOT_PROVIDED = object()


class AgentRouter:
    """Router for managing PPO and HF LLM agents with fallback logic."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        ppo_agent: Any = _AGENT_NOT_PROVIDED,
        hf_agent: Any = _AGENT_NOT_PROVIDED,
    ) -> None:
        self.config = config or self._load_config()
        self.default_agent = self.config.get("default_agent", "ppo_agent")
        self.mode = str(self.config.get("mode", os.getenv("AGENT_MODE", "hybrid"))).lower()

        # Initialize agents
        initialize_ppo = ppo_agent is _AGENT_NOT_PROVIDED
        initialize_hf = hf_agent is _AGENT_NOT_PROVIDED
        self.ppo_agent: Optional[PPOAgent] = None if initialize_ppo else ppo_agent
        self.hf_agent: Optional[HFAgent] = None if initialize_hf else hf_agent
        self.random_agent: RandomAgent = RandomAgent()
        self._initialize_agents(initialize_ppo=initialize_ppo, initialize_hf=initialize_hf)

        # Credit saving modes
        self.full_llm = bool(self.config.get("full_llm", False))
        self.hybrid_threshold = max(1, int(self.config.get("hybrid_threshold", 10)))  # Every N turns for hybrid
        self.turn_counter = 0
        self.last_used_agent = "ppo_agent"
        self.last_fallback_reason: Optional[str] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration."""
        try:
            import yaml
            config_path = "configs/agents.yaml"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
        except ImportError:
            pass
        return {}

    def _initialize_agents(self, *, initialize_ppo: bool = True, initialize_hf: bool = True) -> None:
        """Initialize available agents."""
        if initialize_ppo:
            try:
                self.ppo_agent = PPOAgent()
            except Exception:
                self.ppo_agent = None
                print("[AgentRouter] PPO unavailable at startup.")

        if initialize_hf:
            try:
                hf_agent_class = QwenRLPolicyAgent if self.config.get("hf_policy_mode") == "qwen_rl" else HFAgent
                self.hf_agent = hf_agent_class(timeout=float(self.config.get("hf_timeout", os.getenv("HF_TIMEOUT", 10.0))))
            except Exception as exc:
                self.hf_agent = None
                print(f"[AgentRouter] HF unavailable at startup: {type(exc).__name__}: {exc}")
                print("[AgentRouter] PPO remains default fallback.")

    def _should_use_hf(self, network_risk: float) -> bool:
        """Determine if HF should be used based on mode and risk."""
        if not self.hf_agent or not self.hf_agent.is_available():
            return False

        if self.full_llm:
            return True

        # Hybrid mode: HF on high risk or every N turns
        high_risk = network_risk > 0.7
        every_n_turns = (self.turn_counter % self.hybrid_threshold) == 0

        return high_risk or every_n_turns

    def _hf_source_id(self) -> str:
        return str(getattr(self.hf_agent, "source_id", "hf_llm_agent"))

    def predict_action(self, observation: Any, state: Dict[str, Any]) -> int:
        """Route action prediction to appropriate agent with fallback."""
        self.last_fallback_reason = None
        network_risk = state.get("network_risk", 0.0)
        self.turn_counter += 1

        if self.default_agent == "random_agent":
            self.last_used_agent = "random_agent"
            return int(self.random_agent.predict_action())

        # Forced source selection from API/UI takes precedence.
        if self.default_agent == "hf_llm_agent":
            use_hf = True
        elif self.default_agent == "ppo_agent":
            if self.mode == "full_llm" or self.full_llm:
                use_hf = True
            elif self.mode == "ppo_only":
                use_hf = False
            else:
                # Default hybrid behavior: PPO baseline with selective HF assists.
                use_hf = self._should_use_hf(network_risk)
        else:
            use_hf = self._should_use_hf(network_risk)

        # Ensure HF prompt context includes scenario/attacker when available.
        profile = state.get("profile", {}) if isinstance(state.get("profile", {}), dict) else {}
        if "scenario" not in state and "scenario" in profile:
            state = dict(state)
            state["scenario"] = profile.get("scenario")
            state["attacker"] = profile.get("attacker", state.get("attacker", "unknown"))

        if use_hf and self.hf_agent:
            try:
                action = self.hf_agent.predict_action(state)
                self.last_used_agent = self._hf_source_id()
                return action
            except Exception as exc:
                # Fallback to PPO on HF failure
                if self.default_agent == "hf_llm_agent":
                    print(f"[AgentRouter] HF predict failed in explicit hf_llm_agent mode: {type(exc).__name__}: {exc}")
                    raise
                self.last_fallback_reason = f"HF prediction failed ({type(exc).__name__}); used PPO fallback."

        # Use PPO (or fallback to random if PPO unavailable)
        if self.ppo_agent and self.ppo_agent.is_available():
            try:
                self.last_used_agent = "ppo_agent"
                return self.ppo_agent.predict_action(observation, deterministic=True)
            except Exception as exc:
                raise RuntimeError(f"PPO prediction failed ({type(exc).__name__}): {exc}") from exc

        raise RuntimeError("PPO agent is unavailable; no autonomous action was executed.")

    async def predict_action_async(self, observation: Any, state: Dict[str, Any]) -> int:
        """Async version of predict_action with proper HF handling."""
        self.last_fallback_reason = None
        network_risk = state.get("network_risk", 0.0)
        self.turn_counter += 1

        if self.default_agent == "random_agent":
            self.last_used_agent = "random_agent"
            return int(self.random_agent.predict_action())

        if self.default_agent == "hf_llm_agent":
            use_hf = True
        elif self.default_agent == "ppo_agent":
            if self.mode == "full_llm" or self.full_llm:
                use_hf = True
            elif self.mode == "ppo_only":
                use_hf = False
            else:
                use_hf = self._should_use_hf(network_risk)
        else:
            use_hf = self._should_use_hf(network_risk)

        if use_hf and self.hf_agent:
            try:
                action = await self.hf_agent.predict_action_async(state)
                self.last_used_agent = self._hf_source_id()
                return action
            except Exception as exc:
                # Fallback to PPO on HF failure
                if self.default_agent == "hf_llm_agent":
                    print(f"[AgentRouter] HF async predict failed in explicit hf_llm_agent mode: {type(exc).__name__}: {exc}")
                    raise
                self.last_fallback_reason = f"HF prediction failed ({type(exc).__name__}); used PPO fallback."

        # Use PPO (or fallback to random if PPO unavailable)
        if self.ppo_agent and self.ppo_agent.is_available():
            try:
                # PPO prediction is synchronous, run in thread pool
                action = await asyncio.get_event_loop().run_in_executor(
                    None, self.ppo_agent.predict_action, observation, True
                )
                self.last_used_agent = "ppo_agent"
                return action
            except Exception as exc:
                raise RuntimeError(f"PPO prediction failed ({type(exc).__name__}): {exc}") from exc

        raise RuntimeError("PPO agent is unavailable; no autonomous action was executed.")

    def get_active_agent_name(self) -> str:
        """Get the name of the currently active agent for UI display."""
        if self.last_used_agent in {"hf_llm_agent", QWEN_RL_SOURCE_ID}:
            if self.hf_agent is not None:
                return self.hf_agent.deployment_label()
            return "HF LLM Defender"
        if self.last_used_agent == "random_agent":
            return "Random Baseline"
        else:
            return "PPO Defender"

    def reset_episode(self, seed: Optional[int] = None) -> None:
        """Reset per-episode routing state and the Random policy stream."""
        self.turn_counter = 0
        self.last_fallback_reason = None
        self.random_agent.reset(seed)
        reset_hf = getattr(self.hf_agent, "reset", None)
        if callable(reset_hf):
            reset_hf(seed)

    def is_agent_available(self, agent_name: str) -> bool:
        """Check if a specific agent is available."""
        if agent_name == "ppo_agent":
            return self.ppo_agent is not None and self.ppo_agent.is_available()
        elif agent_name == "hf_llm_agent":
            return self.hf_agent is not None and self.hf_agent.is_available()
        elif agent_name in {"random", "random_agent", "hybrid"}:
            return True
        return False

    def set_mode(self, mode: str) -> None:
        """Set the agent mode (full_llm, hybrid, ppo_only)."""
        self.mode = mode
        if mode == "full_llm":
            self.full_llm = True
        elif mode == "hybrid":
            self.full_llm = False
        elif mode == "ppo_only":
            self.default_agent = "ppo_agent"
            self.full_llm = False

    def switch_agent(self, agent_name: str) -> bool:
        """Switch to a specific agent if available."""
        canonical = {
            "random": "random_agent",
            "random_agent": "random_agent",
            "hybrid": "hybrid",
            "ppo_agent": "ppo_agent",
            "hf_llm_agent": "hf_llm_agent",
        }.get(agent_name)

        if canonical is None:
            return False

        if canonical in {"ppo_agent", "hf_llm_agent"} and not self.is_agent_available(canonical):
            return False

        if canonical == "hybrid":
            self.default_agent = "ppo_agent"
            self.mode = "hybrid"
            self.full_llm = False
            return True

        if canonical == "random_agent":
            self.default_agent = "random_agent"
            self.mode = "ppo_only"
            self.full_llm = False
            return True

        self.default_agent = canonical
        if canonical == "ppo_agent":
            # Explicit PPO selection should never route through HF.
            self.mode = "ppo_only"
            self.full_llm = False
        elif canonical == "hf_llm_agent":
            self.mode = "full_llm"
            self.full_llm = True
        return True
