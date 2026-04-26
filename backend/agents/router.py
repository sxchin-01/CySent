from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from backend.agents.hf_agent import HFAgent
from backend.agents.ppo_agent import PPOAgent


VALID_AGENT_NAMES = {"ppo_agent", "hf_llm_agent"}


class AgentRouter:
    """Router for managing PPO and HF LLM agents with fallback logic."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or self._load_config()
        self.default_agent = self.config.get("default_agent", "ppo_agent")
        self.mode = str(self.config.get("mode", os.getenv("AGENT_MODE", "hybrid"))).lower()

        # Initialize agents
        self.ppo_agent: Optional[PPOAgent] = None
        self.hf_agent: Optional[HFAgent] = None
        self._initialize_agents()

        # Credit saving modes
        self.full_llm = bool(self.config.get("full_llm", False))
        self.hybrid_threshold = int(self.config.get("hybrid_threshold", 10))  # Every N turns for hybrid
        self.turn_counter = 0
        self.last_used_agent = "ppo_agent"

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

    def _initialize_agents(self) -> None:
        """Initialize available agents."""
        try:
            self.ppo_agent = PPOAgent()
        except Exception:
            self.ppo_agent = None
            print("[AgentRouter] PPO unavailable at startup.")

        try:
            self.hf_agent = HFAgent(timeout=float(self.config.get("hf_timeout", os.getenv("HF_TIMEOUT", 10.0))))
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

    def predict_action(self, observation: Any, state: Dict[str, Any]) -> int:
        """Route action prediction to appropriate agent with fallback."""
        network_risk = state.get("network_risk", 0.0)
        self.turn_counter += 1

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
                self.last_used_agent = "hf_llm_agent"
                return action
            except Exception as exc:
                # Fallback to PPO on HF failure
                if self.default_agent == "hf_llm_agent":
                    print(f"[AgentRouter] HF predict failed in explicit hf_llm_agent mode: {type(exc).__name__}: {exc}")
                    raise
                pass

        # Use PPO (or fallback to random if PPO unavailable)
        if self.ppo_agent and self.ppo_agent.is_available():
            try:
                self.last_used_agent = "ppo_agent"
                return self.ppo_agent.predict_action(observation, deterministic=True)
            except Exception:
                pass

        # Final fallback: deterministic safe no-op when PPO is unavailable.
        self.last_used_agent = "ppo_agent"
        return 0

    async def predict_action_async(self, observation: Any, state: Dict[str, Any]) -> int:
        """Async version of predict_action with proper HF handling."""
        network_risk = state.get("network_risk", 0.0)
        self.turn_counter += 1

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
                self.last_used_agent = "hf_llm_agent"
                return action
            except Exception as exc:
                # Fallback to PPO on HF failure
                if self.default_agent == "hf_llm_agent":
                    print(f"[AgentRouter] HF async predict failed in explicit hf_llm_agent mode: {type(exc).__name__}: {exc}")
                    raise
                pass

        # Use PPO (or fallback to random if PPO unavailable)
        if self.ppo_agent and self.ppo_agent.is_available():
            try:
                # PPO prediction is synchronous, run in thread pool
                action = await asyncio.get_event_loop().run_in_executor(
                    None, self.ppo_agent.predict_action, observation, True
                )
                self.last_used_agent = "ppo_agent"
                return action
            except Exception:
                pass

        self.last_used_agent = "ppo_agent"
        return 0

    def get_active_agent_name(self) -> str:
        """Get the name of the currently active agent for UI display."""
        if self.last_used_agent == "hf_llm_agent":
            if self.hf_agent is not None:
                return self.hf_agent.deployment_label()
            return "HF LLM Defender"
        else:
            return "PPO Defender"

    def is_agent_available(self, agent_name: str) -> bool:
        """Check if a specific agent is available."""
        if agent_name == "ppo_agent":
            return self.ppo_agent is not None and self.ppo_agent.is_available()
        elif agent_name == "hf_llm_agent":
            return self.hf_agent is not None and self.hf_agent.is_available()
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
        if agent_name not in ["ppo_agent", "hf_llm_agent"]:
            return False

        if not self.is_agent_available(agent_name):
            return False

        self.default_agent = agent_name
        if agent_name == "ppo_agent":
            # Explicit PPO selection should never route through HF.
            self.mode = "ppo_only"
            self.full_llm = False
        elif agent_name == "hf_llm_agent":
            self.mode = "full_llm"
            self.full_llm = True
        return True
        return False