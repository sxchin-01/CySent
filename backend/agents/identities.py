from __future__ import annotations

from typing import Dict


RANDOM_AGENT = "random"
HEURISTIC_AGENT = "heuristic"
HISTORICAL_PPO_AGENT = "ppo_historical_checkpoint"
FRESH_PPO_AGENT = "ppo_fresh_checkpoint"
QWEN_RL_AGENT = "qwen_rl"
HYBRID_AGENT = "hybrid_router"
LEGACY_HF_AGENT = "hf_generative_legacy"

CANONICAL_RESEARCH_AGENTS = (
    RANDOM_AGENT,
    HEURISTIC_AGENT,
    HISTORICAL_PPO_AGENT,
    FRESH_PPO_AGENT,
    QWEN_RL_AGENT,
    HYBRID_AGENT,
)

LIVE_AGENT_ALIASES: Dict[str, str] = {
    RANDOM_AGENT: RANDOM_AGENT,
    "random_agent": RANDOM_AGENT,
    HISTORICAL_PPO_AGENT: HISTORICAL_PPO_AGENT,
    "ppo_agent": HISTORICAL_PPO_AGENT,
    QWEN_RL_AGENT: QWEN_RL_AGENT,
    "hf_llm_agent": QWEN_RL_AGENT,
    HYBRID_AGENT: HYBRID_AGENT,
    "hybrid": HYBRID_AGENT,
    LEGACY_HF_AGENT: LEGACY_HF_AGENT,
}

LIVE_SELECTABLE_AGENTS = {
    RANDOM_AGENT,
    HISTORICAL_PPO_AGENT,
    QWEN_RL_AGENT,
    HYBRID_AGENT,
    LEGACY_HF_AGENT,
}
