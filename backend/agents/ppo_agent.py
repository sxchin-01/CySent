from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

from backend.env.security_env import CySentSecurityEnv


class PPOAgent:
    """PPO agent for action prediction using trained model."""

    def __init__(self, model_path: str = "backend/train/artifacts/best_model/best_model.zip") -> None:
        self.model_path = model_path
        self.model: Optional[PPO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the PPO model from disk."""
        if PPO is None:
            raise ImportError("stable_baselines3 not installed. Cannot use PPO agent.")

        model_file = Path(self.model_path)
        if not model_file.exists():
            # Try fallback to cysent_ppo.zip
            fallback_path = "backend/train/artifacts/cysent_ppo.zip"
            if Path(fallback_path).exists():
                model_file = Path(fallback_path)
            else:
                raise FileNotFoundError(f"PPO model not found at {self.model_path} or {fallback_path}")

        self.model = PPO.load(str(model_file))

    def predict_action(self, observation: Any, deterministic: bool = True) -> int:
        """Predict action from observation using PPO model.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic prediction

        Returns:
            Action ID (0-11)
        """
        if self.model is None:
            raise RuntimeError("PPO model not loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def is_available(self) -> bool:
        """Check if PPO agent is available for use."""
        return self.model is not None