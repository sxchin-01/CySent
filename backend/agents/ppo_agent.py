from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

from backend.artifacts import artifact_entry, artifact_path, verify_local_artifact


CUSTOM_PPO_IDENTITY = "ppo_custom_checkpoint"


class PPOAgent:
    """PPO agent for action prediction using trained model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        artifact_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if model_path is None:
            artifact_id = artifact_id or "historical_ppo"
            resolved_path = artifact_path(artifact_id)
        else:
            resolved_path = Path(model_path)
        self.model_path = str(resolved_path)
        self.artifact_id = artifact_id
        self.identity = (
            str(artifact_entry(artifact_id)["agent_identity"])
            if artifact_id is not None
            else CUSTOM_PPO_IDENTITY
        )
        self.device = device
        self.model: Optional[PPO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the PPO model from disk."""
        if PPO is None:
            raise ImportError("stable_baselines3 not installed. Cannot use PPO agent.")

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"PPO model not found at configured path: {self.model_path}")

        if self.artifact_id is not None:
            verify_local_artifact(self.artifact_id, model_file)

        if self.device is None:
            self.model = PPO.load(str(model_file))
        else:
            self.model = PPO.load(str(model_file), device=self.device)

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
