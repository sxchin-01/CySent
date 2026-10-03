from __future__ import annotations

import random as _random

from backend.env.security_env import ACTION_NAMES


class RandomAgent:
    """Uniformly random action agent used as a baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = _random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode with an agent-local random stream."""
        self._rng.seed(seed)

    def predict_action(self) -> int:
        return self._rng.randint(0, len(ACTION_NAMES) - 1)

    def is_available(self) -> bool:
        return True
