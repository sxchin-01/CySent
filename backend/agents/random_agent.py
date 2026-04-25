from __future__ import annotations

import random as _random

from backend.env.security_env import ACTION_NAMES


class RandomAgent:
    """Uniformly random action agent used as a baseline."""

    def predict_action(self) -> int:
        return _random.randint(0, len(ACTION_NAMES) - 1)

    def is_available(self) -> bool:
        return True
