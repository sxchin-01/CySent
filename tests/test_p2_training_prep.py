from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.train import train_ppo


class TrainingIsolationTests(unittest.TestCase):
    def test_final_metrics_convert_numpy_bool_to_json_boolean(self) -> None:
        metrics = train_ppo._json_compatible({"model_load_verified": np.bool_(True)})

        self.assertIs(type(metrics["model_load_verified"]), bool)
        self.assertEqual(json.loads(json.dumps(metrics)), {"model_load_verified": True})

    def test_historical_checkpoint_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "historical PPO checkpoint"):
            train_ppo._validate_model_alias(str(train_ppo.HISTORICAL_PPO_PATH))

    def test_p1_baseline_output_is_rejected(self) -> None:
        target = train_ppo.P1_BASELINE_PATH / "fresh_model"
        with self.assertRaisesRegex(ValueError, "P1 baseline"):
            train_ppo._validate_model_alias(str(target))

    def test_existing_fresh_model_alias_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "fresh_model"
            target.with_suffix(".zip").write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "existing model alias"):
                train_ppo._validate_model_alias(str(target))


if __name__ == "__main__":
    unittest.main()
