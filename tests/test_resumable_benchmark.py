from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.train import benchmark_agents as benchmark


class ResumableBenchmarkTests(unittest.TestCase):
    matrix = benchmark.DEFAULT_MATRIX[:1]

    @staticmethod
    def _run(outdir: Path, **overrides):
        options = {
            "agents": ["random"],
            "seeds": [1, 2],
            "matrix": ResumableBenchmarkTests.matrix,
            "max_steps": 2,
            "outdir": outdir,
            "resume": True,
        }
        options.update(overrides)
        return benchmark.run_benchmark(**options)

    def test_interruption_persists_completed_episode_and_resume_skips_it(self) -> None:
        original = benchmark.run_episode
        calls = 0

        def interrupt_after_first(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("runtime disconnected")
            return original(**kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            with patch.object(benchmark, "run_episode", side_effect=interrupt_after_first):
                with self.assertRaises(KeyboardInterrupt):
                    self._run(outdir)

            with (outdir / "episodes.csv").open(encoding="utf-8") as handle:
                first_rows = list(csv.DictReader(handle))
            self.assertEqual(len(first_rows), 1)
            self.assertEqual(json.loads((outdir / "metadata.json").read_text())["status"], "incomplete")

            resumed_calls = []

            def observe_resume(**kwargs):
                resumed_calls.append((kwargs["agent"], kwargs["case"].case_id, kwargs["case"].seed))
                return original(**kwargs)

            with patch.object(benchmark, "run_episode", side_effect=observe_resume):
                result = self._run(outdir)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(resumed_calls), 1)
            with (outdir / "episodes.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            keys = [(row["agent"], row["case_id"], int(row["seed"])) for row in rows]
            self.assertEqual(len(keys), 2)
            self.assertEqual(len(keys), len(set(keys)))

    def test_completed_resume_never_reruns_or_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            self.assertEqual(self._run(outdir)["status"], "ok")
            with patch.object(benchmark, "run_episode") as run_episode:
                self.assertEqual(self._run(outdir)["status"], "ok")
                run_episode.assert_not_called()
            with (outdir / "episodes.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_incompatible_resume_is_refused_before_agent_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            self.assertEqual(self._run(outdir)["status"], "ok")
            with patch.object(benchmark, "PolicySet") as policy_set:
                with self.assertRaisesRegex(RuntimeError, "Refusing incompatible resume"):
                    self._run(outdir, max_steps=3)
                policy_set.assert_not_called()

    def test_model_revision_change_is_an_incompatible_resume(self) -> None:
        model_a = {
            "model_id": "owner/model",
            "resolved_revision": "a" * 40,
            "dtype": "float16",
            "quantization": False,
            "device": "cuda:0",
        }
        model_b = dict(model_a, resolved_revision="b" * 40)
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            with patch.object(benchmark, "PolicySet"), patch.object(benchmark, "run_episode", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self._run(outdir, agents=["qwen_rl"], seeds=[1], model_metadata=model_a)
            with patch.object(benchmark, "PolicySet") as policy_set:
                with self.assertRaisesRegex(RuntimeError, "Refusing incompatible resume"):
                    self._run(outdir, agents=["qwen_rl"], seeds=[1], model_metadata=model_b)
                policy_set.assert_not_called()

    def test_failures_are_preserved_separately_and_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            with patch.object(benchmark, "run_episode", side_effect=RuntimeError("inference failed")):
                result = self._run(outdir, seeds=[1])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_episode_count"], 0)
            with (outdir / "failures.csv").open(encoding="utf-8") as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["error"], "inference failed")
            metadata = json.loads((outdir / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "incomplete")
            self.assertEqual(metadata["failure_count"], 1)

    def test_atomic_outputs_leave_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            with patch.object(benchmark.os, "replace", wraps=benchmark.os.replace) as atomic_replace:
                self.assertEqual(self._run(outdir, seeds=[1])["status"], "ok")
            self.assertGreaterEqual(atomic_replace.call_count, 3)
            self.assertFalse([path for path in outdir.iterdir() if path.suffix == ".tmp"])
            json.loads((outdir / "metadata.json").read_text())
            with (outdir / "episodes.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_explicit_qwen_failure_is_not_recorded_as_a_qwen_decision(self) -> None:
        model = {
            "model_id": "owner/model",
            "resolved_revision": "a" * 40,
            "dtype": ["torch.float16"],
            "quantization": False,
            "device": {"": "cuda:0"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir)
            with patch.object(benchmark, "PolicySet"), patch.object(
                benchmark, "run_episode", side_effect=RuntimeError("Qwen completion failed")
            ):
                result = self._run(outdir, agents=["qwen_rl"], seeds=[1], model_metadata=model)
            self.assertEqual(result["completed_episode_count"], 0)
            with (outdir / "episodes.csv").open(encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with (outdir / "failures.csv").open(encoding="utf-8") as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual(failures[0]["agent"], "qwen_rl")
            self.assertEqual(failures[0]["error"], "Qwen completion failed")

    def test_hybrid_source_and_fallback_attribution_is_truthful(self) -> None:
        class Policies:
            decisions = iter([
                (0, "hf_llm_agent", None),
                (0, "ppo_agent", None),
                (0, "ppo_agent", "HF prediction failed (RuntimeError); used PPO fallback."),
            ])

            def reset_episode(self, agent, seed):
                return None

            def decide(self, agent, env, obs, info):
                return next(self.decisions)

        case = benchmark.ExperimentCase("source", "bank", "hard", "ransomware_gang", 9)
        row = benchmark.run_episode(agent="hybrid_router", episode_index=0, case=case, max_steps=3, policies=Policies())
        self.assertEqual(row.qwen_decisions, 1)
        self.assertEqual(row.ordinary_ppo_decisions, 1)
        self.assertEqual(row.qwen_failures, 1)
        self.assertEqual(row.ppo_fallbacks, 1)
        self.assertEqual(row.fallback_count, 1)
        self.assertEqual(len(json.loads(row.fallback_reasons)), 1)
        self.assertEqual(json.loads(row.underlying_agents), ["hf_llm_agent", "ppo_agent", "ppo_agent"])

    def test_frozen_p1_p2_evidence_is_not_modified(self) -> None:
        frozen = [
            benchmark.PROJECT_ROOT / "outputs/benchmarks/p1_baseline_v2",
            benchmark.PROJECT_ROOT / "outputs/benchmarks/p2_fresh_ppo_primary",
        ]
        before = {path: path.stat().st_mtime_ns for root in frozen for path in root.rglob("*") if path.is_file()}
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(self._run(Path(temp_dir), seeds=[1])["status"], "ok")
        after = {path: path.stat().st_mtime_ns for root in frozen for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
