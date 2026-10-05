# CySent v1 Experiments and Evidence

## 1. Experimental Philosophy

CySent evaluates policy behavior, not only cumulative reward. Each controlled
episode records security outcomes, operational outcomes, requested and
executed actions, and artifact/source provenance. No result below establishes
statistical significance or broad real-network validity.

Generated evidence files are frozen records and must not be hand-edited.

## 2. Frozen Evaluation Matrix

The P1 and P2 controlled evaluations use:

- Seeds: 42, 43, 44
- Bank / hard / `ransomware_gang`
- SaaS / hard / `silent_apt`
- Hospital / medium / `insider_saboteur`
- Maximum episode length: 150

Each policy therefore has nine episodes. Aggregates are arithmetic means and
population standard deviations over those nine rows, with no outlier removal.

## 3. P1 Baseline Methodology

P1 compares Random, deterministic Heuristic, and Historical PPO. It completed
27/27 episodes with zero failures. Raw rows, action distributions, source
hashes, matrix metadata, and reports are tracked at:

`outputs/benchmarks/p1_baseline_v2/`

The benchmark was executed from a recorded dirty working tree and later frozen
at P0/P1 commit `9f7b30b01d77165f00151ab75b915db671fcf1ae`.
Metadata includes hashes for the environment, reward, RED engine, policies,
runner, config, and Historical PPO checkpoint.

## 4. P1 Results

Values are mean +/- population standard deviation over nine episodes.

| Policy | Reward | Breach | Mean risk | Final risk | Uptime | Cost | Survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 58.1793 +/- 41.5480 | 0.1111 +/- 0.0898 | 0.1112 +/- 0.0207 | 0.1066 +/- 0.0580 | 0.9054 +/- 0.0141 | 16.0544 +/- 1.8610 | 144.56 +/- 15.40 |
| Heuristic | 71.3276 +/- 83.5035 | 0.1587 +/- 0.1837 | 0.1754 +/- 0.0620 | 0.2356 +/- 0.1622 | 0.9320 +/- 0.0774 | 10.8656 +/- 3.2798 | 113.22 +/- 43.32 |
| Historical PPO | -41.3488 +/- 39.3746 | 0.2063 +/- 0.0710 | 0.2731 +/- 0.0210 | 0.3821 +/- 0.0448 | 0.8319 +/- 0.0862 | 6.6844 +/- 2.7451 | 111.11 +/- 45.89 |

Heuristic has the highest mean reward and uptime in this matrix. Random has
lower breach/risk values and longer survival. These mixed outcomes are why a
reward-only ranking is not a sufficient security conclusion.

## 5. P1 Action Behavior

Historical PPO requested `investigate_top_alert` on 1000/1000 decisions.
Executed actions were 502 investigations and 498 `do_nothing` substitutions.
The checkpoint's deterministic behavior is therefore action-collapsed in this
matrix, while the environment's constraints materially alter execution.

The artifact is compatible with the current environment and has SHA-256:

`4D4955993DD2D98CC3B8D3319C1CDA92F10EFBA95407B3D78D9763E5D968D1AF`

Its exact original training-source lineage is incomplete.

## 6. P2 Fresh PPO Training

Fresh PPO used Stable-Baselines3 PPO with `MlpPolicy`, seed 42, four CPU
environments, the 67-value observation, 12 actions, and 150 maximum episode
steps. It requested 100,000 timesteps and completed 100,352. VecNormalize was
disabled. The resolved hyperparameters are frozen in `configs/v1_locked.yaml`.

The final training output and evaluation-selected best checkpoint are distinct:

- Final model: `61AB713E312C9A4ED79ABCC9C8130F0E1FFEE5FE989EAD63F6C5B12275917B6E`
- Best checkpoint: `7BA9122F3AE4BD67E539EC3BEE95EB587F63F22091605DDEF044190586DC0197`

The controlled benchmark uses the best checkpoint. Neither PPO binary is
stored in Git.

## 7. P2 Controlled Results

P2 adds Fresh PPO to the frozen matrix, producing 36/36 completed episodes and
zero failures. Evidence is tracked at:

`outputs/benchmarks/p2_fresh_ppo_primary/`

Fresh PPO values are mean +/- population standard deviation over nine
episodes:

| Reward | Breach | Mean risk | Final risk | Uptime | Cost | Survival |
|---:|---:|---:|---:|---:|---:|---:|
| -37.5918 +/- 41.6438 | 0.1905 +/- 0.0952 | 0.2648 +/- 0.0227 | 0.3507 +/- 0.0644 | 0.8375 +/- 0.0930 | 6.9178 +/- 2.9565 | 110.89 +/- 45.75 |

P1 rows were rerun unchanged in the P2 directory and match the authoritative
P1-v2 aggregate values.

## 8. Fresh PPO Action Concentration

Fresh PPO requested:

- `investigate_top_alert`: 967/998 (96.894%)
- `increase_monitoring`: 31/998

Executed actions were:

- `investigate_top_alert`: 490
- `do_nothing`: 477
- `increase_monitoring`: 31

Rate aggregation matters. Episode-mean rates give each episode equal weight;
pooled rates give each decision or repeat opportunity equal weight.

| Aggregation | Substitution | Repeat |
|---|---:|---:|
| Episode mean (frozen headline) | 48.16% | 98.61% |
| Pooled | 47.80% | 98.18% |

The deterministic benchmark policy is highly concentrated and near-collapsed.
This is a behavioral finding, not proof of one training defect.

## 9. PPO Diagnostic Interpretation

Relevant observations and hypotheses are:

- Stochastic policy sampling retained support for all 12 actions.
- Deterministic argmax was narrowly dominated by `investigate_top_alert`.
- The 67-value PPO observation omits alerts and cooldowns.
- A blocked/repeated investigation can execute as `do_nothing`.
- Training and benchmark scenario distributions differ.
- Critic/explained-variance evidence was weak in the training diagnostics.

These factors can interact. The evidence does not isolate one definitive
cause, and the benchmark should not be described as a causal ablation.

## 10. Historical Qwen Policy Reconstruction

Historical RL training did not use free-text generation and action parsing.
The faithful contract is:

```text
historical prompt
-> tokenize with truncation, max_length=256
-> model forward pass
-> final next-token vocabulary logits
-> gather first token ID for each of 12 action names
-> 12 constrained logits
-> categorical sample
-> environment action
```

Policy identity is `historical_first_token_seeded_categorical_v1`.

## 11. First-Token Mapping and Collision

The historical Qwen2.5 action-token IDs, in canonical action order, are:

```text
2982, 3400, 3400, 3400, 16213, 285,
78501, 30804, 35794, 759, 42014, 23169
```

There are 12 categories but 10 unique token IDs. `patch_hr_systems`,
`patch_web_server`, and `patch_auth_server` all map to token `3400`, so their
historically faithful constrained logits and probabilities are identical.
Historical training still treated them as three categorical entries.

Evaluation preserves the collision rather than changing the v1 policy:

- Raw constrained logits retain the model's inference dtype (FP16 on the real
  T4 smoke).
- The 12 gathered logits are converted to FP32 before softmax.
- Sampling uses an agent-local CPU `torch.Generator` reset from episode seed.
- Probabilities are conditional on the historical 12-action constrained
  policy, not calibrated probabilities over every possible output.
- Evaluation is distribution-faithful, not claimed to reproduce historical
  floating-point sampling bit for bit.

The collision is a substantive representation limitation. It does not by
itself explain all Qwen policy behavior.

## 12. Real Qwen Smoke Evidence

The canonical merged artifact is:

- Repository: `sxchin01/CySent-Qwen-RL-merged`
- Immutable revision: `fb75512b037bb37de575916afd900c03ab860cb5`

A real T4 smoke succeeded on all three frozen scenarios with seed 42. Each
case loaded the real merged model locally, used source `qwen_rl_policy`,
selected a valid environment action, reproduced the patch collision, and
advanced the environment once. All selected `patch_auth_server` under that
seeded smoke.

This is correctness evidence for artifact loading, faithful scoring, seeded
sampling, and environment integration. It is not a controlled performance
comparison. The evaluation notebook is tracked without outputs; the smoke
record is not independently preserved as a tracked result directory.

## 13. Hybrid Implementation and Status

Hybrid ordinarily routes to Historical PPO. It invokes Qwen when risk is
greater than `0.7` or on the configured periodic turn (currently every 10th
decision). If a valid Hybrid runtime reaches Qwen and that prediction fails,
it may use Historical PPO while recording a fallback reason. Missing startup
dependencies make Hybrid unavailable.

Automated routing, source attribution, fallback, injection, and reset behavior
are tested. Real-model smoke repeatedly caused abrupt Colab kernel termination.
No Python exception or explicit host/CUDA OOM was captured; the cause remains
unresolved.

No controlled Qwen or Hybrid benchmark has completed, and no comparative
performance claim is supported.

## 14. Artifact and Lineage Evidence

| Artifact | Identity | Verification | Git status | Lineage confidence |
|---|---|---|---|---|
| Historical PPO checkpoint | `ppo_historical_checkpoint` | Exact SHA-256 | Binary not stored | Binary identity verified; original training lineage incomplete |
| Fresh PPO best checkpoint | `ppo_fresh_checkpoint` | Exact SHA-256 | Binary not stored | Training config/source metadata and benchmark identity supported |
| Fresh PPO final model | Training output only | Exact SHA-256 in training record | Binary not stored | Distinct from benchmark-selected best checkpoint |
| Qwen SFT adapter | `sxchin01/CySent-adapter` | No frozen revision in v1 manifest | External | Supported but incomplete |
| Qwen RL adapter | `sxchin01/CySent-Qwen-RL` | No frozen revision in v1 manifest | External | Supported but incomplete |
| Qwen merged model | `sxchin01/CySent-Qwen-RL-merged@fb75512...` | Immutable revision and canonical snapshot layout | External | Artifact identity verified; upstream lineage incomplete |

Canonical Qwen artifact identity does not constitute cryptographic proof that
the public merged model descends from the exact historical pictured training
run.

## 15. Evidence Limitations

- The fixed matrix is small and simulation-specific.
- P1/P2 runs record dirty-tree metadata; source hashes and later freeze commits
  preserve the evaluated files.
- Historical PPO training provenance is incomplete.
- Fresh PPO binaries are external despite tracked hashes and result evidence.
- Qwen upstream SFT/RL adapter revisions are not frozen in the v1 manifest.
- The real Qwen smoke has no tracked output artifact.
- Reward is shaped and must not replace direct security/operational metrics.
- No significance tests, confidence intervals, or broad generalization claims
  are reported.

## 16. Incomplete Experiments

- Controlled Qwen benchmark
- Controlled Hybrid benchmark
- Resolved real-model Hybrid smoke
- OOD/generalization evaluation
- Reward and observation ablations
- Larger seed/scenario study
