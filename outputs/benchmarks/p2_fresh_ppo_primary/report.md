# CySent P1 Baseline Benchmark

This report contains only post-P0 executions from this output directory.

## Experiment

- Run UTC: 2026-10-03T12:59:57.502797+00:00
- Git commit: 4a0574ee601f229de9c4c9957cdb3d1545b06ee6
- Dirty working tree: True
- Seeds: 42, 43, 44
- Maximum episode length: 150
- Raw completed episodes: 36

## Aggregate Results

| Agent | Reward | Breach | Mean Risk | Final Risk | Uptime | Action Cost | Survival | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 58.1793 +/- 41.5480 | 0.1111 +/- 0.0898 | 0.1112 +/- 0.0207 | 0.1066 +/- 0.0580 | 0.9054 +/- 0.0141 | 16.0544 +/- 1.8610 | 144.5556 +/- 15.3992 | 9 |
| heuristic | 71.3276 +/- 83.5035 | 0.1587 +/- 0.1837 | 0.1754 +/- 0.0620 | 0.2356 +/- 0.1622 | 0.9320 +/- 0.0774 | 10.8656 +/- 3.2798 | 113.2222 +/- 43.3174 | 9 |
| ppo_existing_checkpoint | -41.3488 +/- 39.3746 | 0.2063 +/- 0.0710 | 0.2731 +/- 0.0210 | 0.3821 +/- 0.0448 | 0.8319 +/- 0.0862 | 6.6844 +/- 2.7451 | 111.1111 +/- 45.8898 | 9 |
| ppo_fresh_checkpoint | -37.5918 +/- 41.6438 | 0.1905 +/- 0.0952 | 0.2648 +/- 0.0227 | 0.3507 +/- 0.0644 | 0.8375 +/- 0.0930 | 6.9178 +/- 2.9565 | 110.8889 +/- 45.7492 | 9 |

## Action Rate Aggregation

Episode mean gives every episode equal weight. Pooled rate gives every decision (or repeat opportunity) equal weight.

| Agent | Episode-Mean Substitution | Pooled Substitution | Episode-Mean Repeat | Pooled Repeat | Decisions |
|---|---:|---:|---:|---:|---:|
| random | 2.1056% | 1.9985% | 7.7830% | 7.5851% | 1301 |
| heuristic | 0.0000% | 0.0000% | 24.5650% | 18.4158% | 1019 |
| ppo_existing_checkpoint | 49.6877% | 49.8000% | 100.0000% | 100.0000% | 1000 |
| ppo_fresh_checkpoint | 48.1588% | 47.7956% | 98.6102% | 98.1800% | 998 |

## Metric Definitions

- `total_reward` (higher is better): Sum of environment rewards across the episode.
- `breach_rate` (lower is better): Final compromised assets divided by total assets.
- `successful_attacks` (lower is better): Count of RED attempts whose red_log.success is true.
- `prevented_attacks` (higher is better): Count of scheduled RED attempts that failed; no-attack turns are excluded.
- `prevention_rate` (higher is better): Prevented attacks divided by scheduled attack attempts; zero when there were no attempts.
- `compromised_assets` (lower is better): Count of compromised assets in the final state.
- `critical_compromised_assets` (lower is better): Final compromised assets listed as critical by the active scenario.
- `mean_network_risk` (lower is better): Arithmetic mean of post-transition network risk over all turns.
- `final_network_risk` (lower is better): Network risk after the final transition.
- `peak_network_risk` (lower is better): Maximum initial or post-transition network risk.
- `mean_uptime` (higher is better): Mean per-turn fraction of assets with uptime_status=true.
- `final_uptime` (higher is better): Final fraction of assets with uptime_status=true.
- `mean_downtime` (lower is better): One minus mean uptime.
- `action_cost` (lower is better): Sum of action_cost reported by reward_breakdown.
- `substituted_actions` (lower is better): Requested actions replaced by a different executed action due to environment constraints.
- `substitution_rate` (lower is better): Fraction of requested actions replaced by a different executed action.
- `repeated_actions` (lower is better): Consecutive requested actions equal to the preceding requested action.
- `repeated_action_rate` (lower is better): Fraction of decisions after the first that repeat the preceding requested action.
- `survival_turns` (higher is better): Number of transitions before termination or truncation.
- `decision_latency_ms` (lower is better): Mean wall-clock policy decision latency per turn in milliseconds.
- `fallback_count` (lower is better): Decisions where a router reported fallback to another agent.

## Reliability

- No agent failures or timeouts were recorded.

## Limitations

- The historical PPO checkpoint is local, uncommitted, and has incomplete exact training-source provenance.
- Population standard deviation is reported over this fixed benchmark matrix; the sample is not a claim of broad external validity.
- Wasteful-action count is not reported because the environment does not expose that reward-internal classification reliably.
