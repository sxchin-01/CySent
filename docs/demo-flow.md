# CySent v1 Demo Flow

This is a five-to-seven-minute research demonstration. Lead with the artifact-free Random path and keep model availability visible.

## 1. Frame the System

CySent is an experimental cyber-defense simulation and evaluation platform. It compares BLUE policies against the same rule-based RED threat process while recording both requested and executed actions.

Do not present the interface as a production security operations product or as evidence of real-world autonomous defense performance.

## 2. Show Policy Availability

Open the policy selector or `/agents` response. Point out that each canonical identity reports its own availability and reason. Random and Heuristic require no model artifacts; PPO, Qwen, and Hybrid depend on verified artifacts.

Do not imply that an unavailable policy is running through a hidden substitute.

## 3. Run a Random Episode

1. Select `Random`.
2. Reset the environment.
3. Step several turns or start continuous execution.
4. Show risk, uptime, cost, reward, and episode length changing.
5. Show the requested action, executed action, source identity, and any cooldown substitution.

Explain that substitutions are first-class measurements. A requested action may truthfully execute as `do_nothing` when environment constraints block it.

## 4. Pause, Replay, and Act Manually

Pause the episode and navigate the recorded turns. Submit a manual action and show that its source is rendered as manual rather than attributed to an autonomous policy.

Configuration changes are pending until reset; do not suggest that an in-progress episode was silently reconfigured.

## 5. Present Frozen Evidence

Use the separate P1 and P2 tables in the root README or `docs/experiments.md`:

- P1 compares Random, Heuristic, and Historical PPO.
- P2 adds Fresh PPO on the same frozen matrix.
- Reward is considered alongside breach, risk, uptime, cost, survival, diversity, and substitutions.

The historical PPO requested `investigate_top_alert` on every P1 decision. Fresh PPO was also near-collapsed, so neither result should be described as a broad or generally successful learned defense policy.

## 6. Close With Honest Boundaries

The real Qwen policy passed constrained-policy smoke checks. A controlled tracked Qwen benchmark is still incomplete. Hybrid routing is implemented and tested, but its real-model smoke remains incomplete after hosted GPU session termination.

Avoid live PPO, Qwen, or Hybrid demonstrations unless the exact required artifacts are present and verified. Never infer success from policy labels alone.

## Deferred Manual Verification

The following visual checks remain pending after the P5 automated smoke:

- visual initial-load state
- Random episode interaction
- pause/resume
- paused replay navigation
- manual-action rendering
- configuration-pending behavior
- advisory/simulation wording appearance
- readable error rendering
- naturally occurring terminal-state rendering

These are deferred validation items, not evidence that the interactions failed.
