# CySent v1: Two-Minute Research Pitch

Autonomous cyber-defense research can look convincing while hiding a basic problem: the policy label, requested action, and action the environment actually executed may not be the same thing.

CySent is an experimental simulation and evaluation platform built to make those distinctions observable. A shared rule-based RED process challenges six explicitly identified BLUE paths: Random, deterministic Heuristic, Historical PPO, Fresh PPO, a historical Qwen RL policy, and a PPO/Qwen Hybrid Router. Every episode records reward together with breach rate, network risk, uptime, action cost, survival, action distribution, and cooldown substitutions.

The frozen experiments produced a useful negative result. Historical PPO earned competitive reward but requested `investigate_top_alert` on 100% of its P1 decisions, with nearly half substituted to `do_nothing`. Fresh PPO improved several outcome metrics, yet remained near-collapsed: 96.9% of requests were still `investigate_top_alert`. Reward alone therefore did not establish a diverse or trustworthy defense strategy.

CySent also reconstructs the historical Qwen policy faithfully: first-token action scoring over the frozen 12-action vocabulary, including the known three-way patch-token collision, FP32 normalization, and reproducible agent-local sampling. Real-model smoke checks passed. The controlled Qwen benchmark and real-model Hybrid smoke are still incomplete, so v1 makes no comparative performance claim for either path.

The reproducible v1 deliverable is a CPU research/demo path for Random, Heuristic, and manual actions, plus hash-bound loading contracts for external PPO and Qwen artifacts. The interface reports policy availability and failures honestly instead of silently changing agent identity.

CySent v1 is not a production SOC platform. It is a compact research artifact for studying whether autonomous defense evaluations remain truthful when policy behavior, environment constraints, and deployment availability all matter.
