# CySent: Training AI Defenders for Cybersecurity Battles Humans Can't Scale

Cybersecurity teams are being asked to do more with less.

There are more alerts than analysts can realistically investigate, more exposed systems than teams can manually secure, and attackers who move faster than most response processes allow. A single phishing email can lead to stolen credentials, lateral movement, data loss, and ransomware before anyone has time to react.

Most organizations are still trying to solve this with dashboards, triage queues, and more manual effort.

But what if defenders could train autonomous security agents in simulation before deploying them into the real world, similar to how self-driving systems are tested before going on public roads?

That idea became CySent.

## What CySent Is

CySent is an autonomous cyber defense platform where AI agents learn to defend a simulated enterprise network against evolving attacks.

Rather than focusing on static detection tasks or chatbot-style assistance, CySent is built around a harder question:

Can an AI system make good defensive decisions over time, under pressure, with measurable consequences?

Today, the project includes:

- A real-time frontend dashboard
- A Python backend simulation engine
- Reinforcement learning agents
- Hugging Face language-model defenders
- Benchmarking tools
- Training workflows
- Deployable model hosting through Hugging Face

What started as an experiment has become a full-stack AI security project.

## Why This Problem Matters

Most AI tools in cybersecurity still operate at a single-step level:

- Classify a phishing email
- Summarize an alert
- Search logs faster
- Cluster similar incidents

Those tools are useful, but real incidents are not one-step problems.

Security teams constantly make decisions like:

- Isolate a host now or continue monitoring?
- Rotate credentials immediately or avoid disruption?
- Patch an exposed server first or restore backups first?
- Contain ransomware spread or protect critical assets?

Cyber defense is ultimately a decision-making problem.

CySent was built around that reality.

## The Simulated Environment

CySent models a live enterprise network with systems such as:

- Employee email services
- HR systems
- Public web servers
- Finance databases
- Identity infrastructure
- Backup systems
- SOC monitoring tools

Attackers run realistic campaigns that may include:

- Phishing
- Password spraying
- Credential theft
- Privilege escalation
- Lateral movement
- Ransomware deployment

Each turn changes the state of the network.

Risk can rise or fall. Systems can fail or recover. Critical assets can become exposed or secured.

That creates something many AI projects never deal with:

Consequences that unfold over time.

## The Agents Inside CySent

### PPO Reinforcement Learning Agent

One of the core defenders is a PPO (Proximal Policy Optimization) agent trained directly in the live environment.

It improves through rewards tied to outcomes such as:

- Preserving uptime
- Reducing network risk
- Preventing breaches
- Recovering quickly after incidents

This provides a strong learning-based baseline.

### Hugging Face LLM Agent

CySent also includes a language-model defender built on:

- `Qwen/Qwen2.5-3B-Instruct`

The model was first fine-tuned on defense trajectories and later converted into a merged deployable model:

- `sxchin01/CySent-Qwen-RL-merged`

This allows the frontend `hf_llm_agent` option to call a model trained for cyber-defense decisions rather than a general-purpose assistant.

Typical outputs include actions such as:

- `isolate_suspicious_host`
- `rotate_credentials`
- `patch_auth_server`
- `investigate_top_alert`

### Random Baseline

A random agent is also included intentionally.

Without a naive baseline, it is easy to overstate progress. Comparing trained systems against random behavior helps keep results honest.

## What Makes CySent Different

### It's an Environment, Not Just a Model

CySent is built around interaction. Agents operate inside a changing world rather than answering isolated prompts.

### It Supports Multiple Agent Types

The same environment can evaluate:

- PPO agents
- Language-model defenders
- Random baselines

That makes comparisons meaningful.

### It Measures Outcomes

CySent tracks metrics like:

- Reward
- Risk
- Network stability
- Episode success
- Action performance

### It Has a Real Interface

The frontend includes a network graph, controls, and live status views so decisions can be inspected rather than hidden behind an API.

## Why Use Both RL and LLMs?

Reinforcement learning and language models solve different parts of the autonomy problem.

RL tends to be strong at:

- Optimizing long-term reward
- Repeated control scenarios
- Stable policies in known environments

LLMs tend to be strong at:

- Interpreting structured context
- Adapting to new situations
- Leveraging broad prior knowledge

CySent explores the idea that future cyber defense systems may be hybrid rather than relying on a single model type.

## Lessons From Building It

### Benchmarks Matter

Good demos are easy to build. Honest comparisons are harder.

### Deployment Matters

Training a model is only part of the work. Real systems also need routing, hosting, fallbacks, APIs, and observability.

### Security Is Sequential

Many incidents escalate not because one decision was wrong, but because several decisions came too late.

### Transparency Matters

People trust systems they can inspect. That's why CySent emphasizes visualization and measurable outcomes.

## Current Deployment Stack

CySent is currently distributed across three main surfaces:

### GitHub

Source code, version control, reproducibility.

### Hugging Face Space

Hosted interactive application.

### Hugging Face Model Repo

Merged deployable model:

- `sxchin01/CySent-Qwen-RL-merged`

## Where It Could Go Next

There is still plenty of room to expand:

- Attacker vs defender self-play
- Human-in-the-loop analyst workflows
- SIEM integrations
- Policy-constrained decision making
- Multi-agent defense coordination
- Stronger benchmark suites
- Curriculum learning with escalating threats

## Why This Project Matters

The security industry is moving toward automation.

But automation without rigorous testing can create more risk than it removes.

CySent takes a more practical path:

- Simulate first
- Benchmark honestly
- Train repeatedly
- Inspect behavior
- Deploy carefully

That is how autonomy matured in robotics and self-driving systems. Cybersecurity deserves the same discipline.

## Final Thought

The future defender may not be a chatbot window or a rules engine.

It may be a continuously trained system that monitors a network, weighs tradeoffs, reacts in seconds, and learns from thousands of simulated incidents before touching a production environment.

CySent is one step toward that future, and just as importantly, a place to test whether that future should exist at all.
