# CySent: Autonomous Defense Through Simulation

If you've spent any time working in cybersecurity, you know the frustrating reality: most of our tools are purely reactive. They're really good at setting off alarms after the perimeter has been breached, or summarizing the damage after the fact. The result? Analysts are drowning in more notifications than humanly possible to investigate. Meanwhile, adversaries are fully automated—launching phishing campaigns, stealing credentials, and deploying ransomware at machine speed.

It's a lopsided fight. This mismatch led to a fundamental question: What would it look like if a security system could actually learn to defend a network autonomously? That's the idea behind CySent. It isn't just another dashboard or a wrapper around a language model. It's a complete ecosystem—a simulated enterprise network where AI agents train, make real-time decisions, and learn from their mistakes.

## Defense is a Game of Trade-offs

Most AI security tools are built to do one thing at a time, like classifying a single alert. But in the real world, security is a sequence of high-pressure decisions.

If you isolate a machine, do you disrupt the business? Should you patch an exposed service immediately, or wait for confirmation to avoid downtime? If you spot lateral movement, do you protect the backups first or try to contain the infected host?

Every action changes the board. CySent was built specifically to capture this dynamic. We aren't asking the AI to tell us if an email looks suspicious; we're asking if it can keep a network alive over time.

## The Proving Ground

To train an AI to defend, you have to give it something to protect. At the heart of CySent is a simulated, living enterprise network. It includes everything you'd expect: employee email, HR systems, finance databases, public web services, and backup infrastructure.

And it's under constant attack. Adversaries in the simulation actively phish, escalate privileges, and attempt to encrypt critical assets. Risk levels rise and fall based on the moves both sides make. Decisions here have compounding consequences, turning the environment into a true training ground.

## Seeing is Trusting

A massive problem with AI in security is that the logic is usually hidden inside a black box. You feed it data, and it spits out an answer. CySent flips that by including a live frontend interface.

Instead of reading through dry logs, you can literally watch the battle unfold. You can see the network graph, watch the risk levels fluctuate turn-by-turn, and inspect exactly which defensive actions the agent is choosing. If we ever want humans to trust autonomous systems, we have to make their decision-making visible.

Behind the scenes, the backend acts as the "game master." It manages the state of the world, processes attacker actions, and asks the AI for its next move. By keeping the environment totally separate from the agents, we can fairly test different types of AI against the exact same threats.

## The Agents: Three Ways to Play Defense

We didn't just build one type of brain for this platform. CySent features multiple agents that approach the problem differently:

**The RL Baseline (PPO Agent):** This agent uses reinforcement learning (Proximal Policy Optimization). It doesn't start with a textbook knowledge of cybersecurity. Instead, it learns purely through trial and error. It gets rewarded for reducing risk and stabilizing the network, and penalized for breaches. Over time, it figures out complex, long-term strategies—like when it's worth sacrificing a bit of short-term uptime to prevent a catastrophic cascading failure.

**The Language Model (Hugging Face LLM):** We also brought in an LLM defender, specifically built on Qwen 2.5 3B and fine-tuned on project-specific training data. Instead of just chatting, it uses structured reasoning to look at the network state and output concrete commands like `isolate_compromised_host` or `rotate_credentials`. It's an experiment to see if modern LLMs can act as true operational defenders.

**The Random Agent:** This is our reality check. It sounds silly, but if a heavily trained AI model can't drastically outperform a bot mashing random buttons, the training didn't work. We use this baseline to keep our performance claims brutally honest.

## From Concept to Deployment

Training a model on a laptop is a fun experiment, but it isn't a scalable solution. CySent utilizes a full training pipeline running on Hugging Face infrastructure. It handles the repeated simulation episodes, tracks rewards, fine-tunes the adapters, and exports deployable models.

Once trained, the models aren't trapped on a local hard drive. We host the merged models, tokenizers, and configs in Hugging Face model repositories. This modular setup—where the environment, training pipeline, and deployment hosting are all distinctly separated—mirrors how real production systems are actually built.

It also allows us to run rigorous benchmarks. We pit the PPO, LLM, and random agents against each other to measure actual outcomes: total reward, episode survival, and network stability. We want to judge these systems on their behavior, not the hype.

## Where Do We Go From Here?

CySent is already functional, but we are just scratching the surface. The next steps involve exploring attacker-vs-defender self-play, building multi-agent blue teams, integrating with real-world SIEMs, and creating collaborative modes where human analysts work alongside the AI.

Building an AI for cybersecurity is incredibly easy if your only goal is to create another chatbot. It becomes infinitely harder—and much more valuable—when you force that system to make tough decisions under pressure, in a shifting environment, with real consequences.

That is the future CySent is trying to build. Not an AI that just talks about security, but one that actually knows how to defend.
