# CySent: Training AI Defenders for Cybersecurity Battles Humans Can't Scale

If you’ve spent any time in a modern Security Operations Center (SOC), you know the drill. Teams are constantly asked to do more with less. The alert queue never empties, the attack surface keeps expanding, and adversaries are moving faster than manual response processes can handle. A single phishing email can spiral into stolen credentials, lateral movement, and ransomware before anyone even finishes their first cup of coffee.

For years, the industry’s answer has been throwing more dashboards, more triage queues, and more manual effort at the problem. But what if we took a page out of the autonomous vehicle playbook? Before a self-driving car ever hits a public road, its systems are trained and tested in millions of simulated scenarios.

We realized cybersecurity desperately needs the same discipline. That idea became CySent.

## Moving Beyond the "Chatbot" Defender

Most of the AI tools currently hitting the cybersecurity market are single-step assistants. They’re great at classifying a phishing email, summarizing a dense alert, or writing a complex search query. But real-world incidents aren't one-step problems.

Security is fundamentally a decision-making problem under pressure. Analysts constantly weigh trade-offs: Do I isolate this host and disrupt the business, or continue monitoring and risk a breach? Do we prioritize patching the exposed server, or restoring backups? CySent is an autonomous cyber defense platform built around this reality. Instead of static detection tasks, it asks a much harder question: Can an AI system make good defensive decisions over time, with measurable consequences, in a living environment?

To answer that, we built a full-stack simulation engine. It includes a real-time frontend dashboard, a Python-based simulation backend, and a suite of benchmarking and training workflows.

## Building the Proving Ground

To train a real defender, you need a real environment. CySent models a live enterprise network complete with employee email services, HR systems, public web servers, and identity infrastructure.

Inside this environment, simulated attackers run realistic campaigns—phishing, password spraying, privilege escalation, and deploying ransomware. Because the environment is dynamic, every turn changes the state of the network. Risk fluctuates. Systems crash and recover. Actions have consequences that unfold over time, which is a complexity most AI security projects completely ignore. It's an environment of interaction, not just isolated prompts.

## The Competitors: RL, LLMs, and... Chaos

To figure out what actually works, we drop different types of agents into this environment.

### The Reinforcement Learning (RL) Baseline

One of our core defenders is a Proximal Policy Optimization (PPO) agent. It learns purely through trial and error, optimizing for rewards tied to keeping the network up, reducing risk, and recovering quickly. It gives us a strong, learning-based baseline.

### The Language Model (LLM) Defender

We also integrated a Hugging Face LLM agent, built on `Qwen2.5-3B-Instruct`. We fine-tuned the model specifically on defense trajectories and converted it into a deployable, merged model (`sxchin01/CySent-Qwen-RL-merged`). Instead of acting like a helpful assistant, it outputs actionable commands like `isolate_suspicious_host` or `patch_auth_server`.

### The Random Baseline

We intentionally included a completely random agent. It’s incredibly easy to overstate AI progress if you don't have a naive baseline. Comparing our trained systems against pure chaos keeps the results honest.

We use both RL and LLMs because they solve different parts of the autonomy puzzle. RL is fantastic at optimizing long-term rewards and building stable policies for known environments. LLMs, on the other hand, excel at interpreting complex, structured context and adapting to weird, novel situations. We strongly suspect the future of automated defense isn't a single model type, but a hybrid approach.

## Lessons Learned in the Trenches

Building CySent broke a few of our assumptions and reinforced others.

First, benchmarks matter. It is wildly easy to build a flashy AI demo, but creating honest, repeatable comparisons is incredibly difficult. We had to build strict tracking for metrics like network stability, episode success, and risk reduction to ensure we weren't just tricking ourselves.

Second, deployment is half the battle. Training a model is fun, but real systems need APIs, routing, fallbacks, and solid hosting (which is why our deployable model lives on Hugging Face).

Most importantly, we realized that security is sequential. Incidents rarely escalate because of one single bad decision; they escalate because a series of decisions were made too late. And if you want humans to trust an AI to make those decisions faster, you have to provide transparency. That’s why CySent isn't just an API in the dark—it features a live frontend with a network graph where every decision and consequence can be visually inspected.

## What's Next?

We’re just scratching the surface. We’re currently distributed across GitHub (for the source code), a Hugging Face Space (for the interactive app), and the Model Repo.

Looking ahead, we want to explore attacker-vs-defender self-play, multi-agent coordination, and curriculum learning with increasingly aggressive simulated threats. We also want to build better workflows for "human-in-the-loop" analysts, allowing humans and AI to defend a network together.

The security industry is inevitably moving toward automation. But automation without rigorous, simulated testing is a recipe for disaster—it can create more risk than it mitigates. CySent is our way of establishing a practical path forward: simulate first, benchmark honestly, inspect behavior, and deploy carefully.

The future defender probably isn't a chatbot window. It’s a continuously trained system that weighs trade-offs, reacts in milliseconds, and has lived through thousands of simulated breaches before it ever touches your production environment.
