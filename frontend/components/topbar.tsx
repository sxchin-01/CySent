"use client";

import { motion } from "framer-motion";
import { Pause, Play, RotateCcw, Shield, Cpu, Brain, Shuffle, GitMerge } from "lucide-react";

import { ActionSource, AgentAvailability, StrategyMode } from "@/lib/types";

type TopbarProps = {
  scenario: string;
  difficulty: string;
  attacker: string;
  strategyMode: StrategyMode;
  actionSource: ActionSource;
  agentOptions: AgentAvailability[];
  activeAgentLabel: string;
  running: boolean;
  busy: boolean;
  configLocked: boolean;
  startLabel: "Start" | "Pause" | "Resume";
  onScenarioChange: (value: string) => void;
  onDifficultyChange: (value: string) => void;
  onAttackerChange: (value: string) => void;
  onStrategyChange: (value: StrategyMode) => void;
  onActionSourceChange: (value: ActionSource) => void;
  onStartPause: () => void;
  onReset: () => void;
};

const SCENARIOS = ["bank", "hospital", "saas", "government", "manufacturing"].map(toOption);
const DIFFICULTIES = ["easy", "medium", "hard"].map(toOption);
const ATTACKERS = ["ransomware_gang", "credential_thief", "silent_apt", "insider_saboteur", "botnet"].map(toOption);
const STRATEGIES: StrategyMode[] = ["conservative", "balanced", "aggressive"];

const AGENT_LABELS: Record<ActionSource, string> = {
  random: "Random Baseline",
  heuristic: "Heuristic Baseline",
  ppo_historical_checkpoint: "Historical PPO",
  ppo_fresh_checkpoint: "Fresh PPO",
  qwen_rl: "Qwen RL Policy",
  hybrid_router: "Hybrid Router",
};

const AGENT_ICON: Record<string, typeof Cpu> = {
  ppo_historical_checkpoint: Cpu,
  ppo_fresh_checkpoint: Cpu,
  qwen_rl: Brain,
  hybrid_router: GitMerge,
  random: Shuffle,
  heuristic: Cpu,
};

export function Topbar(props: TopbarProps) {
  const AgentIcon = AGENT_ICON[props.actionSource] ?? Cpu;
  const agentOptions = props.agentOptions.map((agent) => ({
    value: agent.identity,
    label: `${AGENT_LABELS[agent.identity]}${agent.live_selectable ? (agent.available ? "" : " - unavailable") : " - benchmark only"}`,
    disabled: !agent.live_selectable || !agent.available,
    title: agent.reason ?? undefined,
  }));

  return (
    <motion.header
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="sticky top-0 z-40"
    >
      <div className="flex items-center justify-between border-b border-white/[0.06] bg-black/60 px-6 py-3 backdrop-blur-2xl">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-gradient-to-br from-orange-500/20 to-orange-600/5 text-orange-400">
            <Shield size={16} strokeWidth={2.5} />
          </div>
          <div className="hidden sm:block">
            <h1 className="text-[15px] font-semibold tracking-tight text-white">CySent</h1>
          </div>
        </div>

        {/* Controls row */}
        <div className="flex flex-wrap items-center gap-2">
          <Select label="Scenario" value={props.scenario} options={SCENARIOS} disabled={props.configLocked} onChange={props.onScenarioChange} />
          <Select label="Difficulty" value={props.difficulty} options={DIFFICULTIES} disabled={props.configLocked} onChange={props.onDifficultyChange} />
          <Select label="Attacker" value={props.attacker} options={ATTACKERS} disabled={props.configLocked} onChange={props.onAttackerChange} />
          <Select
            label="Advisory"
            value={props.strategyMode}
            options={STRATEGIES.map(toOption)}
            disabled={props.configLocked}
            onChange={(v) => props.onStrategyChange(v as StrategyMode)}
          />
          <Select
            label="Agent"
            value={props.actionSource}
            options={agentOptions}
            disabled={props.configLocked}
            onChange={(v) => props.onActionSourceChange(v as ActionSource)}
          />

          <div className="ml-1 h-5 w-px bg-white/[0.06]" />

          <button onClick={props.onStartPause} disabled={props.busy} className="dt-btn-primary disabled:cursor-not-allowed disabled:opacity-40">
            {props.running ? <Pause size={14} /> : <Play size={14} />}
            {props.startLabel}
          </button>
          <button onClick={props.onReset} disabled={props.busy} className="dt-btn-ghost disabled:cursor-not-allowed disabled:opacity-40">
            <RotateCcw size={13} />
            Reset
          </button>

          <div className="ml-1 h-5 w-px bg-white/[0.06]" />

          <div className="flex items-center gap-2 rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-white/60">
            <AgentIcon size={13} className="text-orange-400/80" />
            <span>{props.activeAgentLabel}</span>
          </div>
        </div>
      </div>
    </motion.header>
  );
}

type SelectProps = {
  label: string;
  value: string;
  options: readonly SelectOption[];
  disabled?: boolean;
  onChange: (value: string) => void;
};

type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
  title?: string;
};

function Select({ label, value, options, disabled = false, onChange }: SelectProps) {
  return (
    <label className="dt-select">
      <span className="text-[9px] font-medium uppercase tracking-[0.1em] text-white/30">{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="bg-transparent text-[12px] font-medium text-white/80 outline-none disabled:cursor-not-allowed disabled:text-white/30"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled} title={o.title}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

function toOption(value: string): SelectOption {
  return { value, label: value.replaceAll("_", " ") };
}
