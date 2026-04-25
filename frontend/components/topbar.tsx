"use client";

import { motion } from "framer-motion";
import { Pause, Play, RotateCcw, Shield, Cpu, Brain, Shuffle, GitMerge } from "lucide-react";

import { ActionSource, StrategyMode } from "@/lib/types";

type TopbarProps = {
  scenario: string;
  difficulty: string;
  attacker: string;
  strategyMode: StrategyMode;
  actionSource: ActionSource;
  activeAgentLabel: string;
  running: boolean;
  onScenarioChange: (value: string) => void;
  onDifficultyChange: (value: string) => void;
  onAttackerChange: (value: string) => void;
  onStrategyChange: (value: StrategyMode) => void;
  onActionSourceChange: (value: ActionSource) => void;
  onStartPause: () => void;
  onReset: () => void;
};

const SCENARIOS = ["bank", "hospital", "saas", "government", "manufacturing"];
const DIFFICULTIES = ["easy", "medium", "hard"];
const ATTACKERS = ["ransomware_gang", "credential_thief", "silent_apt", "insider_saboteur", "botnet"];
const STRATEGIES: StrategyMode[] = ["conservative", "balanced", "aggressive"];
const AGENTS: ActionSource[] = ["ppo_agent", "hf_llm_agent", "hybrid", "random"];

const AGENT_ICON: Record<string, typeof Cpu> = {
  ppo_agent: Cpu,
  hf_llm_agent: Brain,
  hybrid: GitMerge,
  random: Shuffle,
};

export function Topbar(props: TopbarProps) {
  const AgentIcon = AGENT_ICON[props.actionSource] ?? Cpu;

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
          <Select label="Scenario" value={props.scenario} options={SCENARIOS} onChange={props.onScenarioChange} />
          <Select label="Difficulty" value={props.difficulty} options={DIFFICULTIES} onChange={props.onDifficultyChange} />
          <Select label="Attacker" value={props.attacker} options={ATTACKERS} onChange={props.onAttackerChange} />
          <Select
            label="Strategy"
            value={props.strategyMode}
            options={STRATEGIES}
            onChange={(v) => props.onStrategyChange(v as StrategyMode)}
          />
          <Select
            label="Agent"
            value={props.actionSource}
            options={AGENTS}
            onChange={(v) => props.onActionSourceChange(v as ActionSource)}
          />

          <div className="ml-1 h-5 w-px bg-white/[0.06]" />

          <button onClick={props.onStartPause} className="dt-btn-primary">
            {props.running ? <Pause size={14} /> : <Play size={14} />}
            {props.running ? "Pause" : "Start Sim"}
          </button>
          <button onClick={props.onReset} className="dt-btn-ghost">
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
  options: readonly string[];
  onChange: (value: string) => void;
};

function Select({ label, value, options, onChange }: SelectProps) {
  return (
    <label className="dt-select">
      <span className="text-[9px] font-medium uppercase tracking-[0.1em] text-white/30">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-transparent text-[12px] font-medium text-white/80 outline-none"
      >
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </label>
  );
}
