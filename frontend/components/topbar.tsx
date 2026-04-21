"use client";

import { motion } from "framer-motion";
import { Pause, Play, RotateCcw, Shield, Swords, Cpu, Brain } from "lucide-react";

import { ActionSource, StrategyMode } from "@/lib/types";

type TopbarProps = {
  scenario: string;
  difficulty: string;
  attacker: string;
  strategyMode: StrategyMode;
  actionSource: ActionSource;
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
const AGENTS: ActionSource[] = ["ppo_agent", "hf_llm_agent"];

export function Topbar(props: TopbarProps) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="sticky top-0 z-40 rounded-2xl border border-cyan-200/10 bg-slate-950/55 px-3 py-2 backdrop-blur-2xl"
    >
      <div className="flex flex-col gap-2 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-center gap-2">
          <div className="rounded-lg border border-cyan-300/40 bg-cyan-400/10 p-1.5 text-cyan-200">
            <Shield size={18} />
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-[0.24em] text-cyan-200/75">CySent Platform</p>
            <h1 className="text-sm font-semibold text-white sm:text-base">Autonomous Cyber Defense Command Center</h1>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-5 xl:flex xl:flex-wrap xl:items-center xl:justify-end">
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

          <button
            onClick={props.onStartPause}
            className="flex items-center justify-center gap-1.5 rounded-lg border border-emerald-300/40 bg-emerald-300/10 px-2.5 py-1.5 text-xs font-medium text-emerald-100 transition hover:bg-emerald-300/20"
          >
            {props.running ? <Pause size={14} /> : <Play size={14} />}
            {props.running ? "Pause" : "Start"}
          </button>
          <button
            onClick={props.onReset}
            className="flex items-center justify-center gap-1.5 rounded-lg border border-amber-300/40 bg-amber-300/10 px-2.5 py-1.5 text-xs font-medium text-amber-100 transition hover:bg-amber-300/20"
          >
            <RotateCcw size={14} />
            Reset View
          </button>
          <div className="col-span-2 flex items-center justify-center gap-1.5 rounded-lg border border-fuchsia-300/40 bg-fuchsia-300/10 px-2.5 py-1.5 text-xs text-fuchsia-100 sm:col-span-5 xl:col-span-1">
            {props.actionSource === "ppo_agent" ? <Cpu size={14} /> : <Brain size={14} />}
            {props.actionSource === "ppo_agent" ? "PPO Defender" : "HF LLM Defender"}
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
    <label className="flex min-w-[126px] flex-col gap-0.5 rounded-lg border border-white/15 bg-white/5 px-2 py-1">
      <span className="text-[9px] uppercase tracking-[0.14em] text-slate-300">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-white/15 bg-slate-900/80 px-2 py-1 text-xs text-white outline-none transition focus:border-cyan-300/60"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
