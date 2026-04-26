"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ChevronRight, Radar } from "lucide-react";

import { AICommander } from "@/components/ai-commander";
import { IncidentFeed } from "@/components/incident-feed";
import { MetricsPanel } from "@/components/metrics-panel";
import { NetworkGraph } from "@/components/network-graph";
import { ReplayControls } from "@/components/replay-controls";
import { Topbar } from "@/components/topbar";
import { fetchState, resetSimulation, step } from "@/lib/api";
import { EnvState, StepResult, StrategyMode, TimelineFrame, TimelinePoint, ActionSource } from "@/lib/types";

const initialState: EnvState = {
  episode_id: "",
  step: 0,
  network_risk: 0,
  risk_breakdown: {},
  assets: [],
  last_action: "reset",
  red_log: {},
  profile: {},
  intelligence: { enabled: true },
  events: [],
  narrative: "",
  termination_reason: "active",
};

export default function HomePage() {
  const [state, setState] = useState<EnvState>(initialState);
  const [scenario, setScenario] = useState("bank");
  const [difficulty, setDifficulty] = useState("hard");
  const [attacker, setAttacker] = useState("ransomware_gang");
  const [strategyMode, setStrategyMode] = useState<StrategyMode>("balanced");
  const [actionSource, setActionSource] = useState<ActionSource>("ppo_agent");
  const [activeAgentLabel, setActiveAgentLabel] = useState("PPO Defender");

  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [booting, setBooting] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [incidents, setIncidents] = useState<string[]>([]);
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [frames, setFrames] = useState<TimelineFrame[]>([]);

  const [replayIndex, setReplayIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [timelineCollapsed, setTimelineCollapsed] = useState(true);

  const stateRef = useRef<EnvState>(initialState);
  const liveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const replayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const activeState = replayPlaying && frames.length > 0 ? frames[Math.min(replayIndex, frames.length - 1)].state : state;

  useEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    const boot = async () => {
      try {
        setErrorMessage(null);
        const live = await fetchState();
        setState(live);
        stateRef.current = live;
        setTimeline([{
          turn: live.step, reward: 0, risk: live.network_risk,
          uptime: uptimeFromAssets(live.assets), breaches: breachesFromAssets(live.assets),
          securityScore: securityScore(live.network_risk),
        }]);
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : "Unable to load backend state.");
      } finally { setBooting(false); }
    };
    void boot();
  }, []);

  useEffect(() => {
    if (!running) {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
      return;
    }
    liveIntervalRef.current = setInterval(() => { void runLiveStep(); }, 1050);
    return () => { if (liveIntervalRef.current) clearInterval(liveIntervalRef.current); liveIntervalRef.current = null; };
  }, [running, busy, scenario, difficulty, attacker, strategyMode]);

  useEffect(() => {
    if (!replayPlaying || frames.length <= 1) {
      if (replayIntervalRef.current) clearInterval(replayIntervalRef.current);
      replayIntervalRef.current = null;
      return;
    }
    const intervalMs = Math.max(120, 700 / replaySpeed);
    replayIntervalRef.current = setInterval(() => {
      setReplayIndex((prev) => {
        if (prev >= frames.length - 1) { setReplayPlaying(false); return prev; }
        return prev + 1;
      });
    }, intervalMs);
    return () => { if (replayIntervalRef.current) clearInterval(replayIntervalRef.current); replayIntervalRef.current = null; };
  }, [replayPlaying, replaySpeed, frames.length]);

  const runLiveStep = async () => {
    if (busy) return;
    setBusy(true);
    try {
      setErrorMessage(null);
      const result = await step();
      const prev = stateRef.current;
      const resolvedState: EnvState = {
        episode_id: result.episode_id, step: prev.step + 1, network_risk: result.network_risk,
        risk_breakdown: result.risk_breakdown, assets: result.assets, last_action: result.action_name,
        red_log: result.red_log, profile: result.profile, intelligence: result.intelligence,
        events: result.events, narrative: result.narrative, termination_reason: result.termination_reason,
      };
      stateRef.current = resolvedState;
      setState(resolvedState);
      setActiveAgentLabel(result.active_agent ?? AGENT_LABELS[actionSource] ?? "PPO Defender");
      setIncidents((p) => [incidentLine(resolvedState, result), ...p].slice(0, 50));
      setTimeline((p) => [...p, {
        turn: resolvedState.step, reward: result.reward, risk: result.network_risk,
        uptime: uptimeFromAssets(result.assets), breaches: breachesFromAssets(result.assets),
        securityScore: securityScore(result.network_risk),
      }].slice(-200));
      setFrames((p) => [...p, { turn: resolvedState.step, state: resolvedState, result, timestamp: Date.now() }].slice(-250));
      if (result.terminated || result.truncated) setRunning(false);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Step request failed.");
      if (err instanceof Error && isBackendReachabilityError(err)) {
        setActiveAgentLabel("Backend Offline");
      }
      setRunning(false);
    } finally { setBusy(false); }
  };

  const handleReset = async () => {
    setRunning(false); setReplayPlaying(false);
    try {
      setErrorMessage(null);
      const synced = await resetSimulation({
        seed: 42, scenario, difficulty, attacker, strategy_mode: strategyMode,
        action_source: actionSource, intelligence_enabled: true,
      });
      setState(synced); stateRef.current = synced; setIncidents([]);
      setTimeline([{ turn: synced.step, reward: 0, risk: synced.network_risk, uptime: uptimeFromAssets(synced.assets), breaches: breachesFromAssets(synced.assets), securityScore: securityScore(synced.network_risk) }]);
      setFrames([]); setReplayIndex(0);
      setActiveAgentLabel(AGENT_LABELS[actionSource] ?? "PPO Defender");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Reset request failed.");
      if (err instanceof Error && isBackendReachabilityError(err)) {
        setActiveAgentLabel("Backend Offline");
      }
    }
  };

  const handleStartPause = async () => {
    if (running) { setRunning(false); return; }
    try {
      setErrorMessage(null);
      const synced = await resetSimulation({
        seed: 42, scenario, difficulty, attacker, strategy_mode: strategyMode,
        action_source: actionSource, intelligence_enabled: true,
      });
      setState(synced); stateRef.current = synced; setIncidents([]);
      setTimeline([{ turn: synced.step, reward: 0, risk: synced.network_risk, uptime: uptimeFromAssets(synced.assets), breaches: breachesFromAssets(synced.assets), securityScore: securityScore(synced.network_risk) }]);
      setFrames([]); setReplayIndex(0);
      setActiveAgentLabel(AGENT_LABELS[actionSource] ?? "PPO Defender");
      setRunning(true);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Unable to start simulation.");
      if (err instanceof Error && isBackendReachabilityError(err)) {
        setActiveAgentLabel("Backend Offline");
      }
    }
  };

  const currentConfidence = activeState.intelligence?.reasoning?.decision_confidence ?? 0;
  const security = useMemo(() => securityScore(activeState.network_risk), [activeState.network_risk]);
  const underAttack = Boolean(activeState.red_log?.attack && activeState.red_log?.attack !== "no_attack");

  return (
    <main className="relative min-h-screen bg-[#050508] text-white/90">
      {/* Cinematic warm glow — Darktrace hero orb */}
      <div className="dt-glow-orb" style={{ top: "5%", left: "30%" }} />
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_50%_35%_at_50%_0%,rgba(200,80,20,0.05),transparent)]" />

      <Topbar
        scenario={scenario} difficulty={difficulty} attacker={attacker} strategyMode={strategyMode}
        actionSource={actionSource} activeAgentLabel={activeAgentLabel} running={running}
        onScenarioChange={setScenario} onDifficultyChange={setDifficulty} onAttackerChange={setAttacker}
        onStrategyChange={setStrategyMode} onActionSourceChange={setActionSource}
        onStartPause={() => void handleStartPause()} onReset={() => void handleReset()}
      />

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 mx-auto flex w-full max-w-[1720px] flex-col gap-6 px-6 py-6 lg:px-10"
      >
        {booting && (
          <div className="rounded-xl border border-orange-500/20 bg-orange-500/[0.06] px-5 py-3.5 text-sm text-orange-300/80">
            Connecting to CySent backend...
          </div>
        )}
        {errorMessage && (
          <div className="rounded-xl border border-red-500/20 bg-red-500/[0.06] px-5 py-3.5 text-sm text-red-300/80">
            {errorMessage}
          </div>
        )}

        {/* ── Hero section: headline + key stats ──────────────────── */}
        <section className="flex flex-col gap-2 pb-2">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Autonomous Cyber Defense
          </h2>
          <p className="max-w-2xl text-sm leading-relaxed text-white/40">
            Real-time AI-driven threat detection, response orchestration, and blue-team policy optimization across your simulated enterprise network.
          </p>
          <div className="mt-3 flex flex-wrap gap-4">
            <StatPill label="Turn" value={String(activeState.step)} />
            <StatPill label="Risk" value={activeState.network_risk.toFixed(3)} />
            <StatPill label="Score" value={`${security}`} />
            <StatPill label="Confidence" value={`${Math.round(currentConfidence * 100)}%`} />
          </div>
        </section>

        {/* ── Main grid ───────────────────────────────────────────── */}
        <motion.section
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08, duration: 0.45 }}
          className="grid grid-cols-1 gap-6 xl:grid-cols-12"
        >
          {/* Network Map */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.14, duration: 0.42 }}
            className="xl:col-span-8"
          >
            <div className="dt-panel overflow-hidden p-5">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="dt-label">Network Operations</p>
                  <h3 className="mt-1 text-lg font-semibold text-white">Live Battlefield</h3>
                </div>
                <motion.div
                  animate={underAttack ? { boxShadow: ["0 0 0 rgba(248,113,113,0)", "0 0 24px rgba(240,100,48,0.4)", "0 0 0 rgba(248,113,113,0)"] } : {}}
                  transition={{ duration: 1.8, repeat: underAttack ? Infinity : 0 }}
                  className="rounded-full bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-white/50 ring-1 ring-white/[0.06]"
                >
                  Risk {activeState.network_risk.toFixed(3)}
                </motion.div>
              </div>

              <NetworkGraph assets={activeState.assets} redTarget={String(activeState.red_log?.target ?? "")} underAttack={underAttack} />

              {activeState.assets.length === 0 && (
                <p className="mt-4 rounded-lg bg-white/[0.03] px-4 py-3 text-xs text-white/30">
                  No asset state available yet. Reset or Start to initialize the simulation feed.
                </p>
              )}
            </div>
          </motion.div>

          {/* AI Commander */}
          <motion.div
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.45 }}
            className="xl:col-span-4"
          >
            <AICommander intelligence={activeState.intelligence} strategyMode={strategyMode} />
          </motion.div>
        </motion.section>

        {/* ── Metrics + Replay ─────────────────────────────────────── */}
        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.26, duration: 0.45 }}
          className="grid grid-cols-1 gap-6 xl:grid-cols-12"
        >
          <div className="xl:col-span-9">
            <MetricsPanel timeline={timeline} />
          </div>
          <div className="space-y-4 xl:col-span-3">
            <ReplayControls
              playing={replayPlaying} speed={replaySpeed}
              canStepBack={replayIndex > 0} canStepForward={replayIndex < frames.length - 1}
              onPlayPause={() => setReplayPlaying((v) => !v)}
              onStepBack={() => setReplayIndex((idx) => Math.max(0, idx - 1))}
              onStepForward={() => setReplayIndex((idx) => Math.min(frames.length - 1, idx + 1))}
              onSpeedChange={setReplaySpeed}
            />
          </div>
        </motion.section>
      </motion.div>

      {/* ── Timeline toggle ──────────────────────────────────────── */}
      <div className="fixed bottom-5 right-5 z-40">
        <motion.button
          onClick={() => setTimelineCollapsed((v) => !v)}
          whileHover={{ y: -2 }}
          whileTap={{ scale: 0.98 }}
          className="dt-btn-ghost flex items-center gap-2 rounded-full px-4 py-2 text-[11px] uppercase tracking-[0.1em] backdrop-blur-xl"
        >
          <Radar size={13} />
          Timeline
          <ChevronRight size={13} className={`transition-transform duration-300 ${timelineCollapsed ? "rotate-0" : "rotate-180"}`} />
        </motion.button>
      </div>

      <motion.aside
        initial={false}
        animate={{ x: timelineCollapsed ? "86%" : "0%" }}
        transition={{ duration: 0.3 }}
        className="fixed right-0 top-[60px] z-30 h-[calc(100vh-80px)] w-[370px] max-w-[88vw] pr-3"
      >
        <IncidentFeed incidents={incidents} collapsed={timelineCollapsed} onToggle={() => setTimelineCollapsed((v) => !v)} />
      </motion.aside>
    </main>
  );
}

/* ── Stat pill (Darktrace-style inline stat) ──────────────────── */
function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-2xl font-bold tracking-tight text-white">{value}</span>
      <span className="text-xs font-medium text-white/30">{label}</span>
    </div>
  );
}

function incidentLine(nextState: EnvState, result: StepResult): string {
  const narrative = String(result.narrative ?? "").trim();
  if (narrative) return narrative;
  const attack = String(result.red_log?.attack ?? "unknown");
  const target = String(result.red_log?.target ?? "network");
  const posture = String(nextState.intelligence?.posture?.level ?? "guarded");
  return `${result.action_name} executed while ${attack} targeted ${target}; posture ${posture}.`;
}

function uptimeFromAssets(assets: EnvState["assets"]): number {
  if (!assets.length) return 1;
  return assets.filter((a) => a.uptime_status).length / assets.length;
}

function breachesFromAssets(assets: EnvState["assets"]): number {
  if (!assets.length) return 0;
  return assets.filter((a) => a.compromised).length / assets.length;
}

function securityScore(networkRisk: number): number {
  return Math.max(0, Math.round((1 - networkRisk) * 100));
}

function isBackendReachabilityError(err: Error): boolean {
  return err.message.includes("Cannot reach backend") || err.message.includes("API timeout");
}

const AGENT_LABELS: Record<string, string> = {
  ppo_agent: "PPO Defender",
  hf_llm_agent: "HF LLM Defender",
  hybrid: "Hybrid Defender",
  random: "Random Baseline",
};
