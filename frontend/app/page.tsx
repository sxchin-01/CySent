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

const themeByScenario: Record<string, string> = {
  bank: "from-[#070b15] via-[#0e1a30] to-[#201538]",
  hospital: "from-[#060f18] via-[#0f2433] to-[#16383f]",
  saas: "from-[#090f1d] via-[#121f35] to-[#1e3252]",
  government: "from-[#080f1f] via-[#122136] to-[#2a1e36]",
  manufacturing: "from-[#0a0f16] via-[#1a232a] to-[#2a2321]",
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

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    const boot = async () => {
      try {
        setErrorMessage(null);
        const live = await fetchState();
        setState(live);
        stateRef.current = live;
        setTimeline([
          {
            turn: live.step,
            reward: 0,
            risk: live.network_risk,
            uptime: uptimeFromAssets(live.assets),
            breaches: breachesFromAssets(live.assets),
            securityScore: securityScore(live.network_risk),
          },
        ]);
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : "Unable to load backend state.");
      } finally {
        setBooting(false);
      }
    };
    void boot();
  }, []);

  useEffect(() => {
    if (!running) {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
      return;
    }

    liveIntervalRef.current = setInterval(() => {
      void runLiveStep();
    }, 1050);

    return () => {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    };
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
        if (prev >= frames.length - 1) {
          setReplayPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      if (replayIntervalRef.current) clearInterval(replayIntervalRef.current);
      replayIntervalRef.current = null;
    };
  }, [replayPlaying, replaySpeed, frames.length]);

  const runLiveStep = async () => {
    if (busy) return;
    setBusy(true);
    try {
      setErrorMessage(null);
      const result = await step();
      const prev = stateRef.current;
      const resolvedState: EnvState = {
        episode_id: result.episode_id,
        step: prev.step + 1,
        network_risk: result.network_risk,
        risk_breakdown: result.risk_breakdown,
        assets: result.assets,
        last_action: result.action_name,
        red_log: result.red_log,
        profile: result.profile,
        intelligence: result.intelligence,
        events: result.events,
        narrative: result.narrative,
        termination_reason: result.termination_reason,
      };

      stateRef.current = resolvedState;
      setState(resolvedState);
      setActiveAgentLabel(result.active_agent ?? (actionSource === "ppo_agent" ? "PPO Defender" : "Colab LLM Defender"));

      setIncidents((prevLines) => [incidentLine(resolvedState, result), ...prevLines].slice(0, 50));
      setTimeline((prevPoints) => [
        ...prevPoints,
        {
          turn: resolvedState.step,
          reward: result.reward,
          risk: result.network_risk,
          uptime: uptimeFromAssets(result.assets),
          breaches: breachesFromAssets(result.assets),
          securityScore: securityScore(result.network_risk),
        },
      ].slice(-200));

      setFrames((prevFrames) => [
        ...prevFrames,
        {
          turn: resolvedState.step,
          state: resolvedState,
          result,
          timestamp: Date.now(),
        },
      ].slice(-250));

      if (result.terminated || result.truncated) {
        setRunning(false);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Step request failed.");
      setRunning(false);
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    setRunning(false);
    setReplayPlaying(false);
    try {
      setErrorMessage(null);
      const synced = await resetSimulation({
        seed: 42,
        scenario,
        difficulty,
        attacker,
        strategy_mode: strategyMode,
        action_source: actionSource,
        intelligence_enabled: true,
      });
      setState(synced);
      stateRef.current = synced;
      setIncidents([]);
      setTimeline([
        {
          turn: synced.step,
          reward: 0,
          risk: synced.network_risk,
          uptime: uptimeFromAssets(synced.assets),
          breaches: breachesFromAssets(synced.assets),
          securityScore: securityScore(synced.network_risk),
        },
      ]);
      setFrames([]);
      setReplayIndex(0);
      setActiveAgentLabel(actionSource === "ppo_agent" ? "PPO Defender" : "Colab LLM Defender");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Reset request failed.");
    }
  };

  const handleStartPause = async () => {
    if (running) {
      setRunning(false);
      return;
    }

    try {
      setErrorMessage(null);
      const synced = await resetSimulation({
        seed: 42,
        scenario,
        difficulty,
        attacker,
        strategy_mode: strategyMode,
        action_source: actionSource,
        intelligence_enabled: true,
      });
      setState(synced);
      stateRef.current = synced;
      setIncidents([]);
      setTimeline([
        {
          turn: synced.step,
          reward: 0,
          risk: synced.network_risk,
          uptime: uptimeFromAssets(synced.assets),
          breaches: breachesFromAssets(synced.assets),
          securityScore: securityScore(synced.network_risk),
        },
      ]);
      setFrames([]);
      setReplayIndex(0);
      setActiveAgentLabel(actionSource === "ppo_agent" ? "PPO Defender" : "Colab LLM Defender");
      setRunning(true);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Unable to start simulation.");
    }
  };

  const themeClass = themeByScenario[scenario] ?? themeByScenario.bank;
  const currentConfidence = activeState.intelligence?.reasoning?.decision_confidence ?? 0;
  const security = useMemo(() => securityScore(activeState.network_risk), [activeState.network_risk]);
  const underAttack = Boolean(activeState.red_log?.attack && activeState.red_log?.attack !== "no_attack");

  return (
    <main className={`min-h-screen bg-gradient-to-br ${themeClass} text-slate-100`}>
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_8%_10%,rgba(34,211,238,0.12),transparent_36%),radial-gradient(circle_at_88%_2%,rgba(244,114,182,0.12),transparent_28%)]" />
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="mx-auto flex w-full max-w-[1720px] flex-col gap-4 px-4 py-4 lg:px-8"
      >
        <Topbar
          scenario={scenario}
          difficulty={difficulty}
          attacker={attacker}
          strategyMode={strategyMode}
          actionSource={actionSource}
          activeAgentLabel={activeAgentLabel}
          running={running}
          onScenarioChange={setScenario}
          onDifficultyChange={setDifficulty}
          onAttackerChange={setAttacker}
          onStrategyChange={setStrategyMode}
          onActionSourceChange={setActionSource}
          onStartPause={() => void handleStartPause()}
          onReset={() => void handleReset()}
        />

        {booting ? (
          <div className="rounded-2xl border border-cyan-200/20 bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100">
            Connecting to CySent backend...
          </div>
        ) : null}

        {errorMessage ? (
          <div className="rounded-2xl border border-rose-300/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {errorMessage}
          </div>
        ) : null}

        <motion.section
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08, duration: 0.45 }}
          className="grid grid-cols-1 gap-4 xl:grid-cols-12"
        >
          <motion.section
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.14, duration: 0.42 }}
            className="xl:col-span-8"
          >
            <article className="rounded-3xl border border-cyan-100/10 bg-black/25 p-4 backdrop-blur-xl">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-200/75">CySent Battlefield</p>
                  <h2 className="text-2xl font-semibold text-white">Live Network Operations Theater</h2>
                </div>
                <motion.div
                  animate={underAttack ? { boxShadow: ["0 0 0 rgba(248,113,113,0)", "0 0 24px rgba(248,113,113,0.5)", "0 0 0 rgba(248,113,113,0)"] } : {}}
                  transition={{ duration: 1.6, repeat: underAttack ? Infinity : 0 }}
                  className="rounded-full border border-cyan-200/30 bg-cyan-300/10 px-3 py-1 text-xs text-cyan-100"
                >
                  Risk {activeState.network_risk.toFixed(3)}
                </motion.div>
              </div>

              <NetworkGraph
                assets={activeState.assets}
                redTarget={String(activeState.red_log?.target ?? "")}
                underAttack={underAttack}
              />

              {activeState.assets.length === 0 ? (
                <p className="mt-3 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300">
                  No asset state available yet. Reset or Start to initialize the simulation feed.
                </p>
              ) : null}
            </article>
          </motion.section>

          <motion.aside
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.45 }}
            className="xl:col-span-4"
          >
            <AICommander intelligence={activeState.intelligence} strategyMode={strategyMode} />
          </motion.aside>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.26, duration: 0.45 }}
          className="grid grid-cols-1 gap-4 xl:grid-cols-12"
        >
          <div className="xl:col-span-9">
            <MetricsPanel timeline={timeline} />
          </div>

          <div className="space-y-3 xl:col-span-3">
            <article className="rounded-2xl border border-cyan-100/10 bg-slate-950/65 p-4 backdrop-blur-xl">
              <h3 className="text-xs uppercase tracking-[0.14em] text-slate-300">Mission Snapshot</h3>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <Stat label="Turn" value={String(activeState.step)} />
                <Stat label="Risk" value={activeState.network_risk.toFixed(3)} />
                <Stat label="SecurityScore" value={`${security}`} />
                <Stat label="Confidence" value={`${Math.round(currentConfidence * 100)}%`} />
              </div>
            </article>

            <ReplayControls
              playing={replayPlaying}
              speed={replaySpeed}
              canStepBack={replayIndex > 0}
              canStepForward={replayIndex < frames.length - 1}
              onPlayPause={() => setReplayPlaying((v) => !v)}
              onStepBack={() => setReplayIndex((idx) => Math.max(0, idx - 1))}
              onStepForward={() => setReplayIndex((idx) => Math.min(frames.length - 1, idx + 1))}
              onSpeedChange={setReplaySpeed}
            />
          </div>
        </motion.section>
      </motion.div>

      <div className="fixed bottom-5 right-5 z-40">
        <motion.button
          onClick={() => setTimelineCollapsed((v) => !v)}
          whileHover={{ y: -2 }}
          whileTap={{ scale: 0.98 }}
          className="group flex items-center gap-2 rounded-full border border-cyan-200/30 bg-slate-950/80 px-3 py-2 text-xs uppercase tracking-[0.14em] text-cyan-100 backdrop-blur-xl hover:bg-slate-900"
        >
          <Radar size={14} />
          Timeline
          <ChevronRight
            size={14}
            className={`transition-transform duration-300 ${timelineCollapsed ? "rotate-0" : "rotate-180"}`}
          />
        </motion.button>
      </div>

      <motion.aside
        initial={false}
        animate={{ x: timelineCollapsed ? "86%" : "0%" }}
        transition={{ duration: 0.3 }}
        className="fixed right-0 top-[78px] z-30 h-[calc(100vh-102px)] w-[370px] max-w-[88vw] pr-3"
      >
        <IncidentFeed incidents={incidents} collapsed={timelineCollapsed} onToggle={() => setTimelineCollapsed((v) => !v)} />
      </motion.aside>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-2">
      <p className="text-[10px] uppercase tracking-[0.12em] text-slate-300">{label}</p>
      <p className="mt-1 text-base font-semibold text-slate-100">{value}</p>
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
  const up = assets.filter((a) => a.uptime_status).length;
  return up / assets.length;
}

function breachesFromAssets(assets: EnvState["assets"]): number {
  if (!assets.length) return 0;
  const breached = assets.filter((a) => a.compromised).length;
  return breached / assets.length;
}

function securityScore(networkRisk: number): number {
  return Math.max(0, Math.round((1 - networkRisk) * 100));
}
