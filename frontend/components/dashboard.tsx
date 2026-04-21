"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Shield, Siren, BrainCircuit, Gauge, Sparkles } from "lucide-react";

import { fetchState, step } from "@/lib/api";
import { NetworkGraph } from "@/components/network-graph";
import { UICard } from "@/components/ui-card";
import { Button } from "@/components/ui/button";
import { AssetState, EnvState } from "@/types";

const ACTIONS = [
  "do_nothing",
  "patch_hr_systems",
  "patch_web_server",
  "patch_auth_server",
  "rotate_credentials",
  "isolate_suspicious_host",
  "increase_monitoring",
  "restore_backup",
  "deploy_honeypot",
  "phishing_training",
  "investigate_top_alert",
  "segment_finance_database",
];

type TelemetryPoint = {
  turn: number;
  reward: number;
  risk: number;
  uptime: number;
  breaches: number;
};

const initialState: EnvState = {
  episode_id: "",
  step: 0,
  network_risk: 0,
  risk_breakdown: {},
  assets: [],
  last_action: "reset",
  red_log: {},
  termination_reason: "active",
};

export function Dashboard() {
  const [state, setState] = useState<EnvState>(initialState);
  const [feed, setFeed] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const boot = async () => {
      const s = await fetchState();
      setState(s);
      setTelemetry([
        {
          turn: s.step,
          reward: 0,
          risk: s.network_risk,
          uptime: uptimeFromAssets(s.assets),
          breaches: breachesFromAssets(s.assets),
        },
      ]);
    };

    void boot();
  }, []);

  const score = useMemo(() => Math.max(0, Math.round((1 - state.network_risk) * 100)), [state.network_risk]);

  const runAction = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const result = await step();
      setState((prev: EnvState) => ({
        ...prev,
        episode_id: result.episode_id,
        step: prev.step + 1,
        network_risk: result.network_risk,
        risk_breakdown: result.risk_breakdown,
        assets: result.assets,
        last_action: result.action_name,
        red_log: result.red_log,
        termination_reason: result.termination_reason,
      }));

      const summary = `${result.action_name} | reward ${result.reward.toFixed(2)} | red ${String(result.red_log.attack ?? "n/a")} on ${String(result.red_log.target ?? "n/a")}`;
      setFeed((prev: string[]) => [summary, ...prev].slice(0, 12));
      setTelemetry((prev: TelemetryPoint[]) => [
        ...prev,
        {
          turn: prev.length,
          reward: result.reward,
          risk: result.network_risk,
          uptime: uptimeFromAssets(result.assets),
          breaches: breachesFromAssets(result.assets),
        },
      ].slice(-120));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto flex w-full max-w-[1500px] flex-col gap-4 px-4 py-5 lg:px-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-emerald-700/70">CySent</p>
          <h1 className="text-3xl font-semibold text-emerald-950">AI Security Operations Commander</h1>
        </div>
        <div className="rounded-full border border-emerald-800/25 bg-emerald-100/70 px-4 py-2 text-sm font-medium text-emerald-900">
          Episode {state.episode_id.slice(0, 8) || "-"}
        </div>
      </header>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <UICard className="xl:col-span-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium text-emerald-900">
            <Shield size={16} /> Enterprise Topology
          </div>
          <NetworkGraph assets={state.assets} />
        </UICard>

        <UICard className="xl:col-span-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-amber-900">
            <Siren size={16} /> Live Attack Feed
          </div>
          <div className="h-[340px] space-y-2 overflow-auto rounded-xl bg-[#f5f6e9] p-3 font-[var(--font-mono)] text-xs">
            {feed.length === 0 ? <p>No events yet. Execute a Blue action.</p> : null}
            {feed.map((line: string, idx: number) => (
              <motion.p
                key={`${line}-${idx}`}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
                className="rounded border border-amber-200/80 bg-amber-50/70 px-2 py-1"
              >
                {line}
              </motion.p>
            ))}
          </div>
        </UICard>

        <UICard className="xl:col-span-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-cyan-900">
            <BrainCircuit size={16} /> Blue Commander
          </div>
          <div className="mb-3 rounded-xl border border-cyan-800/25 bg-cyan-50/70 p-3">
            <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-900">
              <Sparkles size={14} /> AI Action Reasoning
            </p>
            <div className="space-y-1 text-xs text-cyan-900/90">
              {deriveRecommendations(state).map((reason, idx) => (
                <p key={`${reason}-${idx}`} className="rounded bg-white/70 px-2 py-1">
                  {reason}
                </p>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {ACTIONS.map((name, idx) => (
              <Button
                key={name}
                onClick={() => void runAction()}
                disabled={busy}
                variant="outline"
                size="sm"
                className="h-auto justify-start border-cyan-900/20 bg-cyan-50 px-2 py-2 text-left text-xs hover:bg-cyan-100"
              >
                <p className="font-semibold text-cyan-900">{idx}</p>
                <p className="font-[var(--font-mono)] text-cyan-800">{name}</p>
              </Button>
            ))}
          </div>
        </UICard>
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <UICard className="xl:col-span-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium text-emerald-900">
            <Gauge size={16} /> Security Score
          </div>
          <p className="text-5xl font-bold text-emerald-950">{score}</p>
          <p className="mt-2 text-sm text-emerald-900/80">Network risk: {state.network_risk.toFixed(3)}</p>
          <p className="mt-1 text-xs text-emerald-900/70">
            Segment gap: {(state.risk_breakdown.segmentation_gap ?? 0).toFixed(2)} | Monitoring weakness: {(state.risk_breakdown.monitoring_weakness ?? 0).toFixed(2)}
          </p>
        </UICard>

        <UICard className="xl:col-span-3">
          <p className="text-sm font-medium text-slate-800">Breach Trend</p>
          <div className="mt-2 h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8e3cf" />
                <XAxis dataKey="turn" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="breaches" stroke="#b91c1c" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </UICard>

        <UICard className="xl:col-span-3">
          <p className="text-sm font-medium text-slate-800">Reward Curve</p>
          <div className="mt-2 h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={telemetry}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8e3cf" />
                <XAxis dataKey="turn" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="reward" stroke="#0f766e" fill="#99f6e4" fillOpacity={0.65} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </UICard>

        <UICard className="xl:col-span-3">
          <p className="text-sm font-medium text-slate-800">Uptime</p>
          <div className="mt-2 h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8e3cf" />
                <XAxis dataKey="turn" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="uptime" stroke="#15803d" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="risk" stroke="#0369a1" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </UICard>
      </section>
    </main>
  );
}

function deriveRecommendations(state: EnvState): string[] {
  const breakdown = state.risk_breakdown ?? {};
  const tips: string[] = [];

  if ((breakdown.segmentation_gap ?? 0) > 0.5) {
    tips.push("High lateral exposure: prioritize segment_finance_database.");
  }
  if ((breakdown.credential_exposure ?? 0) > 0.45) {
    tips.push("Credential risk elevated: rotate_credentials or phishing_training.");
  }
  if ((breakdown.monitoring_weakness ?? 0) > 0.45) {
    tips.push("Detection weak: increase_monitoring then investigate_top_alert.");
  }
  if ((breakdown.ransomware_spread ?? 0) > 0.4) {
    tips.push("Ransomware pressure rising: restore_backup and isolate_suspicious_host.");
  }

  const lastAttack = String(state.red_log.attack ?? "");
  if (lastAttack === "phishing_email" || lastAttack === "credential_theft") {
    tips.push("Recent credential-focused attack observed: reinforce email and auth defenses.");
  }

  if (tips.length === 0) {
    tips.push("Risk posture stable: patch critical servers and preserve action economy.");
  }

  return tips.slice(0, 4);
}

function uptimeFromAssets(assets: AssetState[]): number {
  if (!assets.length) return 1;
  const up = assets.filter((a) => a.uptime_status).length;
  return up / assets.length;
}

function breachesFromAssets(assets: AssetState[]): number {
  if (!assets.length) return 0;
  const breached = assets.filter((a) => a.compromised).length;
  return breached / assets.length;
}
