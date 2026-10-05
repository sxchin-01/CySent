"use client";

import { useState } from "react";

import { StepResult } from "@/lib/types";

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
] as const;

type ActionControlsProps = {
  latestResult: StepResult | null;
  manualEnabled: boolean;
  busy: boolean;
  onManualAction: (actionName: string, actionId: number) => void;
};

export function ActionControls({ latestResult, manualEnabled, busy, onManualAction }: ActionControlsProps) {
  const [manualActionId, setManualActionId] = useState(0);
  const requested = latestResult?.selected_action_name;
  const executed = latestResult?.action_name;
  const completed = Boolean(latestResult?.terminated || latestResult?.truncated);

  return (
    <section className="dt-panel grid gap-4 p-4 lg:grid-cols-[minmax(250px,0.8fr)_minmax(0,2.2fr)]">
      <div>
        <p className="dt-label">Manual BLUE Action</p>
        <div className="mt-2 flex gap-2">
          <select
            value={manualActionId}
            disabled={!manualEnabled || busy}
            onChange={(event) => setManualActionId(Number(event.target.value))}
            className="min-w-0 flex-1 rounded-md border border-white/[0.08] bg-black/40 px-3 py-2 text-xs text-white/80 outline-none disabled:cursor-not-allowed disabled:opacity-40"
          >
            {ACTIONS.map((action, actionId) => (
              <option key={action} value={actionId}>{formatAction(action)}</option>
            ))}
          </select>
          <button
            type="button"
            disabled={!manualEnabled || busy}
            onClick={() => onManualAction(ACTIONS[manualActionId], manualActionId)}
            className="dt-btn-ghost disabled:cursor-not-allowed disabled:opacity-40"
          >
            Execute
          </button>
        </div>
        <p className="mt-2 text-[11px] text-white/30">
          {manualEnabled ? "Manual execution bypasses autonomous policy prediction." : "Reset or start a nonterminal episode to enable manual actions."}
        </p>
      </div>

      <div>
        <div className="flex items-center justify-between gap-3">
          <p className="dt-label">Latest BLUE Decision</p>
          <span className={`text-xs font-medium ${completed ? "text-amber-300" : "text-white/40"}`}>
            {latestResult ? (completed ? `Completed: ${latestResult.termination_reason}` : `Turn ${latestResult.step}`) : "No action yet"}
          </span>
        </div>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:grid-cols-3 xl:grid-cols-6">
          <DecisionField label="Requested" value={formatAction(requested)} />
          <DecisionField label="Executed" value={formatAction(executed)} />
          <DecisionField label="Mode" value={latestResult?.action_mode ?? "-"} />
          <DecisionField label="Source" value={formatAction(latestResult?.action_source)} />
          <DecisionField label="Reward" value={latestResult ? latestResult.reward.toFixed(3) : "-"} />
          <DecisionField label="Alerts" value={latestResult ? String(latestResult.alerts?.length ?? 0) : "-"} />
        </dl>
        {latestResult?.fallback_reason && (
          <p className="mt-3 rounded-md border border-amber-500/20 bg-amber-500/[0.06] px-3 py-2 text-xs text-amber-200/80">
            Hybrid fallback: {latestResult.fallback_reason}
          </p>
        )}
      </div>
    </section>
  );
}

function DecisionField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase text-white/25">{label}</dt>
      <dd className="mt-0.5 break-words font-medium capitalize text-white/70">{value}</dd>
    </div>
  );
}

function formatAction(value?: string): string {
  if (!value) return "-";
  return value.replaceAll("_", " ");
}
