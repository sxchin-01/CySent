"use client";

import { motion } from "framer-motion";

import { IntelligencePayload } from "@/lib/types";

type AICommanderProps = {
  intelligence?: IntelligencePayload;
  strategyMode: string;
};

export function AICommander({ intelligence, strategyMode }: AICommanderProps) {
  const recommendation = intelligence?.recommendation;
  const reasoning = intelligence?.reasoning;
  const forecast = intelligence?.forecast;
  const posture = intelligence?.posture;

  const confidence = reasoning?.decision_confidence ?? recommendation?.confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);

  return (
    <section className="dt-panel relative flex h-[560px] flex-col overflow-hidden p-5">
      <header className="mb-5 flex items-center justify-between">
        <div>
          <p className="dt-label text-orange-400/60">AI Command Center</p>
          <h3 className="mt-1 text-lg font-semibold text-white">Blue Decision Core</h3>
        </div>
        <span className="rounded-full bg-white/[0.05] px-3 py-1 text-[10px] uppercase tracking-[0.12em] font-medium text-white/40">
          {strategyMode}
        </span>
      </header>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{ hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.06 } } }}
        className="flex flex-1 flex-col gap-3 overflow-y-auto"
      >
        {/* Recommended Action */}
        <motion.div
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
          className="dt-tile p-4"
        >
          <p className="dt-label">Recommended Action</p>
          <p className="mt-2 text-xl font-bold text-white">{formatAction(recommendation?.recommended_action_name)}</p>
          <p className="mt-1 text-xs text-white/35">{recommendation?.rationale ?? "AI recommendation updates every turn."}</p>
        </motion.div>

        {/* Confidence + Posture */}
        <motion.div variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }} className="grid grid-cols-2 gap-3">
          <div className="dt-tile flex flex-col items-center p-4">
            <p className="dt-label self-start text-orange-400/50">Confidence</p>
            <RadialConfidence percent={confidencePct} />
          </div>
          <div className="dt-tile p-4">
            <p className="dt-label text-amber-400/50">Security Posture</p>
            <p className="mt-3 text-base font-bold capitalize text-amber-200/90">{posture?.level ?? "guarded"}</p>
            <p className="mt-1 line-clamp-4 text-xs text-white/30">{posture?.summary ?? "Posture summary unavailable."}</p>
          </div>
        </motion.div>

        {/* Threat Forecast */}
        <motion.div variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }} className="dt-tile p-4">
          <p className="dt-label text-cyan-400/50">Threat Forecast</p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {(forecast?.top_predictions ?? []).slice(0, 4).map((item) => (
              <span
                key={`${item.attack}-${item.probability}`}
                className="rounded-full bg-white/[0.05] px-3 py-1 text-[11px] font-medium text-white/60"
              >
                {item.attack} <span className="text-orange-400/80">{Math.round(item.probability * 100)}%</span>
              </span>
            ))}
            {!forecast?.top_predictions?.length && (
              <span className="text-xs text-white/25">Forecast data pending.</span>
            )}
          </div>
        </motion.div>

        {/* Why This Action */}
        <motion.div variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }} className="dt-tile p-4">
          <p className="dt-label">Why This Action</p>
          <p className="mt-2 line-clamp-6 text-sm leading-relaxed text-white/50">
            {reasoning?.explanation ?? "Reasoning context appears once the simulation executes the first action."}
          </p>
        </motion.div>
      </motion.div>
    </section>
  );
}

function RadialConfidence({ percent }: { percent: number }) {
  const clamped = Math.max(0, Math.min(100, percent));
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="mt-2 flex items-center justify-center">
      <svg width="110" height="110" viewBox="0 0 120 120" className="drop-shadow-[0_0_20px_rgba(240,100,48,0.2)]">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="7" />
        <motion.circle
          cx="60" cy="60" r={radius} fill="none"
          stroke="url(#conf-grad)" strokeWidth="7" strokeLinecap="round"
          strokeDasharray={circumference}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.45 }}
          transform="rotate(-90 60 60)"
        />
        <defs>
          <linearGradient id="conf-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f06530" />
            <stop offset="100%" stopColor="#e84525" />
          </linearGradient>
        </defs>
        <text x="60" y="64" textAnchor="middle" className="fill-white text-lg font-bold">{clamped}%</text>
      </svg>
    </div>
  );
}

function formatAction(value?: string): string {
  if (!value) return "Pending";
  return value.split("_").map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(" ");
}
