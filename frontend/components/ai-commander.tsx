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
    <section className="relative h-[560px] rounded-3xl border border-fuchsia-200/10 bg-[linear-gradient(170deg,rgba(15,23,42,0.92),rgba(3,7,18,0.94))] p-4 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur-xl">
      <motion.div
        className="pointer-events-none absolute inset-x-4 top-3 h-px bg-gradient-to-r from-transparent via-fuchsia-300/45 to-transparent"
        animate={{ opacity: [0.3, 0.8, 0.3] }}
        transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
      />
      <header className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-[0.16em] text-fuchsia-200/75">AI Command Center</p>
          <h2 className="text-lg font-semibold text-white">Autonomous Blue Decision Core</h2>
        </div>
        <span className="rounded-full border border-cyan-300/35 bg-cyan-400/10 px-2 py-1 text-[10px] uppercase tracking-[0.16em] text-cyan-100">
          {strategyMode}
        </span>
      </header>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.06 } },
        }}
        className="grid grid-cols-1 gap-3"
      >
        <motion.section
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
          whileHover={{ y: -2 }}
          className="rounded-2xl border border-white/10 bg-white/5 p-3"
        >
          <p className="text-[10px] uppercase tracking-[0.14em] text-slate-300">Recommended Action</p>
          <p className="mt-1 text-xl font-semibold text-white">{formatAction(recommendation?.recommended_action_name)}</p>
          <p className="mt-1 text-xs text-slate-300">{recommendation?.rationale ?? "AI recommendation updates every turn."}</p>
        </motion.section>

        <motion.section
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
          className="grid grid-cols-2 gap-3"
        >
          <motion.div whileHover={{ y: -2 }} className="rounded-2xl border border-fuchsia-300/20 bg-fuchsia-400/5 p-3">
            <p className="text-[10px] uppercase tracking-[0.14em] text-fuchsia-100/85">Confidence</p>
            <RadialConfidence percent={confidencePct} />
          </motion.div>

          <motion.div whileHover={{ y: -2 }} className="rounded-2xl border border-amber-300/20 bg-amber-400/5 p-3">
            <p className="text-[10px] uppercase tracking-[0.14em] text-amber-100/90">Security Posture</p>
            <p className="mt-2 text-base font-semibold capitalize text-amber-50">{posture?.level ?? "guarded"}</p>
            <p className="mt-1 line-clamp-4 text-xs text-amber-100/80">{posture?.summary ?? "Posture summary unavailable."}</p>
          </motion.div>
        </motion.section>

        <motion.section
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
          whileHover={{ y: -2 }}
          className="rounded-2xl border border-cyan-300/20 bg-cyan-400/5 p-3"
        >
          <p className="text-[10px] uppercase tracking-[0.14em] text-cyan-100/90">Threat Forecast</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {(forecast?.top_predictions ?? []).slice(0, 4).map((item) => (
              <span
                key={`${item.attack}-${item.probability}`}
                className="rounded-full border border-cyan-200/30 bg-cyan-300/10 px-2 py-1 text-[11px] text-cyan-50"
              >
                {item.attack} {Math.round(item.probability * 100)}%
              </span>
            ))}
            {!forecast?.top_predictions?.length ? (
              <span className="text-xs text-cyan-100/80">Forecast data pending.</span>
            ) : null}
          </div>
        </motion.section>

        <motion.section
          variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
          whileHover={{ y: -2 }}
          className="rounded-2xl border border-white/10 bg-white/5 p-3"
        >
          <p className="text-[10px] uppercase tracking-[0.14em] text-slate-300">Why This Action</p>
          <p className="mt-2 line-clamp-6 text-sm leading-relaxed text-slate-100">
            {reasoning?.explanation ?? "Reasoning context appears once the simulation executes the first action."}
          </p>
        </motion.section>
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
      <svg width="120" height="120" viewBox="0 0 120 120" className="drop-shadow-[0_0_20px_rgba(244,114,182,0.35)]">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(244,114,182,0.15)" strokeWidth="10" />
        <motion.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="rgba(244,114,182,0.95)"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.45 }}
          transform="rotate(-90 60 60)"
        />
        <text x="60" y="64" textAnchor="middle" className="fill-fuchsia-50 text-lg font-semibold">
          {clamped}%
        </text>
      </svg>
    </div>
  );
}

function formatAction(value?: string): string {
  if (!value) return "Pending";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
