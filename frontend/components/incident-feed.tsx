"use client";

import { motion, AnimatePresence } from "framer-motion";

type IncidentFeedProps = {
  incidents: string[];
  collapsed?: boolean;
  onToggle?: () => void;
};

export function IncidentFeed({ incidents, collapsed = false, onToggle }: IncidentFeedProps) {
  return (
    <section
      className={`h-full rounded-2xl border border-cyan-100/10 bg-slate-950/75 p-3 backdrop-blur-xl transition-all duration-300 ${
        collapsed ? "w-[62px]" : "w-full"
      }`}
    >
      <div className="mb-2 flex items-center justify-between">
        <h2 className={`text-xs uppercase tracking-[0.14em] text-slate-300 ${collapsed ? "hidden" : "block"}`}>Threat Timeline</h2>
        {onToggle ? (
          <button
            onClick={onToggle}
            className="rounded-md border border-white/15 bg-white/5 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-slate-200 hover:bg-white/10"
          >
            {collapsed ? ">" : "<"}
          </button>
        ) : null}
      </div>

      {collapsed ? (
        <div className="flex h-[480px] flex-col items-center justify-start gap-2 pt-2">
          {incidents.slice(0, 8).map((entry, idx) => (
            <motion.div
              key={`${entry}-${idx}`}
              initial={{ opacity: 0, scale: 0.85, y: 6 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: idx * 0.03, duration: 0.2 }}
              whileHover={{ scale: 1.06, y: -1 }}
              className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-300/25 bg-cyan-400/10 text-sm shadow-[0_0_16px_rgba(34,211,238,0.2)]"
            >
              {iconForEntry(entry)}
            </motion.div>
          ))}
        </div>
      ) : null}

      <div className={`space-y-2 overflow-y-auto pr-1 ${collapsed ? "hidden" : "h-[500px]"}`}>
        <AnimatePresence initial={false}>
          {incidents.length === 0 ? (
            <motion.p
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-slate-300"
            >
              Awaiting incident stream. Press Start to run live simulation.
            </motion.p>
          ) : null}
          {incidents.map((entry, index) => (
            <motion.article
              key={`${entry}-${index}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22, delay: Math.min(index * 0.015, 0.15) }}
              whileHover={{ y: -2 }}
              className="rounded-xl border border-cyan-300/20 bg-slate-900/80 p-2.5"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 text-sm">{iconForEntry(entry)}</span>
                <div>
                  <p className="text-[10px] uppercase tracking-[0.14em] text-cyan-300/80">Turn {Math.max(1, incidents.length - index)}</p>
                  <p className="mt-1 text-sm text-slate-100">{entry}</p>
                </div>
              </div>
            </motion.article>
          ))}
        </AnimatePresence>
      </div>
    </section>
  );
}

function iconForEntry(entry: string): string {
  const text = entry.toLowerCase();
  if (text.includes("blocked") || text.includes("disrupted") || text.includes("defense")) return "🛡";
  if (text.includes("phishing") || text.includes("credential") || text.includes("ransomware")) return "⚠";
  if (text.includes("rotated") || text.includes("patched") || text.includes("segmented")) return "✅";
  if (text.includes("lateral") || text.includes("exfiltration")) return "🚫";
  return "•";
}
