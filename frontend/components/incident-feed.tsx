"use client";

import { motion, AnimatePresence } from "framer-motion";

type IncidentFeedProps = {
  incidents: string[];
  collapsed?: boolean;
  onToggle?: () => void;
};

export function IncidentFeed({ incidents, collapsed = false, onToggle }: IncidentFeedProps) {
  return (
    <section className={`dt-panel h-full p-3 transition-all duration-300 ${collapsed ? "w-[62px]" : "w-full"}`}>
      <div className="mb-2 flex items-center justify-between">
        <h2 className={`dt-label ${collapsed ? "hidden" : "block"}`}>Threat Timeline</h2>
        {onToggle && (
          <button onClick={onToggle} className="dt-btn-ghost rounded-md px-2 py-1 text-[10px]">
            {collapsed ? ">" : "<"}
          </button>
        )}
      </div>

      {collapsed && (
        <div className="flex h-[480px] flex-col items-center justify-start gap-2 pt-2">
          {incidents.slice(0, 8).map((entry, idx) => (
            <motion.div
              key={`${entry}-${idx}`}
              initial={{ opacity: 0, scale: 0.85, y: 6 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: idx * 0.03, duration: 0.2 }}
              className="grid h-8 w-8 place-items-center rounded-md bg-white/[0.04] text-sm text-white/40"
            >
              {iconForEntry(entry)}
            </motion.div>
          ))}
        </div>
      )}

      <div className={`space-y-2 overflow-y-auto pr-1 ${collapsed ? "hidden" : "h-[500px]"}`}>
        <AnimatePresence initial={false}>
          {incidents.length === 0 && (
            <motion.p
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="rounded-lg bg-white/[0.03] p-3 text-sm text-white/25"
            >
              Awaiting incident stream. Press Start to run live simulation.
            </motion.p>
          )}
          {incidents.map((entry, index) => (
            <motion.article
              key={`${entry}-${index}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22, delay: Math.min(index * 0.015, 0.15) }}
              className="dt-tile rounded-lg p-2.5"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 text-sm text-white/30">{iconForEntry(entry)}</span>
                <div>
                  <p className="text-[10px] font-medium uppercase tracking-[0.08em] text-orange-400/40">
                    Turn {Math.max(1, incidents.length - index)}
                  </p>
                  <p className="mt-1 text-sm text-white/55">{entry}</p>
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
