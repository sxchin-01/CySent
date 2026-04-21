"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

import { AssetState } from "@/lib/types";

const CytoscapeComponent: any = dynamic(() => import("react-cytoscapejs") as any, { ssr: false });

type Props = {
  assets: AssetState[];
  redTarget?: string;
  underAttack?: boolean;
};

const EDGE_MAP: Array<[string, string]> = [
  ["Employee Email", "HR Systems"],
  ["Employee Email", "Web Server"],
  ["Web Server", "Auth Server"],
  ["Auth Server", "Finance Database"],
  ["Backup Infrastructure", "Finance Database"],
  ["SOC Monitoring Console", "Web Server"],
  ["SOC Monitoring Console", "Auth Server"],
];

export function NetworkGraph({ assets, redTarget, underAttack = false }: Props) {
  const nodeElements = useMemo(
    () =>
      assets.map((asset) => {
        const isPatched = asset.patch_level >= 0.78;
        const state = asset.compromised
          ? "compromised"
          : asset.infected
            ? "infected"
            : asset.isolated
              ? "isolated"
              : isPatched
                ? "patched"
                : "healthy";

        const criticality = asset.criticality_score ?? asset.criticality ?? 0.5;
        return {
          data: {
            id: asset.name,
            label: asset.name,
            state,
            critical: criticality > 0.85 ? "yes" : "no",
            tooltip: [
              `Patch: ${(asset.patch_level * 100).toFixed(0)}%`,
              `Credential Risk: ${(asset.credential_risk * 100).toFixed(0)}%`,
              `Detection: ${(asset.detection_level * 100).toFixed(0)}%`,
            ].join(" | "),
          },
        };
      }),
    [assets],
  );

  const edgeElements = useMemo(
    () =>
      EDGE_MAP.map(([source, target], i) => {
        const active = underAttack && redTarget && (source === redTarget || target === redTarget);
        return {
          data: { id: `e${i}`, source, target, active: active ? "yes" : "no" },
        };
      }),
    [redTarget, underAttack],
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="relative h-[560px] w-full overflow-hidden rounded-3xl border border-cyan-100/10 bg-slate-950/40"
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_15%,rgba(34,211,238,0.08),transparent_38%),radial-gradient(circle_at_82%_0%,rgba(244,114,182,0.08),transparent_35%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:38px_38px] opacity-30" />
      <motion.div
        className="pointer-events-none absolute -left-20 top-1/3 h-40 w-40 rounded-full bg-cyan-400/10 blur-3xl"
        animate={{ x: [0, 30, 0], opacity: [0.3, 0.55, 0.3] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="pointer-events-none absolute left-0 top-0 h-12 w-full bg-gradient-to-b from-cyan-300/20 to-transparent"
        animate={{ y: [-40, 560, -40], opacity: [0, 0.6, 0] }}
        transition={{ duration: 5.8, repeat: Infinity, ease: "linear" }}
      />

      <CytoscapeComponent
        elements={[...nodeElements, ...edgeElements]}
        style={{ width: "100%", height: "100%" }}
        userZoomingEnabled
        userPanningEnabled
        boxSelectionEnabled={false}
        minZoom={0.45}
        maxZoom={2.2}
        layout={{ name: "cose", animate: false, nodeRepulsion: 7000, idealEdgeLength: 130 }}
        stylesheet={[
          {
            selector: "node",
            style: {
              label: "data(label)",
              "font-size": "10px",
              "font-family": "var(--font-display)",
              "text-valign": "center",
              "text-halign": "center",
              "background-color": "#10b981",
              width: 56,
              height: 56,
              color: "#f8fafc",
              "text-wrap": "wrap",
              "text-max-width": "100px",
              "border-width": 1.2,
              "border-color": "#d1fae5",
              "overlay-opacity": 0,
              "text-margin-y": 34,
            },
          },
          {
            selector: "node[state='patched']",
            style: {
              "background-color": "#2563eb",
              "shadow-color": "#60a5fa",
              "shadow-blur": 18,
              "shadow-opacity": 0.75,
            },
          },
          {
            selector: "node[state='infected']",
            style: {
              "background-color": "#f59e0b",
              "shadow-color": "#f59e0b",
              "shadow-blur": 16,
              "shadow-opacity": 0.65,
            },
          },
          {
            selector: "node[state='isolated']",
            style: { "background-color": "#6b7280", "border-color": "#94a3b8" },
          },
          {
            selector: "node[state='compromised']",
            style: {
              "background-color": "#dc2626",
              color: "#fff8f8",
              "shadow-color": "#f43f5e",
              "shadow-blur": 22,
              "shadow-opacity": 0.85,
            },
          },
          {
            selector: "node[critical='yes']",
            style: {
              "border-color": "#fbbf24",
              "border-width": 2.6,
            },
          },
          {
            selector: "edge",
            style: {
              width: 2,
              opacity: 0.55,
              "line-color": "#64748b",
              "target-arrow-color": "#64748b",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
            },
          },
          {
            selector: "edge[active='yes']",
            style: {
              width: 4,
              opacity: 1,
              "line-color": "#f59e0b",
              "target-arrow-color": "#f59e0b",
              "line-style": "dashed",
              "line-dash-pattern": [8, 6],
              "shadow-blur": 12,
              "shadow-color": "#f59e0b",
              "shadow-opacity": 0.7,
            },
          },
        ]}
      />

      <div className="pointer-events-none absolute bottom-2 left-3 rounded-lg border border-white/10 bg-black/40 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-slate-300">
        Battlefield Live / Zoom + Pan Enabled
      </div>
    </motion.div>
  );
}
