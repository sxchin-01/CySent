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
    () => {
      const nodeIds = new Set(assets.map((asset) => asset.name));
      return EDGE_MAP.filter(([source, target]) => nodeIds.has(source) && nodeIds.has(target)).map(
        ([source, target], i) => {
          const active = underAttack && redTarget && (source === redTarget || target === redTarget);
          return { data: { id: `e${i}`, source, target, active: active ? "yes" : "no" } };
        },
      );
    },
    [assets, redTarget, underAttack],
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="relative h-[560px] w-full overflow-hidden rounded-xl bg-black/60 ring-1 ring-white/[0.04]"
    >
      {/* Grid overlay */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:48px_48px]" />

      {/* Focal glow */}
      <motion.div
        className="pointer-events-none absolute left-1/3 top-1/4 h-60 w-60 rounded-full bg-orange-500/[0.04] blur-[80px]"
        animate={{ x: [0, 40, 0], opacity: [0.4, 0.7, 0.4] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="pointer-events-none absolute bottom-1/4 right-1/4 h-40 w-40 rounded-full bg-cyan-500/[0.03] blur-[60px]"
        animate={{ x: [0, -30, 0], opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
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
              width: 52,
              height: 52,
              color: "#ffffff",
              "text-wrap": "wrap",
              "text-max-width": "100px",
              "border-width": 1.5,
              "border-color": "rgba(255,255,255,0.1)",
              "overlay-opacity": 0,
              "text-margin-y": 34,
              "shadow-color": "#10b981",
              "shadow-blur": 12,
              "shadow-opacity": 0.3,
            },
          },
          {
            selector: "node[state='patched']",
            style: {
              "background-color": "#3b82f6",
              "shadow-color": "#60a5fa",
              "shadow-blur": 20,
              "shadow-opacity": 0.6,
            },
          },
          {
            selector: "node[state='infected']",
            style: {
              "background-color": "#f59e0b",
              "shadow-color": "#f59e0b",
              "shadow-blur": 20,
              "shadow-opacity": 0.6,
            },
          },
          {
            selector: "node[state='isolated']",
            style: {
              "background-color": "#4b5563",
              "border-color": "rgba(255,255,255,0.08)",
              "shadow-opacity": 0,
            },
          },
          {
            selector: "node[state='compromised']",
            style: {
              "background-color": "#ef4444",
              "shadow-color": "#ef4444",
              "shadow-blur": 28,
              "shadow-opacity": 0.8,
            },
          },
          {
            selector: "node[critical='yes']",
            style: {
              "border-color": "#f59e0b",
              "border-width": 3,
            },
          },
          {
            selector: "edge",
            style: {
              width: 1.5,
              opacity: 0.35,
              "line-color": "rgba(255,255,255,0.15)",
              "target-arrow-color": "rgba(255,255,255,0.15)",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
            },
          },
          {
            selector: "edge[active='yes']",
            style: {
              width: 3.5,
              opacity: 1,
              "line-color": "#f06530",
              "target-arrow-color": "#f06530",
              "line-style": "dashed",
              "line-dash-pattern": [8, 6],
              "shadow-blur": 16,
              "shadow-color": "#f06530",
              "shadow-opacity": 0.7,
            },
          },
        ]}
      />

      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-black/40 px-2.5 py-1 text-[10px] uppercase tracking-[0.1em] text-white/20 backdrop-blur">
        Battlefield Live / Zoom + Pan
      </div>
    </motion.div>
  );
}
