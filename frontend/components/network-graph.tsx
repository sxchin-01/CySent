"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";

import { AssetState } from "@/types";

const CytoscapeComponent = dynamic(() => import("react-cytoscapejs"), { ssr: false });

type Props = {
  assets: AssetState[];
};

export function NetworkGraph({ assets }: Props) {
  const nodes = assets.map((asset) => ({
    data: {
      id: asset.name,
      label: asset.name,
      state: asset.compromised ? "compromised" : asset.infected ? "infected" : "healthy",
    },
  }));

  const edges = [
    ["Employee Email", "HR Systems"],
    ["Employee Email", "Web Server"],
    ["Web Server", "Auth Server"],
    ["Auth Server", "Finance Database"],
    ["Backup Infrastructure", "Finance Database"],
    ["SOC Monitoring Console", "Web Server"],
    ["SOC Monitoring Console", "Auth Server"],
  ].map(([source, target], i) => ({ data: { id: `e${i}`, source, target } }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="h-[340px] w-full rounded-xl border border-border/70 bg-[#edf4e8]"
    >
      <CytoscapeComponent
        elements={[...nodes, ...edges]}
        style={{ width: "100%", height: "100%" }}
        layout={{ name: "cose", animate: false, nodeRepulsion: 5500 }}
        stylesheet={[
          {
            selector: "node",
            style: {
              label: "data(label)",
              "font-size": "10px",
              "background-color": "#5f8f45",
              color: "#1f241d",
              "text-wrap": "wrap",
              "text-max-width": "85px",
            },
          },
          {
            selector: "node[state='infected']",
            style: { "background-color": "#f59e0b" },
          },
          {
            selector: "node[state='compromised']",
            style: { "background-color": "#dc2626", color: "#fff8f8" },
          },
          {
            selector: "edge",
            style: { width: 2, "line-color": "#7f8f7d", opacity: 0.7 },
          },
        ]}
      />
    </motion.div>
  );
}
