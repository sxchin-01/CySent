"use client";

import type { ReactNode } from "react";

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

import { TimelinePoint } from "@/lib/types";

type MetricsPanelProps = {
  timeline: TimelinePoint[];
};

const tooltipStyle = {
  background: "rgba(10, 10, 14, 0.95)",
  border: "1px solid rgba(255, 255, 255, 0.08)",
  borderRadius: 10,
  fontSize: 11,
  color: "rgba(255,255,255,0.7)",
  boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
};

export function MetricsPanel({ timeline }: MetricsPanelProps) {
  return (
    <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <ChartCard title="Risk Score" accentColor="#f06530">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="turn" stroke="rgba(255,255,255,0.15)" fontSize={10} />
            <YAxis domain={[0, 1]} stroke="rgba(255,255,255,0.15)" fontSize={10} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey="risk" stroke="#f06530" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Uptime & Breaches" accentColor="#4ade80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="turn" stroke="rgba(255,255,255,0.15)" fontSize={10} />
            <YAxis domain={[0, 1]} stroke="rgba(255,255,255,0.15)" fontSize={10} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey="uptime" stroke="#4ade80" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="breaches" stroke="#f97316" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Score & Reward" accentColor="#a78bfa">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="turn" stroke="rgba(255,255,255,0.15)" fontSize={10} />
            <YAxis stroke="rgba(255,255,255,0.15)" fontSize={10} />
            <Tooltip contentStyle={tooltipStyle} />
            <Area type="monotone" dataKey="securityScore" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.08} />
            <Area type="monotone" dataKey="reward" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.06} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
    </section>
  );
}

function ChartCard({ title, accentColor, children }: { title: string; accentColor: string; children: ReactNode }) {
  return (
    <article className="dt-panel relative h-[250px] overflow-hidden p-4">
      <div
        className="pointer-events-none absolute left-0 top-0 h-[2px] w-full"
        style={{ background: `linear-gradient(90deg, transparent, ${accentColor}40, transparent)` }}
      />
      <h3 className="dt-label mb-3">{title}</h3>
      <div className="h-[190px]">{children}</div>
    </article>
  );
}
