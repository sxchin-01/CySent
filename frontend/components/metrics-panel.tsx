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

export function MetricsPanel({ timeline }: MetricsPanelProps) {
  return (
    <section className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      <ChartCard title="Risk Score Over Time">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="#253448" opacity={0.35} />
            <XAxis dataKey="turn" stroke="#8fa8bd" />
            <YAxis domain={[0, 1]} stroke="#8fa8bd" />
            <Tooltip />
            <Line type="monotone" dataKey="risk" stroke="#22d3ee" strokeWidth={2.6} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Uptime and Breach Attempts">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="#253448" opacity={0.35} />
            <XAxis dataKey="turn" stroke="#8fa8bd" />
            <YAxis domain={[0, 1]} stroke="#8fa8bd" />
            <Tooltip />
            <Line type="monotone" dataKey="uptime" stroke="#4ade80" strokeWidth={2.6} dot={false} />
            <Line type="monotone" dataKey="breaches" stroke="#f97316" strokeWidth={2.6} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="SecurityScore and Reward Trend">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="#253448" opacity={0.35} />
            <XAxis dataKey="turn" stroke="#8fa8bd" />
            <YAxis stroke="#8fa8bd" />
            <Tooltip />
            <Area type="monotone" dataKey="securityScore" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.3} />
            <Area type="monotone" dataKey="reward" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.24} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
    </section>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <article className="h-[250px] rounded-2xl border border-cyan-100/10 bg-slate-950/70 p-3 shadow-[0_12px_40px_rgba(0,0,0,0.35)] backdrop-blur-xl">
      <h3 className="mb-2 text-[11px] uppercase tracking-[0.16em] text-slate-300">{title}</h3>
      <div className="h-[200px]">{children}</div>
    </article>
  );
}
