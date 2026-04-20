import { EnvState, StepResult } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function fetchState(): Promise<EnvState> {
  const res = await fetch(`${API_BASE}/state`, { cache: "no-store" });
  if (!res.ok) throw new Error(`State request failed: ${res.status}`);
  return res.json();
}

export async function step(action: number): Promise<StepResult> {
  const res = await fetch(`${API_BASE}/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) throw new Error(`Step request failed: ${res.status}`);
  return res.json();
}

export async function fetchTrainingStatus(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/training-status`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Training status request failed: ${res.status}`);
  return res.json();
}

export async function runBenchmark(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`Benchmark request failed: ${res.status}`);
  return res.json();
}
