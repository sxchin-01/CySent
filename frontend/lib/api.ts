import { EnvState, StepResult } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 12_000;

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      cache: init?.cache ?? "no-store",
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`API ${path} failed: ${res.status} ${res.statusText}`);
    }

    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`API timeout after ${REQUEST_TIMEOUT_MS / 1000}s at ${API_BASE}${path}.`);
    }
    if (err instanceof TypeError) {
      throw new Error(
        `Cannot reach backend at ${API_BASE}. Start the API server and retry. (${err.message})`,
      );
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchState(): Promise<EnvState> {
  return requestJson<EnvState>("/state");
}

export async function step(): Promise<StepResult> {
  return requestJson<StepResult>("/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export async function stepWithActionName(_actionName: string, _actionId: number): Promise<StepResult> {
  return step();
}

export async function resetSimulation(payload: {
  seed: number;
  scenario: string;
  difficulty: string;
  attacker: string;
  strategy_mode: string;
  action_source: string;
  intelligence_enabled: boolean;
}): Promise<EnvState> {
  return requestJson<EnvState>("/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchTrainingStatus(): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>("/training-status");
}

export async function runBenchmark(): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>("/benchmark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}
