import { EnvState, StepResult } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 12_000;

function buildApiBases(): string[] {
  const bases = new Set<string>();
  const normalizedPrimary = API_BASE.replace(/\/$/, "");
  bases.add(normalizedPrimary);

  // Browser fallback for loopback host mismatches (localhost vs 127.0.0.1).
  if (typeof window !== "undefined") {
    const browserLoopbacks = ["http://127.0.0.1:8000", "http://localhost:8000"];
    for (const base of browserLoopbacks) {
      bases.add(base);
    }
  }

  return Array.from(bases);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const apiBases = buildApiBases();
  let lastTypeError: TypeError | null = null;
  let lastBase = API_BASE;

  for (const base of apiBases) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    lastBase = base;

    try {
      const res = await fetch(`${base}${path}`, {
        ...init,
        cache: init?.cache ?? "no-store",
        signal: controller.signal,
      });

      if (!res.ok) {
        let detail = "";
        try {
          const payload = await res.clone().json() as { detail?: unknown };
          if (payload && payload.detail !== undefined) {
            detail = ` - ${String(payload.detail)}`;
          }
        } catch {
          try {
            const text = (await res.text()).trim();
            if (text) {
              detail = ` - ${text}`;
            }
          } catch {
            // no-op: keep status-only message
          }
        }
        throw new Error(`API ${path} failed: ${res.status} ${res.statusText}${detail}`);
      }

      return (await res.json()) as T;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new Error(`API timeout after ${REQUEST_TIMEOUT_MS / 1000}s at ${base}${path}.`);
      }

      if (err instanceof TypeError) {
        lastTypeError = err;
        continue;
      }

      throw err;
    } finally {
      clearTimeout(timeout);
    }
  }

  if (lastTypeError) {
    throw new Error(
      `Cannot reach backend at ${lastBase}. Start the API server and retry. (${lastTypeError.message})`,
    );
  }

  throw new Error(`Cannot reach backend at ${lastBase}. Start the API server and retry.`);
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
