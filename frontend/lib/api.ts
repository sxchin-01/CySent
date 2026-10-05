import { AgentsResponse, EnvState, StepResult } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 12_000;

function buildApiBases(): string[] {
  const bases = new Set<string>();
  const normalizedPrimary = API_BASE.replace(/\/$/, "");
  bases.add(normalizedPrimary);

  // Browser fallback only for local loopback host mismatches.
  if (typeof window !== "undefined" && isLoopbackUrl(normalizedPrimary)) {
    const browserLoopbacks = ["http://127.0.0.1:8000", "http://localhost:8000"];
    for (const base of browserLoopbacks) {
      bases.add(base);
    }
  }

  return Array.from(bases);
}

function isLoopbackUrl(value: string): boolean {
  try {
    const hostname = new URL(value).hostname;
    return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
  } catch {
    return false;
  }
}

function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (!detail || typeof detail !== "object") return String(detail ?? "");

  const payload = detail as Record<string, unknown>;
  const parts = [payload.message, payload.reason, payload.hint, payload.error]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0);
  return Array.from(new Set(parts)).join(" ") || JSON.stringify(payload);
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
            detail = ` - ${formatErrorDetail(payload.detail)}`;
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

export async function fetchAgents(): Promise<AgentsResponse> {
  return requestJson<AgentsResponse>("/agents");
}

export async function step(): Promise<StepResult> {
  return requestJson<StepResult>("/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export async function stepWithActionName(actionName: string, actionId: number): Promise<StepResult> {
  return requestJson<StepResult>("/step/manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: actionId, action_name: actionName }),
  });
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
