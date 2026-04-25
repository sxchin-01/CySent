export type PostureLevel = "healthy" | "guarded" | "elevated" | "critical";

export type StrategyMode = "conservative" | "balanced" | "aggressive";
export type ActionSource = "ppo_agent" | "hf_llm_agent" | "hybrid" | "random";

export type AssetState = {
  name: string;
  patch_level: number;
  infected: boolean;
  isolated: boolean;
  compromised: boolean;
  criticality_score: number;
  criticality?: number;
  credential_risk: number;
  detection_level: number;
  backup_health: number;
  backup_status?: number;
  uptime_status: boolean;
  uptime?: number;
  business_dependency?: number;
};

export type RiskBreakdown = {
  asset_exposure: number;
  compromised_hosts: number;
  infected_hosts: number;
  patch_debt: number;
  credential_exposure: number;
  monitoring_weakness: number;
  ransomware_spread: number;
  segmentation_gap: number;
  ignored_alert_pressure: number;
  network_risk: number;
};

export type ForecastEntry = {
  attack: string;
  probability: number;
};

export type IntelligencePayload = {
  enabled: boolean;
  strategy_mode?: StrategyMode;
  action_source?: ActionSource;
  forecast?: {
    primary: ForecastEntry;
    secondary: ForecastEntry;
    top_predictions: ForecastEntry[];
  };
  posture?: {
    level: PostureLevel;
    summary: string;
    highlights: string[];
    network_risk: number;
  };
  reasoning?: {
    action: string;
    strategy_mode: StrategyMode;
    signals: Array<Record<string, unknown>>;
    threat_focus: {
      primary_attack: string;
      primary_probability: number;
    };
    decision_confidence: number;
    explanation: string;
  };
  recommendation?: {
    source: ActionSource | string;
    recommended_action_id: number;
    recommended_action_name: string;
    confidence: number;
    rationale: string;
  };
  action_comparison?: {
    aligned: boolean;
    executed_action: string;
    recommended_action: string;
    delta: string;
  };
  incident_log?: {
    turn: number;
    summary: string;
    decision_confidence: number;
    executed_action: string;
  };
  llm_summary_candidate?: string;
};

export type StepResult = {
  episode_id: string;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  action_name: string;
  active_agent?: string;
  network_risk: number;
  risk_breakdown: Partial<RiskBreakdown>;
  assets: AssetState[];
  red_log: Record<string, unknown>;
  profile?: Record<string, unknown>;
  intelligence?: IntelligencePayload;
  events?: Array<Record<string, unknown>>;
  narrative?: string;
  metrics: Record<string, number>;
  termination_reason: string;
};

export type EnvState = {
  episode_id: string;
  step: number;
  network_risk: number;
  risk_breakdown: Partial<RiskBreakdown>;
  assets: AssetState[];
  last_action: string;
  red_log: Record<string, unknown>;
  profile?: Record<string, unknown>;
  intelligence?: IntelligencePayload;
  events?: Array<Record<string, unknown>>;
  narrative?: string;
  termination_reason: string;
};

export type TimelinePoint = {
  turn: number;
  risk: number;
  uptime: number;
  breaches: number;
  reward: number;
  securityScore: number;
};

export type TimelineFrame = {
  turn: number;
  state: EnvState;
  result: StepResult;
  timestamp: number;
};
