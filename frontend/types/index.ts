export type AssetState = {
  name: string;
  patch_level: number;
  infected: boolean;
  isolated: boolean;
  compromised: boolean;
  criticality_score: number;
  credential_risk: number;
  detection_level: number;
  backup_health: number;
  uptime_status: boolean;
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

export type StepResult = {
  episode_id: string;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  action_name: string;
  network_risk: number;
  risk_breakdown: Partial<RiskBreakdown>;
  assets: AssetState[];
  red_log: Record<string, unknown>;
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
  termination_reason: string;
};
