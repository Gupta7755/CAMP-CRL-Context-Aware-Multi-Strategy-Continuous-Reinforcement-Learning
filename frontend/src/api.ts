// Typed client for the CAMP-CRL backend. Mirrors camp_crl/api/app.py.

export const APPROACHES = ['N', 'E', 'S', 'W'] as const
export type Approach = (typeof APPROACHES)[number]

export const CONTROLLERS = ['fixed_time', 'fixed_time_tuned', 'actuated', 'max_pressure', 'camp_rule'] as const
export type ControllerName = (typeof CONTROLLERS)[number]

export const CONTROLLER_LABEL: Record<ControllerName, string> = {
  fixed_time: 'Fixed-Time (30/30)',
  fixed_time_tuned: 'Fixed-Time (tuned)',
  actuated: 'Actuated',
  max_pressure: 'Max-Pressure',
  camp_rule: 'CAMP rule',
}

// Colour follows the entity, fixed order (validated palette, see index.css).
export const CONTROLLER_COLOR: Record<ControllerName, string> = {
  camp_rule: '#3987E5',
  max_pressure: '#D95926',
  actuated: '#199E70',
  fixed_time_tuned: '#C98500',
  fixed_time: '#D55181',
}
const RL_LABEL: Record<string, string> = {
  ddpg_raw: 'DDPG (raw state)',
  td3_raw: 'TD3 (raw state)',
  td3_camp: 'TD3 (CAMP state)',
  sac_camp: 'SAC (CAMP state)',
  camp_crl_camp: 'CAMP prior only (α = 0)',
  camp_crl_fixed: 'CAMP-CRL (fixed α = 0.5)',
  camp_crl_gated: 'CAMP-CRL (gated α)',
}
/** Display name for any controller in results, including Phase-2 RL controllers. */
export const ctrlLabel = (c: string): string => (CONTROLLER_LABEL as Record<string, string>)[c] ?? RL_LABEL[c] ?? c

export const APPROACH_COLOR: Record<Approach, string> = {
  N: '#3987E5', E: '#D95926', S: '#199E70', W: '#C98500',
}
export const APPROACH_LABEL: Record<Approach, string> = { N: 'North', E: 'East', S: 'South', W: 'West' }

export interface ApproachState {
  vehicle_count: number
  queue_veh: number
  arrival_veh_per_h: number
  mean_speed_mps: number
  occupancy_pct: number
  max_wait_s: number
  mean_wait_s: number
  spillback_ratio: number
  downstream_occupancy_pct: number
  downstream_count: number
  stopline_count: number
  arrivals_step: number
  downstream_fill_ratio: number
  downstream_congestion: number
}

export interface TrafficState {
  time_s: number
  source: string
  stale: string[]
  approaches: Record<Approach, ApproachState>
}

export interface ApproachContext {
  pressure: number
  momentum: number
  risk: number
  fairness: number
  composite: number
  hats: Record<string, number>
  downstream_occupancy_pct: number
  downstream_fill_ratio: number
  downstream_congestion: number
  trend: 'RISING' | 'FALLING' | 'STABLE'
  anomaly_score: number
  anomaly: boolean
  stale: boolean
}

export interface CampContext {
  time_s: number
  pattern: string
  max_wait_s: number
  wait_dispersion_s: number
  confidence: number
  approaches: Record<Approach, ApproachContext>
}

/** Phase 3: one row per approach and step when the run uses camera-driven demand. */
export interface PerceptionRow {
  approach: Approach; mode: 'live' | 'fallback'; inserted: number; fallback_rate_veh_h: number
  status?: 'ok' | 'degraded' | 'stale' | 'disconnected'; confidence?: number; fps?: number; reason?: string
  vehicle_count?: number; queue_veh?: number; arrivals_total?: number; frame_age_s?: number
}

export interface Snapshot {
  time_s: number
  step: number
  total_steps: number
  controller: ControllerName
  scenario: string
  seed: number
  signal: { mode: 'GREEN' | 'YELLOW' | 'ALL_RED'; green_phase: number; green_phase_name: string; target_phase: number; state: string; time_in_mode_s: number }
  requested_phase: number
  decision: Record<string, unknown>
  interventions: { kind: string; detail: string }[]
  events: string[]
  truth: TrafficState
  observed: TrafficState
  camp: CampContext | null
  perception?: Record<Approach, PerceptionRow> | null
  kpi: {
    total_queue_veh: number
    arrived_total: number
    teleports: number
    phase_switches: number
    interventions_total: number
    vehicles_in_network: number
    mean_wait_s: number
    spillback: boolean
    downstream_blocking: boolean
  }
}

export type HistoryPoint = { t: number; phase: number; mode: string; total_queue: number; arrived: number; mean_wait: number } & Record<string, number | string>

export type SimStatusName = 'idle' | 'starting' | 'running' | 'stopping' | 'finished' | 'stopped' | 'error'

export interface SimStatus {
  status: SimStatusName
  params: { controller: ControllerName; scenario: string; seed: number; speed: number; camera_demand?: boolean; camera_clock?: 'sim' | 'wall' | null } | null
  error: string | null
  started_at: string | null
  step: number
  total_steps: number | null
  sim_time_s: number | null
  last_update_age_s: number | null
  step_latency_ms: number | null
  run_dir: string | null
  metrics: Record<string, unknown> | null
  seq: number
  stale?: boolean
}

export interface Health {
  status: string
  version: string
  uptime_s: number
  sumo: { available: boolean; version: string | null; error?: string }
  live_status: string
  camera: { status: string; reason: string; demo_runs?: number; latest_demo?: string | null }
}

export interface ScenarioInfo {
  name: string
  split: 'train' | 'validation' | 'test'
  description: string
  demand_veh_per_h: Record<Approach, number>
  has_profile: boolean
  bottlenecks: { edge: string; speed_mps: number; start_s: number; end_s: number }[]
  sensor: { noise_std_frac: number; dropout_prob: number }
}

export interface ControllerInfo { name: ControllerName; available: boolean; reason: string | null; green_s?: number[] }

export interface ExperimentManifest {
  id: string
  created: string
  status: 'queued' | 'running' | 'completed' | 'completed_with_failures' | 'cancelled' | 'failed'
  params: { controllers: ControllerName[]; scenarios: string[]; seeds: number[]; baseline: ControllerName; kind: string; workers: number }
  total_runs: number
  completed_runs: number
  failed_runs: number
  error: string | null
  finished: string | null
  cancellable?: boolean
  note?: string
}

export interface Described { mean: number | null; std: number | null; n: number; ci95: [number, number] | null }
export interface Paired { mean_diff: number | null; ci95: [number, number] | null; ci_excludes_zero: boolean; improvement_pct_of_means: number | null; lower_is_better: boolean; n_pairs: number }
export interface Summary {
  baseline: ControllerName
  controllers: ControllerName[]
  scenarios: Record<string, { split: string; controllers: Record<string, Record<string, Described>>; paired_vs_baseline: Record<string, Record<string, Paired>> }>
  by_split: Record<string, Record<string, Record<string, number | null>> & { _scenarios: string[] }>
  metrics: { name: string; lower_is_better: boolean }[]
  notes: string[]
  fusion_alpha?: Record<string, Record<string, number>>
  training_seed_spread?: Record<string, { delay_mean_s_by_training_seed: Record<string, number>; std_across_training_seeds: number | null }>
  rl_runs?: string[]
}

export interface LogRecord { seq: number; time: string; level: string; logger: string; message: string }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init })
  } catch {
    throw new ApiError(0, 'Backend unreachable. Is the server running on port 8000?')
  }
  const text = await res.text()
  let body: unknown = null
  try { body = text ? JSON.parse(text) : null } catch { body = text }
  if (!res.ok) {
    const detail = (body as { detail?: unknown })?.detail
    const msg = typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.map((d: { msg?: string; loc?: unknown[] }) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`).join('; ')
      : `HTTP ${res.status}`
    throw new ApiError(res.status, msg)
  }
  return body as T
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) => request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) => request<T>(path, { method: 'PUT', body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
}

export const fmt = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(digits)
