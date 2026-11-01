import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LiveGate } from '../LiveGate'
import Evaluation from '../pages/Evaluation'
import ExperimentLab, { parseSeeds } from '../pages/ExperimentLab'
import CameraPanel, { LiveCameras } from '../Cameras'
import CameraSetup from '../pages/CameraSetup'
import LiveAdvisory from '../pages/LiveAdvisory'
import { TelemetryContext, type Telemetry } from '../telemetry'
import type { Snapshot } from '../api'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

function withTelemetry(t: Partial<Telemetry>, ui: ReactNode) {
  const value: Telemetry = { connected: true, status: null, snapshot: null, history: [], lastMessageAt: Date.now(), stale: false, refresh: () => {}, ...t }
  return render(<MemoryRouter><TelemetryContext.Provider value={value}>{ui}</TelemetryContext.Provider></MemoryRouter>)
}

function mockFetch(routes: Record<string, { status: number; body: unknown }>) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const key = Object.keys(routes).find((k) => url.startsWith(k))
    const r = key ? routes[key] : { status: 404, body: { detail: 'not found' } }
    return new Response(JSON.stringify(r.body), { status: r.status })
  }))
}

const snap = { time_s: 5, step: 5, total_steps: 100 } as unknown as Snapshot

describe('LiveGate states', () => {
  it('shows connecting before the first message', () => {
    withTelemetry({ connected: false, lastMessageAt: null }, <LiveGate>{() => <p>child</p>}</LiveGate>)
    expect(screen.getByText(/Connecting to backend/)).toBeInTheDocument()
  })
  it('shows empty state when connected without data', () => {
    withTelemetry({}, <LiveGate>{() => <p>child</p>}</LiveGate>)
    expect(screen.getByText('No live simulation data')).toBeInTheDocument()
  })
  it('shows disconnected banner but keeps last data', () => {
    withTelemetry({ connected: false, snapshot: snap }, <LiveGate>{() => <p>child</p>}</LiveGate>)
    expect(screen.getByText('Disconnected from backend')).toBeInTheDocument()
    expect(screen.getByText('child')).toBeInTheDocument()
  })
  it('shows stale banner', () => {
    withTelemetry({ stale: true, snapshot: snap }, <LiveGate>{() => <p>child</p>}</LiveGate>)
    expect(screen.getByText('Stale data')).toBeInTheDocument()
  })
  it('shows simulation error', () => {
    withTelemetry({ status: { status: 'error', error: 'SUMO crashed' } as Telemetry['status'] }, <LiveGate>{() => <p>child</p>}</LiveGate>)
    expect(screen.getByText('SUMO crashed')).toBeInTheDocument()
  })
})

describe('Evaluation page states', () => {
  it('empty state when no evaluation exists (404)', async () => {
    mockFetch({ '/api/evaluation/latest': { status: 404, body: { detail: 'none' } } })
    withTelemetry({}, <Evaluation />)
    expect(await screen.findByText('No evaluation has been run yet')).toBeInTheDocument()
  })
  it('error state on server error', async () => {
    mockFetch({ '/api/evaluation/latest': { status: 500, body: { detail: 'boom' } } })
    withTelemetry({}, <Evaluation />)
    expect(await screen.findByText('Could not load data')).toBeInTheDocument()
    expect(screen.getByText(/boom/)).toBeInTheDocument()
  })
  it('error state when backend unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('network') }))
    withTelemetry({}, <Evaluation />)
    expect(await screen.findByText(/Backend unreachable/)).toBeInTheDocument()
  })
  it('renders summary tables', async () => {
    const d = { mean: 12.5, std: 1, n: 3, ci95: [11, 14] }
    const metrics = { delay_mean_s: d, delay_p95_s: d, queue_mean_veh: d }
    const summary = {
      baseline: 'fixed_time', controllers: ['fixed_time', 'camp_rule'],
      metrics: [{ name: 'delay_mean_s', lower_is_better: true }],
      scenarios: { balanced: { split: 'train', controllers: { fixed_time: metrics, camp_rule: metrics },
        paired_vs_baseline: { camp_rule: { delay_mean_s: { mean_diff: -1, ci95: [-2, -0.5], ci_excludes_zero: true, improvement_pct_of_means: 8, lower_is_better: true, n_pairs: 3 } } } } },
      by_split: { train: { fixed_time: { delay_mean_s: 13 }, camp_rule: { delay_mean_s: 12 }, _scenarios: ['balanced'] } },
      notes: ['note one'],
    }
    mockFetch({ '/api/evaluation/latest': { status: 200, body: { manifest: { id: 'x1', params: { seeds: [1, 2, 3] }, finished: '2026-10-25T21:00:00' }, summary } } })
    withTelemetry({}, <Evaluation />)
    expect(await screen.findByText('Seen: train')).toBeInTheDocument()
    expect(screen.getByText(/Unseen scenarios not evaluated/)).toBeInTheDocument()
    expect(screen.getByText('note one')).toBeInTheDocument()
    expect(screen.getAllByText(/\+8\.0 % better/).length).toBeGreaterThan(0)
  })
})

describe('Experiment Lab', () => {
  it('parses seeds', () => {
    expect(parseSeeds('1-3, 7')).toEqual([1, 2, 3, 7])
    expect(parseSeeds('3-1')).toBeNull()
    expect(parseSeeds('a')).toBeNull()
    expect(parseSeeds('1-40')).toBeNull()
    expect(parseSeeds('')).toBeNull()
  })
  it('shows empty experiment list and disabled tuned controller with reason', async () => {
    mockFetch({
      '/api/experiments': { status: 200, body: [] },
      '/api/scenarios': { status: 200, body: [{ name: 'balanced', split: 'train', description: '' }, { name: 'held', split: 'test', description: '' }] },
      '/api/controllers': { status: 200, body: [{ name: 'fixed_time', available: true, reason: null }, { name: 'fixed_time_tuned', available: false, reason: 'No tuned plan yet' }] },
      '/api/logs/recent': { status: 200, body: [] },
      '/api/rl/runs': { status: 200, body: [] },
    })
    withTelemetry({}, <ExperimentLab />)
    expect(await screen.findByText('No experiments yet')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(/No tuned plan yet/)).toBeInTheDocument())
    const held = screen.getByText('held').closest('label')!.querySelector('input')!
    expect(held).toBeDisabled()
    expect(await screen.findByText(/No RL training runs yet/)).toBeInTheDocument()
  })
})

describe('Camera panel states', () => {
  const base = {
    cameras: [{ approach: 'N', source: 'rtsp://***@10.0.0.5/live', kind: 'stream', faults: [] }],
    detector: { kind: 'yolox', model: 'models/yolox_nano.onnx', model_present: true, license: 'YOLOX (Megvii), Apache-2.0' },
    health: { stale_after_s: 2, min_fps: 6, target_fps: 10, min_contrast: 6 },
    bridge: { min_confidence: 0.5, ewma_tau_s: 120, min_trusted_s: 60 },
    fps: 10, packages: { cv2: '4.14.0', onnxruntime: '1.31.0' }, latest_demo: null, demo_runs: 0,
    how_to_run: 'python -m camp_crl.vision.demo', note: 'Camera data only drives the SUMO simulation.',
  }
  it('error state when the vision config is invalid', async () => {
    mockFetch({ '/api/vision/status': { status: 422, body: { detail: 'configs/vision.yaml: bad roi' } } })
    withTelemetry({}, <CameraPanel />)
    expect(await screen.findByText('configs/vision.yaml: bad roi')).toBeInTheDocument()
  })
  it('empty state before any demo, with redacted source and detector info', async () => {
    mockFetch({ '/api/vision/status': { status: 200, body: base } })
    withTelemetry({}, <CameraPanel />)
    expect(await screen.findByText('No camera demo has been run yet')).toBeInTheDocument()
    expect(screen.getByText('rtsp://***@10.0.0.5/live')).toBeInTheDocument()
    expect(screen.getByText(/Detector: yolox/)).toBeInTheDocument()
  })
  it('shows demo results, synthetic-detector warning and status timeline', async () => {
    const appr = { camera_source: 'synthetic:N', true_arrivals: 100, perceived_arrivals: 98, count_error_pct: -2, live_count_error_pct: 0.5,
      inserted_into_sumo: 101, fallback_s: 120, reconnects: 1, statuses_s: { ok: 1680, degraded: 0, stale: 0, disconnected: 120 } }
    const latest = { id: '20261026-000000-000000_actuated_camera_demo_s1', controller: 'actuated', seed: 1, faults: true, duration_s: 1800,
      detector: 'synthetic', note: 'synthetic cameras and detector', approaches: { N: appr, E: appr, S: appr, W: appr },
      transitions: [], wall_time_s: 60, delay_mean_s: 14.4 }
    mockFetch({
      '/api/vision/status': { status: 200, body: { ...base, latest_demo: latest, demo_runs: 1 } },
      '/api/vision/demos/': { status: 200, body: { run: latest.id, every_s: 10, overlays: ['N_00600.jpg'],
        points: [{ t: 0, approach: 'N', status: 'ok', mode: 'live', confidence: 0.9, fps: 10, vehicle_count: 2, queue_veh: 1 },
                 { t: 10, approach: 'N', status: 'disconnected', mode: 'fallback', confidence: 0, fps: 0, vehicle_count: 0, queue_veh: 0 }],
        events: [{ time_s: '600', approach: 'N', event: 'disconnected (stream dropped)' }] } },
    })
    withTelemetry({}, <CameraPanel />)
    expect(await screen.findByText(/synthetic cameras and detector/)).toBeInTheDocument()
    expect(screen.getAllByText('+0.5 %')).toHaveLength(4)
    expect(await screen.findByText(/stream dropped/)).toBeInTheDocument()
    expect(screen.getByTitle(/t=10s · Disconnected · fallback/)).toBeInTheDocument()
    expect(screen.getByAltText(/North camera at t=600s/)).toBeInTheDocument()
  })
})

describe('Live camera card', () => {
  it('renders per-approach camera status and fallback reason from the snapshot', () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })))
    const row = (a: string, extra: object) => ({ approach: a, mode: 'live', inserted: 0, fallback_rate_veh_h: 400, status: 'ok', confidence: 0.92, fps: 10, vehicle_count: 3, queue_veh: 1, ...extra })
    const s = { ...snap, perception: { N: row('N', {}), E: row('E', { status: 'disconnected', mode: 'fallback', confidence: 0, reason: 'stream dropped' }), S: row('S', {}), W: row('W', {}) } } as unknown as Snapshot
    withTelemetry({}, <LiveCameras snap={s} clock="wall" />)
    expect(screen.getByText(/wall \(real streams, real time\)/)).toBeInTheDocument()
    expect(screen.getAllByText('Camera demand')).toHaveLength(3)
    expect(screen.getByText('Fallback demand')).toBeInTheDocument()
    expect(screen.getByText(/stream dropped/)).toBeInTheDocument()
    expect(screen.getByText('400 veh/h')).toBeInTheDocument()
  })
})

describe('Cameras page', () => {
  const cam = (approach: string, extra: object = {}) => ({ approach, source: 'synthetic', kind: 'synthetic', origin: 'vision.yaml', preview: null, ...extra })
  const body = (extra: object = {}) => ({
    cameras: [cam('N', { source: 'http://192.168.0.110:8080/video', kind: 'stream', origin: 'dashboard' }), cam('E'), cam('S', { source: null, kind: 'none', origin: 'dashboard' }), cam('W')],
    detector: { kind: 'auto', model: 'models/yolox_nano.onnx', model_present: true }, locked: false, note: 'Camera data only drives the SUMO simulation.', ...extra,
  })
  it('lists one card per direction with source, kind and origin', async () => {
    mockFetch({ '/api/cameras': { status: 200, body: body() } })
    withTelemetry({}, <CameraSetup />)
    expect(await screen.findByText('Phone / IP stream')).toBeInTheDocument()
    // shown as the North card's current source (the help banner uses the same example address)
    expect(screen.getAllByText('http://192.168.0.110:8080/video').some((el) => el.classList.contains('font-mono') && el.tagName === 'SPAN')).toBe(true)
    expect(screen.getAllByText('Synthetic (demo)')).toHaveLength(2)
    expect(screen.getAllByRole('button', { name: 'No camera' })).toHaveLength(4)
    expect(screen.getAllByText('No camera')).toHaveLength(5)  // 4 buttons + South's badge
    // no camera on South: its live view cannot be started
    const startButtons = screen.getAllByRole('button', { name: 'Start live view' })
    expect(startButtons.filter((b) => (b as HTMLButtonElement).disabled)).toHaveLength(1)
  })
  it('saves a pasted phone URL with PUT', async () => {
    const calls: { url: string; init?: RequestInit }[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init })
      return new Response(JSON.stringify(body()), { status: 200 })
    }))
    withTelemetry({}, <CameraSetup />)
    const input = await screen.findByLabelText('Camera URL for East')
    const { fireEvent } = await import('@testing-library/react')
    fireEvent.change(input, { target: { value: 'http://192.168.0.111:8080/video' } })
    fireEvent.submit(input.closest('form')!)
    await waitFor(() => expect(calls.some((c) => c.url === '/api/cameras/E' && c.init?.method === 'PUT')).toBe(true))
    const put = calls.find((c) => c.init?.method === 'PUT')!
    expect(JSON.parse(String(put.init!.body))).toEqual({ source: 'http://192.168.0.111:8080/video' })
  })
  it('locks source changes during a camera-driven run', async () => {
    mockFetch({ '/api/cameras': { status: 200, body: body({ locked: true }) } })
    withTelemetry({}, <CameraSetup />)
    expect(await screen.findByText('Sources locked')).toBeInTheDocument()
    expect((screen.getByLabelText('Camera URL for East') as HTMLInputElement).disabled).toBe(true)
  })
})

describe('Live Advisory page', () => {
  const approach = (known: boolean, light: string, extra: object = {}) => ({ known, light, vehicles: known ? 0 : null, queue: known ? 0 : null, arrivals_veh_h: known ? 0 : null, longest_wait_s: known ? 0 : null, trend: known ? 'STABLE' : null, surge: known ? false : null, urgency: null, red_for_s: 4, ...extra })
  const advice = {
    advisory_only: true, mode: 'default_loop', mode_reason: 'no working camera on the East–West road, so its traffic is unknown', strategy: null, pattern: null,
    signal: { mode: 'GREEN', green_phase: 0, green_label: 'North–South', target_phase: 0, time_in_interval_s: 12 },
    plan: [{ label: 'North–South green', color: 'green', seconds: 18, max_seconds: 48 }, { label: 'North–South yellow', color: 'yellow', seconds: 3 }, { label: 'All red', color: 'red', seconds: 2 }, { label: 'East–West green', color: 'green', seconds: 30 }],
    approaches: { N: approach(true, 'green'), S: approach(true, 'green'), E: approach(false, 'red'), W: approach(false, 'red') },
    cameras_working: 2, default_green_s: 30, timing: { min_green_s: 10, max_green_s: 60, yellow_s: 3, all_red_s: 2, max_red_s: 120 },
  }
  it('shows the empty state before advice starts', async () => {
    mockFetch({ '/api/advisory': { status: 200, body: { running: false, error: null, advice: null, note: 'Advisory only.' } } })
    withTelemetry({}, <LiveAdvisory />)
    expect(await screen.findByText('Advice is not running')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start live advice' })).toBeInTheDocument()
  })
  it('shows recommendation, countdown, plan and unknown directions', async () => {
    mockFetch({ '/api/advisory': { status: 200, body: { running: true, error: null, advice, note: 'Advisory only.' } }, '/api/cameras': { status: 200, body: {} } })
    withTelemetry({}, <LiveAdvisory />)
    expect(await screen.findByText('18 s left')).toBeInTheDocument()
    expect(screen.getByText('Default 30 s loop')).toBeInTheDocument()
    expect(screen.getByText('2 of 4 cameras working')).toBeInTheDocument()
    expect(screen.getByText(/East–West green · 30 s/)).toBeInTheDocument()
    expect(screen.getAllByText('Unknown')).toHaveLength(2)
    expect(screen.getByText(/so its traffic is unknown/)).toBeInTheDocument()
  })
  it('shows holding with maximum instead of a fake countdown', async () => {
    const holding = { ...advice, mode: 'adaptive', plan: [{ label: 'North–South green', color: 'green', seconds: null, max_seconds: 34, note: 'holding while traffic keeps coming (re-decided every second)' }, ...advice.plan.slice(1)] }
    mockFetch({ '/api/advisory': { status: 200, body: { running: true, error: null, advice: holding, note: 'Advisory only.' } } })
    withTelemetry({}, <LiveAdvisory />)
    expect(await screen.findByText(/at most 34 s more/)).toBeInTheDocument()
    expect(screen.getByText('Adaptive (CAMP)')).toBeInTheDocument()
  })
})
