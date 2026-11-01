import { useEffect, useState } from 'react'
import { APPROACHES, APPROACH_COLOR, APPROACH_LABEL, CONTROLLER_LABEL, api, fmt, type ControllerInfo, type Health, type ScenarioInfo, type Snapshot } from '../api'
import { ApproachLines } from '../charts'
import { LiveGate } from '../LiveGate'
import { useTelemetry } from '../telemetry'
import { Badge, Banner, Button, Card, Field, SplitBadge, Stat, inputCls, useResource, type Tone } from '../ui'

const STATUS_TONE: Record<string, Tone> = {
  idle: 'neutral', starting: 'info', running: 'good', stopping: 'info', finished: 'good', stopped: 'neutral', error: 'crit',
}

export function SimulationControl() {
  const t = useTelemetry()
  const scenarios = useResource<ScenarioInfo[]>('/api/scenarios')
  const controllers = useResource<ControllerInfo[]>('/api/controllers')
  const [controller, setController] = useState('camp_rule')
  const [scenario, setScenario] = useState('balanced')
  const [seed, setSeed] = useState(1)
  const [speed, setSpeed] = useState(10)
  const [cameras, setCameras] = useState(false)
  const vision = useResource<{ cameras: { kind: string }[] }>('/api/vision/status')
  const realStreams = Boolean(vision.data?.cameras.some((c) => c.kind !== 'synthetic'))
  const camUnavailable = vision.error ? `Camera config unavailable: ${vision.error}` : null
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const running = ['starting', 'running', 'stopping'].includes(t.status?.status ?? '')

  useEffect(() => {
    if (scenarios.data && !scenarios.data.some((s) => s.name === scenario)) setScenario(scenarios.data[0]?.name ?? '')
  }, [scenarios.data, scenario])

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setErr(null)
    try { await fn(); t.refresh() } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  const sel = scenarios.data?.find((s) => s.name === scenario)

  return (
    <Card title="Simulation control" subtitle="Runs SUMO on this machine. Simulation only: never connected to real signals.">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Field label="Controller">
          <select className={inputCls} value={controller} disabled={running} onChange={(e) => setController(e.target.value)}>
            {(controllers.data ?? []).map((c) => (
              <option key={c.name} value={c.name} disabled={!c.available} title={c.reason ?? ''}>
                {CONTROLLER_LABEL[c.name]}{c.available ? '' : ' (unavailable)'}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Scenario">
          <select className={inputCls} value={scenario} disabled={running} onChange={(e) => setScenario(e.target.value)}>
            {(scenarios.data ?? []).map((s) => <option key={s.name} value={s.name}>{s.name} ({s.split})</option>)}
          </select>
        </Field>
        <Field label="Seed">
          <input className={inputCls} type="number" min={0} max={1000000} value={seed} disabled={running}
            onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))} />
        </Field>
        <Field label="Speed (sim s per real s)" hint="0 = as fast as possible">
          <input className={inputCls} type="number" min={0} max={200} value={cameras && realStreams ? 1 : speed} disabled={running || (cameras && realStreams)}
            onChange={(e) => setSpeed(Math.min(200, Math.max(0, Number(e.target.value) || 0)))} />
        </Field>
        <div className="flex items-end gap-2">
          <Button disabled={busy || running || !scenario}
            onClick={() => act(() => api.post('/api/simulation/start', { controller, scenario, seed, speed: cameras && realStreams ? 1 : speed, camera_demand: cameras }))}>Start</Button>
          <Button variant="danger" disabled={busy || !running} onClick={() => act(() => api.post('/api/simulation/stop'))}>Stop</Button>
        </div>
      </div>
      <label className={`mt-3 flex items-start gap-2 text-sm ${camUnavailable ? 'text-ink-3' : 'text-ink-2'}`} title={camUnavailable ?? ''}>
        <input type="checkbox" className="mt-0.5 accent-[#06B6D4]" checked={cameras} disabled={running || Boolean(camUnavailable)}
          onChange={(e) => setCameras(e.target.checked)} />
        <span>
          Camera-driven demand (Phase 3): vehicles enter SUMO from the cameras in <code className="font-mono">configs/vision.yaml</code> instead of the scenario's flows; the scenario still sets turning shares.
          {realStreams ? ' Real streams are configured, so the run is locked to real time (speed 1).' : ' All cameras are synthetic (repeatable demo with scripted faults).'}
          {camUnavailable && <span className="block text-warn">▲ {camUnavailable}</span>}
        </span>
      </label>
      {sel && <p className="mt-3 text-xs text-ink-3"><SplitBadge split={sel.split} /> <span className="ml-2">{sel.description}</span></p>}
      {err && <div className="mt-3"><Banner tone="crit" title="Request failed">{err}</Banner></div>}
    </Card>
  )
}

function Warnings({ snap }: { snap: Snapshot }) {
  const items: { tone: Tone; text: string }[] = []
  if (snap.kpi.spillback) items.push({ tone: 'warn', text: 'Approach queue reaches the upstream end (spillback).' })
  if (snap.kpi.downstream_blocking) items.push({ tone: 'warn', text: 'A downstream exit is more than half full (blocking risk).' })
  if (snap.observed.stale.length) items.push({ tone: 'warn', text: `Sensor dropout: controller sees stale data for ${snap.observed.stale.join(', ')}.` })
  if (snap.kpi.teleports) items.push({ tone: 'crit', text: `${snap.kpi.teleports} vehicle teleport(s): SUMO removed jammed vehicles.` })
  if (snap.kpi.interventions_total) items.push({ tone: 'info', text: `Safety filter has overridden ${snap.kpi.interventions_total} controller request(s).` })
  snap.events.forEach((e) => items.push({ tone: 'info', text: `Event: ${e}` }))
  return (
    <Card title="Warnings" subtitle="Current step">
      {items.length === 0 ? <p className="text-sm text-ink-3">No active warnings.</p> : (
        <ul className="space-y-2">{items.map((w, i) => <li key={i}><Badge tone={w.tone}>{w.tone === 'crit' ? 'Critical' : w.tone === 'warn' ? 'Warning' : 'Info'}</Badge> <span className="ml-2 text-sm text-ink-2">{w.text}</span></li>)}</ul>
      )}
    </Card>
  )
}

function LiveOverview({ snap }: { snap: Snapshot }) {
  const t = useTelemetry()
  const metrics = t.status?.status === 'finished' ? t.status.metrics as Record<string, number | null> | null : null
  const strategy = snap.decision.strategy as string | undefined
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        <Stat label="Simulation time" value={`${snap.time_s.toFixed(0)}`} unit="s" hint={`step ${snap.step} / ${snap.total_steps}`} />
        <Stat label="Current phase" value={snap.signal.mode === 'GREEN' ? snap.signal.green_phase_name.replace('_GREEN', '') : snap.signal.mode.replace('_', '-')}
          hint={snap.signal.mode === 'GREEN' ? `green for ${snap.signal.time_in_mode_s.toFixed(0)} s` : 'clearance interval'} />
        <Stat label="Total queue" value={snap.kpi.total_queue_veh.toFixed(0)} unit="veh" hint="halting vehicles, 4 approaches" />
        <Stat label="Mean wait (on approaches)" value={fmt(snap.kpi.mean_wait_s)} unit="s" />
        <Stat label="Trips completed" value={snap.kpi.arrived_total} hint="since start (incl. warm-up)" />
        <Stat label="Phase switches" value={snap.kpi.phase_switches} hint={`${snap.kpi.vehicles_in_network} vehicles in network`} />
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {APPROACHES.map((a) => {
          const s = snap.truth.approaches[a]
          const c = snap.camp?.approaches[a]
          const green = snap.signal.mode === 'GREEN' && (snap.signal.green_phase === 0 ? 'NS' : 'EW').includes(a)
          return (
            <Card key={a} title={<span className="flex items-center gap-2"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: APPROACH_COLOR[a] }} aria-hidden />{APPROACH_LABEL[a]} approach</span>}
              actions={green ? <Badge tone="good">Green</Badge> : <Badge tone="neutral">Red</Badge>}>
              <dl className="grid grid-cols-2 gap-y-1 text-sm tabular">
                <dt className="text-ink-3">Queue</dt><dd className="text-right">{s.queue_veh.toFixed(0)} veh</dd>
                <dt className="text-ink-3">Longest wait</dt><dd className="text-right">{s.max_wait_s.toFixed(0)} s</dd>
                <dt className="text-ink-3">Arrivals</dt><dd className="text-right">{s.arrival_veh_per_h.toFixed(0)} veh/h</dd>
                <dt className="text-ink-3">Occupancy</dt><dd className="text-right">{s.occupancy_pct.toFixed(0)} %</dd>
                <dt className="text-ink-3">Trend</dt><dd className="text-right">{c?.trend ?? '—'}</dd>
              </dl>
            </Card>
          )
        })}
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Queue per approach" subtitle="Halting vehicles (speed < 0.1 m/s)" className="xl:col-span-2">
          {t.history.length > 1 ? <ApproachLines data={t.history} prefix="q_" unit="veh" /> : <p className="text-sm text-ink-3">Collecting data…</p>}
        </Card>
        <Warnings snap={snap} />
      </div>
      {metrics && (
        <Card title="Final episode metrics" subtitle="From SUMO trip records, after warm-up. Delay covers completed trips only.">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Stat label="Mean delay" value={fmt(metrics.delay_mean_s as number)} unit="s" />
            <Stat label="p95 delay" value={fmt(metrics.delay_p95_s as number)} unit="s" />
            <Stat label="Throughput" value={String(metrics.throughput_completed)} unit="trips" />
            <Stat label="Unfinished" value={String(metrics.unfinished)} unit="trips" />
            <Stat label="Max wait" value={fmt(metrics.max_wait_s as number)} unit="s" />
          </div>
        </Card>
      )}
      {strategy && <p className="text-xs text-ink-3">Active CAMP strategy: <span className="text-ink-2">{strategy}</span>. See CAMP Explainability for the reasoning.</p>}
    </div>
  )
}

export default function Overview() {
  const t = useTelemetry()
  const health = useResource<Health>('/health', 5000)
  const st = t.status
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={t.connected ? 'good' : 'crit'}>Backend {t.connected ? 'connected' : 'disconnected'}</Badge>
        <Badge tone={health.data?.sumo.available ? 'good' : health.error ? 'crit' : 'neutral'}>
          SUMO {health.data?.sumo.available ? 'available' : health.error ? 'unknown' : '…'}
        </Badge>
        <Badge tone={STATUS_TONE[st?.status ?? 'idle']}>Simulation {st?.status ?? 'idle'}</Badge>
        {t.stale && <Badge tone="warn">Stale data</Badge>}
        {st?.params && <span className="text-xs text-ink-3">{CONTROLLER_LABEL[st.params.controller]} · {st.params.scenario} · seed {st.params.seed}</span>}
      </div>
      <SimulationControl />
      <LiveGate>{(snap) => <LiveOverview snap={snap} />}</LiveGate>
    </div>
  )
}
