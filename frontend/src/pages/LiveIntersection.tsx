import { APPROACHES, APPROACH_LABEL, type Approach, type Snapshot } from '../api'
import { LiveCameras } from '../Cameras'
import { LiveGate } from '../LiveGate'
import { useTelemetry } from '../telemetry'
import { Badge, Card, Table } from '../ui'

const PHASE_NAME = ['North–South green', 'East–West green']
const SIGNAL_FILL = { green: '#0CA30C', yellow: '#FAB219', red: '#D03B3B' }
const SIGNAL_TEXT = { green: 'G', yellow: 'Y', red: 'R' }

function signalFor(snap: Snapshot, a: Approach): keyof typeof SIGNAL_FILL {
  const axis = snap.signal.green_phase === 0 ? 'NS' : 'EW'
  if (!axis.includes(a)) return 'red'
  return snap.signal.mode === 'GREEN' ? 'green' : snap.signal.mode === 'YELLOW' ? 'yellow' : 'red'
}

/** Top-down schematic. Queue bars grow upstream from each stop line, scaled to the 40-vehicle normalisation cap. */
function Schematic({ snap }: { snap: Snapshot }) {
  const S = 360, C = S / 2, ROAD = 56, H = ROAD / 2, ARM = C - H
  const geom: Record<Approach, { bar: (len: number) => [number, number, number, number]; light: [number, number]; label: [number, number] }> = {
    N: { bar: (l) => [C - H + 4, C - H - l, H - 8, l], light: [C - H / 2, C - H - 10], label: [C - H - 8, 24] },
    S: { bar: (l) => [C + 4, C + H, H - 8, l], light: [C + H / 2, C + H + 10], label: [C + H + 8, S - 16] },
    E: { bar: (l) => [C + H, C - H + 4, l, H - 8], light: [C + H + 10, C - H / 2], label: [S - 64, C - H - 10] },
    W: { bar: (l) => [C - H - l, C + 4, l, H - 8], light: [C - H - 10, C + H / 2], label: [12, C + H + 22] },
  }
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto w-full max-w-md" role="img"
      aria-label={`Intersection: ${PHASE_NAME[snap.signal.green_phase]}, mode ${snap.signal.mode}`}>
      <rect width={S} height={S} fill="#101827" rx={8} />
      <rect x={C - H} y={0} width={ROAD} height={S} fill="#2B3A55" />
      <rect x={0} y={C - H} width={S} height={ROAD} fill="#2B3A55" />
      <line x1={C} y1={0} x2={C} y2={C - H} stroke="#7F8DA6" strokeDasharray="6 6" />
      <line x1={C} y1={C + H} x2={C} y2={S} stroke="#7F8DA6" strokeDasharray="6 6" />
      <line x1={0} y1={C} x2={C - H} y2={C} stroke="#7F8DA6" strokeDasharray="6 6" />
      <line x1={C + H} y1={C} x2={S} y2={C} stroke="#7F8DA6" strokeDasharray="6 6" />
      {APPROACHES.map((a) => {
        const q = snap.truth.approaches[a].queue_veh
        const len = Math.min(q / 40, 1) * (ARM - 16)
        const [x, y, w, h] = geom[a].bar(len)
        const sig = signalFor(snap, a)
        const [lx, ly] = geom[a].light
        const [tx, ty] = geom[a].label
        return (
          <g key={a}>
            {len > 0 && <rect x={x} y={y} width={w} height={h} fill="#E6EDF7" opacity={0.75} rx={2} />}
            <circle cx={lx} cy={ly} r={8} fill={SIGNAL_FILL[sig]} stroke="#0B1220" strokeWidth={2} />
            <text x={lx} y={ly + 3.5} textAnchor="middle" fontSize={9} fontWeight={700} fill="#0B1220">{SIGNAL_TEXT[sig]}</text>
            <text x={tx} y={ty} fontSize={12} fill="#A9B6CC" textAnchor={a === 'N' ? 'end' : 'start'}>{a} · {q.toFixed(0)} veh</text>
          </g>
        )
      })}
    </svg>
  )
}

function Body({ snap }: { snap: Snapshot }) {
  const t = useTelemetry()
  const overridden = snap.interventions.length > 0
  const transitioning = snap.signal.mode !== 'GREEN'
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      <Card title="Intersection" subtitle="Queue bars scale to 40 vehicles; signal heads show G / Y / R" className="xl:col-span-3">
        <Schematic snap={snap} />
        <p className="mt-2 text-center text-xs text-ink-3">t = {snap.time_s.toFixed(0)} s · {snap.controller} · {snap.scenario}</p>
      </Card>
      <div className="space-y-4 xl:col-span-2">
        <Card title="Proposed vs applied action" subtitle="The controller proposes; the independent safety filter decides">
          <dl className="grid grid-cols-2 gap-y-2 text-sm">
            <dt className="text-ink-3">Controller requested</dt><dd>{PHASE_NAME[snap.requested_phase] ?? `invalid (${snap.requested_phase})`}</dd>
            <dt className="text-ink-3">Signal now</dt>
            <dd>{transitioning ? `${snap.signal.mode.replace('_', '-')} → ${PHASE_NAME[snap.signal.target_phase]}` : PHASE_NAME[snap.signal.green_phase]}</dd>
            <dt className="text-ink-3">Time in interval</dt><dd className="tabular">{snap.signal.time_in_mode_s.toFixed(0)} s</dd>
            <dt className="text-ink-3">Safety filter</dt>
            <dd>{overridden ? <Badge tone="warn">Overrode request</Badge> : transitioning ? <Badge tone="info">Clearance in progress</Badge> : <Badge tone="good">Passed through</Badge>}</dd>
          </dl>
          {overridden && <ul className="mt-3 space-y-1 text-xs text-ink-2">{snap.interventions.map((i, k) => <li key={k}><span className="font-mono">{i.kind}</span>: {i.detail}</li>)}</ul>}
          {typeof snap.decision.decision_reason === 'string' && <p className="mt-3 text-xs text-ink-3">Controller reason: {snap.decision.decision_reason}</p>}
          <p className="mt-3 break-all font-mono text-[11px] text-ink-3" title="SUMO link states (G/g green, y yellow, r red)">state {snap.signal.state}</p>
        </Card>
        <Card title="Approach queues">
          <Table head={['Approach', 'Queue', 'Vehicles', 'Near stop line', 'Spillback']}>
            {APPROACHES.map((a) => {
              const s = snap.truth.approaches[a]
              return (
                <tr key={a} className="border-b border-line/50">
                  <td className="px-2 py-1.5">{APPROACH_LABEL[a]}</td>
                  <td className="px-2 py-1.5">{s.queue_veh.toFixed(0)}</td>
                  <td className="px-2 py-1.5">{s.vehicle_count}</td>
                  <td className="px-2 py-1.5">{s.stopline_count}</td>
                  <td className="px-2 py-1.5">{(s.spillback_ratio * 100).toFixed(0)} %</td>
                </tr>
              )
            })}
          </Table>
        </Card>
      </div>
      {snap.perception && <div className="xl:col-span-5"><LiveCameras snap={snap} clock={t.status?.params?.camera_clock} /></div>}
    </div>
  )
}

export default function LiveIntersection() {
  return <LiveGate>{(snap) => <Body snap={snap} />}</LiveGate>
}
