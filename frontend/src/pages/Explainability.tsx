import { APPROACHES, APPROACH_LABEL, fmt, type Snapshot } from '../api'
import { ApproachLines } from '../charts'
import { LiveGate } from '../LiveGate'
import { useTelemetry } from '../telemetry'
import { Badge, Banner, Card, Stat, Table, useResource } from '../ui'

interface ContextResponse {
  weights: { pressure: Record<string, number>; momentum: Record<string, number>; lambda_momentum: number; lambda_risk: number; lambda_fairness: number }
}

function Bar01({ v, signed = false }: { v: number; signed?: boolean }) {
  const pct = Math.min(Math.abs(v), 1) * 100
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-1.5 w-16 rounded bg-bg-2" aria-hidden>
        {signed ? (
          <div className="absolute top-0 h-1.5 rounded bg-accent" style={{ left: v >= 0 ? '50%' : `${50 - pct / 2}%`, width: `${pct / 2}%` }} />
        ) : <div className="h-1.5 rounded bg-accent" style={{ width: `${pct}%` }} />}
      </div>
      <span className="tabular">{v.toFixed(2)}</span>
    </div>
  )
}

function Body({ snap }: { snap: Snapshot }) {
  const t = useTelemetry()
  const ctx = useResource<ContextResponse>('/api/camp/context')
  const camp = snap.camp
  if (!camp) return <Banner tone="info" title="No CAMP context yet">The first step has not been evaluated.</Banner>
  const usedForControl = snap.controller === 'camp_rule'
  const w = ctx.data?.weights
  return (
    <div className="space-y-4">
      {!usedForControl && (
        <Banner tone="info" title="Observational context">
          The active controller is <b>{snap.controller}</b>. CAMP context is computed alongside for explanation only and does not influence its decisions.
        </Banner>
      )}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Traffic pattern" value={<span className="text-lg">{camp.pattern.replace('_', ' ')}</span>} />
        <Stat label="Strategy" value={<span className="text-lg">{(snap.decision.strategy as string) ?? '—'}</span>} hint={usedForControl ? 'with hysteresis' : 'n/a for this controller'} />
        <Stat label="Data confidence score" value={camp.confidence.toFixed(2)} hint="share of fresh observations; not a probability" />
        <Stat label="Longest wait / dispersion" value={`${camp.max_wait_s.toFixed(0)} s`} hint={`per-approach mean-wait std ${camp.wait_dispersion_s.toFixed(1)} s`} />
      </div>
      {usedForControl && (
        <Card title="Why this action">
          <p className="text-sm text-ink-2"><span className="text-ink-3">Strategy reason:</span> {String(snap.decision.strategy_reason ?? '—')}</p>
          <p className="mt-1 text-sm text-ink-2"><span className="text-ink-3">Phase decision:</span> {String(snap.decision.decision_reason ?? '—')}</p>
          <p className="mt-1 text-xs text-ink-3">Axis urgency = max composite C over its two approaches; switch when the other axis exceeds the current by the strategy's margin.</p>
        </Card>
      )}
      <Card title="Per-approach CAMP terms" subtitle="P, S, F in [0, 1]; M in [−1, 1]; C is a ranking score (not a probability)">
        <Table head={['Approach', 'Pressure P', 'Momentum M', 'Risk F (heuristic)', 'Fairness S', 'Composite C', 'Trend', 'Anomaly (CUSUM)', 'Data']}>
          {APPROACHES.map((a) => {
            const c = camp.approaches[a]
            return (
              <tr key={a} className="border-b border-line/50">
                <td className="px-2 py-1.5">{APPROACH_LABEL[a]}</td>
                <td className="px-2 py-1.5"><Bar01 v={c.pressure} /></td>
                <td className="px-2 py-1.5"><Bar01 v={c.momentum} signed /></td>
                <td className="px-2 py-1.5"><Bar01 v={c.risk} /></td>
                <td className="px-2 py-1.5"><Bar01 v={c.fairness} /></td>
                <td className="px-2 py-1.5 font-semibold">{c.composite.toFixed(2)}</td>
                <td className="px-2 py-1.5">{c.trend === 'RISING' ? '↑ rising' : c.trend === 'FALLING' ? '↓ falling' : '→ stable'}</td>
                <td className="px-2 py-1.5">{c.anomaly ? <Badge tone="warn">Surge {fmt(c.anomaly_score, 2)}</Badge> : <span className="tabular text-ink-3">{c.anomaly_score.toFixed(2)}</span>}</td>
                <td className="px-2 py-1.5">{c.stale ? <Badge tone="warn">Stale</Badge> : <Badge tone="good">Fresh</Badge>}</td>
              </tr>
            )
          })}
        </Table>
      </Card>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Composite urgency C over time">
          {t.history.length > 1 ? <ApproachLines data={t.history} prefix="c_" unit="" /> : <p className="text-sm text-ink-3">Collecting data…</p>}
        </Card>
        <Card title="Weights in use" subtitle="Starting defaults, not tuned values">
          {w ? (
            <div className="space-y-2 text-sm">
              <p className="text-ink-2">P = {Object.entries(w.pressure).map(([k, v]) => `${v.toFixed(2)}·${k}`).join(' + ')}</p>
              <p className="text-ink-2">M = {w.momentum.beta_queue}·queue growth + {w.momentum.beta_arrival}·arrival growth − {w.momentum.beta_speed}·speed change</p>
              <p className="text-ink-2">C = P + {w.lambda_momentum}·M + {w.lambda_risk}·F + {w.lambda_fairness}·S</p>
              <p className="text-xs text-ink-3">Trend, pattern, anomaly and confidence are reported for explanation; Phase-1 control decisions do not use them.</p>
            </div>
          ) : <p className="text-sm text-ink-3">Loading…</p>}
        </Card>
      </div>
    </div>
  )
}

export default function Explainability() {
  return <LiveGate>{(snap) => <Body snap={snap} />}</LiveGate>
}
