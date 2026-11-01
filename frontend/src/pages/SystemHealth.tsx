import { useEffect, useState } from 'react'
import { api, fmt, type Health, type LogRecord } from '../api'
import { useTelemetry } from '../telemetry'
import CameraPanel from '../Cameras'
import { Badge, Card, Empty, ErrorState, Field, Loading, Stat, inputCls, useResource } from '../ui'

function useRoundTrip(ms = 3000): number | null {
  const [rtt, setRtt] = useState<number | null>(null)
  useEffect(() => {
    let alive = true
    const ping = () => {
      const t0 = performance.now()
      api.get('/health').then(() => alive && setRtt(performance.now() - t0)).catch(() => alive && setRtt(null))
    }
    ping()
    const id = setInterval(ping, ms)
    return () => { alive = false; clearInterval(id) }
  }, [ms])
  return rtt
}

export default function SystemHealth() {
  const t = useTelemetry()
  const health = useResource<Health>('/health', 5000)
  const [level, setLevel] = useState('INFO')
  const logs = useResource<LogRecord[]>(`/api/logs/recent?limit=200&level=${level}`, 3000)
  const rtt = useRoundTrip()
  const msgAge = t.lastMessageAt ? (Date.now() - t.lastMessageAt) / 1000 : null

  return (
    <div className="space-y-4">
      {health.loading ? <Loading /> : health.error ? <ErrorState error={health.error} onRetry={health.reload} /> : health.data && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Stat label="Backend" value={<Badge tone="good">OK</Badge>} hint={`v${health.data.version}, up ${Math.round(health.data.uptime_s)} s`} />
          <Stat label="SUMO" value={<Badge tone={health.data.sumo.available ? 'good' : 'crit'}>{health.data.sumo.available ? 'Available' : 'Missing'}</Badge>} hint={health.data.sumo.version ?? health.data.sumo.error} />
          <Stat label="API round trip" value={fmt(rtt, 0)} unit="ms" />
          <Stat label="Telemetry socket" value={<Badge tone={t.connected ? 'good' : 'crit'}>{t.connected ? 'Connected' : 'Reconnecting'}</Badge>} hint={msgAge !== null ? `last message ${msgAge.toFixed(1)} s ago` : 'no messages yet'} />
          <Stat label="Live simulation" value={<span className="text-lg">{t.status?.status ?? '—'}</span>} hint={t.stale ? 'stale' : undefined} />
          <Stat label="Step compute time" value={fmt(t.status?.step_latency_ms ?? null, 1)} unit="ms" hint="SUMO step + control, per step" />
        </div>
      )}
      <CameraPanel />
      <Card title="Logs" actions={
        <Field label="Minimum level">
          <select className={inputCls} value={level} onChange={(e) => setLevel(e.target.value)}>
            {['DEBUG', 'INFO', 'WARNING', 'ERROR'].map((l) => <option key={l}>{l}</option>)}
          </select>
        </Field>}>
        {logs.error ? <ErrorState error={logs.error} /> : !logs.data?.length ? <Empty title="No log entries at this level" /> : (
          <ol className="max-h-[28rem] space-y-0.5 overflow-y-auto font-mono text-[11px] text-ink-2">
            {[...logs.data].reverse().map((r) => (
              <li key={r.seq}>
                <span className="text-ink-3">{r.time.replace('T', ' ')}</span>{' '}
                <span className={r.level === 'ERROR' ? 'text-[#F08A8A]' : r.level === 'WARNING' ? 'text-warn' : ''}>{r.level.padEnd(7)}</span>{' '}
                <span className="text-ink-3">{r.logger}</span> {r.message}
              </li>
            ))}
          </ol>
        )}
      </Card>
    </div>
  )
}
