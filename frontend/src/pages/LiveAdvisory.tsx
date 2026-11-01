import { useState } from 'react'
import { APPROACHES, APPROACH_LABEL, api, fmt, type Approach } from '../api'
import { Badge, Banner, Button, Card, Empty, ErrorState, Loading, Table, useResource } from '../ui'

type Light = 'green' | 'yellow' | 'red'
interface PlanStep { label: string; color: Light; seconds: number | null; max_seconds?: number; note?: string | null }
interface AdviceApproach {
  known: boolean; light: Light; vehicles: number | null; queue: number | null; arrivals_veh_h: number | null
  longest_wait_s: number | null; trend: string | null; surge: boolean | null; urgency: number | null; red_for_s: number
}
interface Advice {
  advisory_only: true; mode: 'default_loop' | 'adaptive'; mode_reason: string; strategy: string | null; pattern: string | null
  signal: { mode: 'GREEN' | 'YELLOW' | 'ALL_RED'; green_phase: number; green_label: string; target_phase: number; time_in_interval_s: number }
  plan: PlanStep[]; approaches: Record<Approach, AdviceApproach>; cameras_working: number; default_green_s: number
  timing: { min_green_s: number; max_green_s: number; yellow_s: number; all_red_s: number; max_red_s: number }
}
interface AdvisoryStatus { running: boolean; error: string | null; advice: Advice | null; note: string }

const LIGHT_HEX: Record<Light, string> = { green: '#0CA30C', yellow: '#FAB219', red: '#D03B3B' }
const LIGHT_TEXT: Record<Light, string> = { green: 'GREEN', yellow: 'YELLOW', red: 'RED' }

function SignalHead({ light, label }: { light: Light; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex flex-col gap-1 rounded-md border border-line bg-bg p-1.5" aria-hidden>
        {(['red', 'yellow', 'green'] as Light[]).map((l) => (
          <span key={l} className="block h-4 w-4 rounded-full" style={{ background: l === light ? LIGHT_HEX[l] : '#1E293B' }} />
        ))}
      </div>
      <div>
        <div className="text-xs text-ink-3">{label}</div>
        <div className="text-sm font-semibold" style={{ color: LIGHT_HEX[light] }}>{LIGHT_TEXT[light]}</div>
      </div>
    </div>
  )
}

function stepTime(p: PlanStep): string {
  if (p.seconds === null) return `holding (max ${p.max_seconds} s)`
  return `${p.seconds} s`
}

export default function LiveAdvisory() {
  const res = useResource<AdvisoryStatus>('/api/advisory', 1000)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setErr(null)
    try { await fn(); res.reload() } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  if (res.loading) return <Loading />
  if (res.error && !res.data) return <ErrorState error={res.error} onRetry={res.reload} />
  const d = res.data!
  const a = d.advice
  const current = a?.plan[0]

  return (
    <div className="space-y-4">
      <Banner tone="warn" title="Advisory only">{d.note} Research prototype: timing advice is not validated for use at a real junction.</Banner>
      <Card title="Live signal-timing advice from your cameras"
        subtitle={`When all four roads are clear: default loop of ${a?.default_green_s ?? 30} s green each way. When traffic is seen: CAMP adapts green time from live counts, queues and waits. Yellow ${a?.timing.yellow_s ?? 3} s and all-red ${a?.timing.all_red_s ?? 2} s are fixed for safety.`}
        actions={d.running
          ? <Button variant="ghost" disabled={busy} onClick={() => act(() => api.post('/api/advisory/stop'))}>Stop advice</Button>
          : <Button disabled={busy} onClick={() => act(() => api.post('/api/advisory/start'))}>{busy ? 'Starting…' : 'Start live advice'}</Button>}>
        {err && <div className="mb-3"><Banner tone="crit" title="Could not start">{err}</Banner></div>}
        {d.error && <div className="mb-3"><Banner tone="crit" title="Advice stopped">{d.error}</Banner></div>}
        {!d.running && !a ? (
          <Empty title="Advice is not running">Set up cameras on the Cameras page (one per direction), then press Start live advice. Directions without a working camera are shown as unknown, never guessed.</Empty>
        ) : !a ? <Loading label="Waiting for the first camera readings…" /> : (
          <div className="grid gap-4 lg:grid-cols-[1.1fr_1fr]">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={a.mode === 'adaptive' ? 'info' : 'neutral'}>{a.mode === 'adaptive' ? 'Adaptive (CAMP)' : `Default ${a.default_green_s} s loop`}</Badge>
                <Badge tone={a.cameras_working === 4 ? 'good' : a.cameras_working ? 'warn' : 'crit'}>{a.cameras_working} of 4 cameras working</Badge>
                {!d.running && <Badge tone="neutral">Stopped (last advice)</Badge>}
              </div>
              <div className="rounded-lg border border-line bg-bg-2 p-4">
                <div className="text-xs text-ink-3">Recommended now</div>
                <div className="mt-1 text-2xl font-semibold" style={{ color: current ? LIGHT_HEX[current.color] : undefined }}>
                  {current?.label ?? '—'}
                </div>
                <div className="mt-1 text-lg tabular text-ink">{current ? (current.seconds === null ? `hold while traffic keeps coming · at most ${current.max_seconds} s more` : `${current.seconds} s left`) : ''}</div>
                {current?.note && <div className="mt-1 text-xs text-ink-3">{current.note}</div>}
              </div>
              <div>
                <div className="mb-1 text-xs text-ink-3">Coming next</div>
                <ol className="flex flex-wrap items-center gap-1.5 text-sm">
                  {a.plan.map((p, i) => (
                    <li key={i} className="flex items-center gap-1.5">
                      {i > 0 && <span aria-hidden className="text-ink-3">→</span>}
                      <span className="rounded border px-2 py-0.5 tabular" style={{ borderColor: LIGHT_HEX[p.color], color: LIGHT_HEX[p.color] }}>
                        {p.label} · {stepTime(p)}{i === a.plan.length - 1 && a.mode === 'adaptive' && p.color === 'green' ? ' (est.)' : ''}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
              <p className="text-xs text-ink-2"><span className="text-ink-3">Why:</span> {a.mode_reason}</p>
              {a.mode === 'adaptive' && <p className="text-xs text-ink-3">Strategy {a.strategy ?? '—'} · traffic pattern {a.pattern ?? '—'}. Green estimates assume about one vehicle leaves per second per approach after a 2 s start; CAMP re-decides every second, within {a.timing.min_green_s}–{a.timing.max_green_s} s green.</p>}
            </div>
            <div className="grid grid-cols-2 gap-3 self-start rounded-lg border border-line bg-bg-2 p-4">
              {APPROACHES.map((x) => <SignalHead key={x} light={a.approaches[x].light} label={`${APPROACH_LABEL[x]}${a.approaches[x].known ? '' : ' (no camera)'}`} />)}
            </div>
          </div>
        )}
      </Card>

      {a && (
        <Card title="What each camera sees" subtitle="Live values used for the advice. Unknown = no working camera on that direction.">
          <Table head={['Direction', 'Camera', 'Light', 'Vehicles', 'Queued', 'Arrivals', 'Longest stop', 'Trend', 'Surge', 'Red for']}>
            {APPROACHES.map((x) => {
              const r = a.approaches[x]
              return (
                <tr key={x} className="border-b border-line/60">
                  <td className="px-2 py-1.5">{APPROACH_LABEL[x]}</td>
                  <td className="px-2 py-1.5">{r.known ? <Badge tone="good">Working</Badge> : <Badge tone="neutral">Unknown</Badge>}</td>
                  <td className="px-2 py-1.5 font-semibold" style={{ color: LIGHT_HEX[r.light] }}>{LIGHT_TEXT[r.light]}</td>
                  <td className="px-2 py-1.5">{r.vehicles ?? '—'}</td>
                  <td className="px-2 py-1.5">{r.queue ?? '—'}</td>
                  <td className="px-2 py-1.5">{r.arrivals_veh_h === null ? '—' : `${r.arrivals_veh_h} veh/h`}</td>
                  <td className="px-2 py-1.5">{r.longest_wait_s === null ? '—' : `${fmt(r.longest_wait_s, 0)} s`}</td>
                  <td className="px-2 py-1.5">{r.trend ?? '—'}</td>
                  <td className="px-2 py-1.5">{r.surge === null ? '—' : r.surge ? <Badge tone="warn">Surge</Badge> : 'no'}</td>
                  <td className="px-2 py-1.5">{r.light === 'red' ? `${r.red_for_s} s` : '—'}</td>
                </tr>
              )
            })}
          </Table>
        </Card>
      )}

      {a && d.running && (
        <Card title="Live camera views" subtitle="Same video the advice is computed from (detections drawn).">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {APPROACHES.filter((x) => a.approaches[x].known).map((x) => (
              <figure key={x} className="space-y-1">
                <img src={`/api/cameras/${x}/live.mjpg?annotate=true`} alt={`${APPROACH_LABEL[x]} camera live video`}
                  className="aspect-video w-full rounded border border-line bg-bg object-contain" />
                <figcaption className="text-xs text-ink-3">{APPROACH_LABEL[x]}</figcaption>
              </figure>
            ))}
          </div>
          {!APPROACHES.some((x) => a.approaches[x].known) && <Empty title="No working cameras right now" />}
        </Card>
      )}
    </div>
  )
}
