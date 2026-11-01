import { useEffect, useMemo, useState } from 'react'
import { APPROACHES, APPROACH_LABEL, fmt, type Approach, type Snapshot } from './api'
import { Badge, Banner, Card, Empty, ErrorState, Field, Loading, Table, inputCls, useResource, type Tone } from './ui'

type CamStatus = 'ok' | 'degraded' | 'stale' | 'disconnected'

interface DemoApproach {
  camera_source: string | null; true_arrivals: number | null; perceived_arrivals: number | null
  count_error_pct: number | null; live_count_error_pct: number | null; inserted_into_sumo: number
  fallback_s: number; reconnects: number | null; statuses_s: Record<CamStatus, number>
}
interface VisionStatus {
  cameras: { approach: Approach; source: string; kind: string; faults: { kind: string; start_s: number; end_s: number }[] }[]
  detector: { kind: string; model: string; model_present: boolean; license: string }
  health: { stale_after_s: number; min_fps: number; target_fps: number; min_contrast: number }
  bridge: { min_confidence: number; ewma_tau_s: number; min_trusted_s: number }
  fps: number
  packages: Record<string, string | null>
  latest_demo: null | {
    id: string; controller: string; seed: number; faults: boolean; duration_s: number; detector: string; note: string
    approaches: Record<Approach, DemoApproach>; transitions: { time_s: number; approach: Approach; mode: string; reason: string }[]
    wall_time_s: number; delay_mean_s: number
  }
  demo_runs: number; how_to_run: string; note: string
}
interface TimelinePoint { t: number; approach: Approach; status: CamStatus; mode: 'live' | 'fallback'; confidence: number; fps: number; vehicle_count: number; queue_veh: number }
interface Timeline { run: string; every_s: number; points: TimelinePoint[]; events: { time_s: string; approach: string; event: string }[]; overlays: string[] }

const STATUS_TONE: Record<CamStatus, Tone> = { ok: 'good', degraded: 'warn', stale: 'warn', disconnected: 'crit' }
// Fill for the timeline strip; every cell also has a text tooltip and the legend uses icon + label badges.
const STATUS_FILL: Record<CamStatus, string> = { ok: 'bg-good/70', degraded: 'bg-warn/80', stale: 'bg-warn/40', disconnected: 'bg-crit/80' }
const STATUS_LABEL: Record<CamStatus, string> = { ok: 'OK', degraded: 'Degraded', stale: 'Stale', disconnected: 'Disconnected' }

function StatusStrip({ points, approach }: { points: TimelinePoint[]; approach: Approach }) {
  const row = points.filter((p) => p.approach === approach)
  return (
    <div className="flex h-5 w-full gap-px" role="img" aria-label={`${APPROACH_LABEL[approach]} camera status over time`}>
      {row.map((p) => (
        <div key={p.t} className={`flex-1 rounded-[1px] ${STATUS_FILL[p.status] ?? 'bg-line'} ${p.mode === 'fallback' ? 'opacity-100 ring-1 ring-inset ring-bg' : ''}`}
          title={`t=${p.t}s · ${STATUS_LABEL[p.status] ?? p.status} · ${p.mode} · conf ${p.confidence.toFixed(2)} · ${p.fps.toFixed(1)} fps · ${p.vehicle_count} veh, ${p.queue_veh} queued`} />
      ))}
    </div>
  )
}

function OverlayViewer({ run, names }: { run: string; names: string[] }) {
  const [approach, setApproach] = useState<Approach>('N')
  const forApproach = useMemo(() => names.filter((n) => n.startsWith(`${approach}_`)), [names, approach])
  const [picked, setIdx] = useState<number | null>(null)  // null: start mid-run, not on the empty t=0 frame
  const idx = Math.min(picked ?? Math.floor(forApproach.length / 2), forApproach.length - 1)
  const name = forApproach[idx]
  const t = name ? Number(name.slice(2, 7)) : null
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Camera">
          <select className={inputCls} value={approach} onChange={(e) => { setApproach(e.target.value as Approach); setIdx(null) }}>
            {APPROACHES.map((a) => <option key={a} value={a}>{APPROACH_LABEL[a]}</option>)}
          </select>
        </Field>
        <Field label={`Snapshot (${forApproach.length})`} hint={t !== null ? `t = ${t} s` : undefined}>
          <input type="range" min={0} max={Math.max(0, forApproach.length - 1)} value={Math.max(0, idx)}
            onChange={(e) => setIdx(Number(e.target.value))} disabled={forApproach.length < 2} className="w-56 accent-[#06B6D4]" />
        </Field>
      </div>
      {name ? (
        <img src={`/api/vision/demos/${run}/overlays/${name}`} alt={`${APPROACH_LABEL[approach]} camera at t=${t}s with region of interest and tracked vehicles`}
          className="w-full max-w-md rounded border border-line [image-rendering:pixelated]" />
      ) : <Empty title="No snapshots for this camera" />}
      <p className="text-[11px] text-ink-3">Cyan: region of interest up to the stop line. Blue boxes: moving tracks. Amber boxes: stationary (queued after the dwell time). Header: status, vehicles, queue, processed fps.</p>
    </div>
  )
}

export default function CameraPanel() {
  const vs = useResource<VisionStatus>('/api/vision/status', 10000)
  const latest = vs.data?.latest_demo ?? null
  const tl = useResource<Timeline>(latest ? `/api/vision/demos/${latest.id}/timeline?every_s=10` : null)

  if (vs.loading) return <Card title="Cameras (Phase 3)"><Loading /></Card>
  if (vs.error || !vs.data) return <Card title="Cameras (Phase 3)"><ErrorState error={vs.error ?? 'no data'} onRetry={vs.reload} /></Card>
  const d = vs.data
  return (
    <div className="space-y-4">
      <Banner tone="info" title="Camera perception drives the SUMO simulation only">
        {d.note} Set up and watch cameras (one per direction) on the Cameras page; to drive a live run from them, tick "Camera-driven demand" on Overview. Recorded demo: <code className="font-mono text-ink">{d.how_to_run}</code>. Check a phone first with <code className="font-mono text-ink">python -m camp_crl.vision.probe --source &lt;url&gt;</code>.
      </Banner>
      <Card title="Camera configuration" subtitle={`configs/vision.yaml · processing ${d.fps} fps · stale after ${d.health.stale_after_s} s · degraded below ${d.health.min_fps} fps · bridge uses a camera only when OK and confidence ≥ ${d.bridge.min_confidence}`}>
        <div className="mb-3 flex flex-wrap gap-2 text-xs">
          <Badge tone={d.detector.kind === 'yolox' ? 'info' : 'neutral'}>Detector: {d.detector.kind}</Badge>
          <Badge tone={d.detector.model_present ? 'good' : 'warn'}>{d.detector.model} {d.detector.model_present ? 'present' : 'missing'}</Badge>
          {Object.entries(d.packages).map(([k, v]) => <Badge key={k} tone={v ? 'good' : 'crit'}>{k} {v ?? 'missing'}</Badge>)}
        </div>
        <p className="mb-3 text-xs text-ink-3">{d.detector.license}</p>
        <Table head={['Approach', 'Source', 'Type', 'Scripted faults (demo)']}>
          {d.cameras.map((c) => (
            <tr key={c.approach} className="border-b border-line/60">
              <td className="px-2 py-1.5">{APPROACH_LABEL[c.approach]}</td>
              <td className="px-2 py-1.5 font-mono text-xs">{c.source}</td>
              <td className="px-2 py-1.5">{c.kind}</td>
              <td className="px-2 py-1.5 text-xs text-ink-2">{c.faults.length ? c.faults.map((f) => `${f.kind} ${f.start_s}–${f.end_s} s`).join(', ') : '—'}</td>
            </tr>
          ))}
        </Table>
      </Card>

      {!latest ? (
        <Card title="Latest camera demo"><Empty title="No camera demo has been run yet">Run <code className="font-mono">{d.how_to_run}</code> and reload.</Empty></Card>
      ) : (
        <>
          <Card title="Latest camera demo" subtitle={`${latest.id} · controller ${latest.controller} · seed ${latest.seed} · ${latest.duration_s} s simulated · faults ${latest.faults ? 'on' : 'off'} · detector ${latest.detector}`}>
            {latest.detector === 'synthetic' && <p className="mb-3 text-xs text-warn">▲ {latest.note}.</p>}
            <Table head={['Approach', 'True arrivals', 'Perceived', 'Error (live periods)', 'Inserted into SUMO', 'Fallback time', 'Reconnects', 'Time per status']}>
              {APPROACHES.map((a) => {
                const r = latest.approaches[a]
                return (
                  <tr key={a} className="border-b border-line/60">
                    <td className="px-2 py-1.5">{APPROACH_LABEL[a]}</td>
                    <td className="px-2 py-1.5">{r.true_arrivals ?? '—'}</td>
                    <td className="px-2 py-1.5">{r.perceived_arrivals ?? '—'}</td>
                    <td className="px-2 py-1.5">{r.live_count_error_pct === null ? '—' : `${r.live_count_error_pct > 0 ? '+' : ''}${fmt(r.live_count_error_pct, 1)} %`}</td>
                    <td className="px-2 py-1.5">{r.inserted_into_sumo}</td>
                    <td className="px-2 py-1.5">{fmt(r.fallback_s, 0)} s</td>
                    <td className="px-2 py-1.5">{r.reconnects ?? '—'}</td>
                    <td className="px-2 py-1.5 text-xs text-ink-2">{(Object.keys(STATUS_LABEL) as CamStatus[]).filter((s) => r.statuses_s[s]).map((s) => `${STATUS_LABEL[s]} ${r.statuses_s[s]} s`).join(' · ')}</td>
                  </tr>
                )
              })}
            </Table>
            <p className="mt-2 text-[11px] text-ink-3">SUMO mean delay with camera-driven demand: {fmt(latest.delay_mean_s, 2)} s/veh. Run time {latest.wall_time_s} s. "Error (live periods)" compares perceived with true arrivals only while the bridge used the camera.</p>
          </Card>

          <Card title="Camera status over time" subtitle="One cell per 10 s. Hover a cell for status, bridge mode, confidence and fps.">
            <div className="mb-3 flex flex-wrap gap-2">
              {(Object.keys(STATUS_LABEL) as CamStatus[]).map((s) => <Badge key={s} tone={STATUS_TONE[s]}>{STATUS_LABEL[s]}</Badge>)}
            </div>
            {tl.loading ? <Loading /> : tl.error ? <ErrorState error={tl.error} onRetry={tl.reload} /> : tl.data && (
              <div className="space-y-2">
                {APPROACHES.map((a) => (
                  <div key={a} className="grid grid-cols-[4.5rem_1fr] items-center gap-2 text-xs">
                    <span className="text-ink-2">{APPROACH_LABEL[a]}</span>
                    <StatusStrip points={tl.data!.points} approach={a} />
                  </div>
                ))}
                <div className="grid grid-cols-[4.5rem_1fr] text-[11px] text-ink-3"><span />
                  <div className="flex justify-between"><span>0 s</span><span>{latest.duration_s} s</span></div>
                </div>
              </div>
            )}
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Annotated camera frames">
              {tl.data?.overlays.length ? <OverlayViewer run={latest.id} names={tl.data.overlays} /> : <Empty title="No overlay snapshots in this run" />}
            </Card>
            <Card title="Connection and fallback events">
              {!tl.data?.events.length ? <Empty title="No events" /> : (
                <ol className="max-h-80 space-y-0.5 overflow-y-auto font-mono text-[11px] text-ink-2">
                  {tl.data.events.map((e, i) => (
                    <li key={i}><span className="text-ink-3">t={Number(e.time_s).toFixed(0).padStart(5)}s</span> <span className="text-ink">{e.approach}</span> {e.event}</li>
                  ))}
                </ol>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

/** Overlay that refreshes every second; keeps showing the last good frame if a refresh fails. */
function LiveOverlay({ approach }: { approach: Approach }) {
  const [tick, setTick] = useState(0)
  const [failed, setFailed] = useState(false)
  useEffect(() => { const id = setInterval(() => setTick((k) => k + 1), 1000); return () => clearInterval(id) }, [])
  return failed && tick < 2 ? <div className="flex aspect-[4/3] items-center justify-center rounded border border-line text-xs text-ink-3">No frame yet</div> : (
    <img src={`/api/vision/live/${approach}/overlay.jpg?k=${tick}`} alt={`${APPROACH_LABEL[approach]} camera, live annotated frame`}
      onError={() => setFailed(true)} onLoad={() => setFailed(false)}
      className="aspect-[4/3] w-full rounded border border-line bg-bg-2 object-contain [image-rendering:pixelated]" />
  )
}

/** Live Intersection card for runs started with camera-driven demand. */
export function LiveCameras({ snap, clock }: { snap: Snapshot; clock: 'sim' | 'wall' | null | undefined }) {
  const rows = snap.perception
  if (!rows) return null
  return (
    <Card title="Cameras driving this run" subtitle={`Vehicles enter SUMO from camera counts; an approach whose camera is not OK uses fallback demand. Clock: ${clock === 'wall' ? 'wall (real streams, real time)' : 'simulation (synthetic cameras)'}.`}>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {APPROACHES.map((a) => {
          const r = rows[a]
          const st = (r?.status ?? 'disconnected') as CamStatus
          return (
            <div key={a} className="space-y-2 rounded-lg border border-line bg-bg-2 p-2">
              <div className="flex flex-wrap items-center justify-between gap-1 text-sm">
                <span className="font-medium">{APPROACH_LABEL[a]}</span>
                <span className="flex gap-1">
                  {r?.status ? <Badge tone={STATUS_TONE[st]}>{STATUS_LABEL[st]}</Badge> : <Badge tone="neutral">No camera</Badge>}
                  <Badge tone={r?.mode === 'live' ? 'info' : 'warn'}>{r?.mode === 'live' ? 'Camera demand' : 'Fallback demand'}</Badge>
                </span>
              </div>
              {r?.status && <LiveOverlay approach={a} />}
              <dl className="grid grid-cols-2 gap-x-2 text-xs tabular">
                <dt className="text-ink-3">Confidence</dt><dd>{fmt(r?.confidence ?? null, 2)}</dd>
                <dt className="text-ink-3">Processed fps</dt><dd>{fmt(r?.fps ?? null, 1)}</dd>
                <dt className="text-ink-3">In view / queued</dt><dd>{r?.vehicle_count ?? '—'} / {r?.queue_veh ?? '—'}</dd>
                <dt className="text-ink-3">Fallback rate</dt><dd>{r?.mode === 'fallback' ? `${fmt(r.fallback_rate_veh_h, 0)} veh/h` : '—'}</dd>
              </dl>
              {r?.reason && <p className="text-[11px] text-warn">▲ {r.reason}</p>}
            </div>
          )
        })}
      </div>
    </Card>
  )
}
