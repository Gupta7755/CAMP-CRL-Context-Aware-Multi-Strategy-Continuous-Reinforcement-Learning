import { useEffect, useRef, useState } from 'react'
import { APPROACHES, APPROACH_LABEL, api, fmt, type Approach } from '../api'
import { Badge, Banner, Button, Card, ErrorState, Loading, inputCls, useResource, type Tone } from '../ui'

interface Perception {
  status: 'ok' | 'degraded' | 'stale' | 'disconnected'; reason: string; fps: number; confidence: number
  vehicle_count: number; queue_veh: number; arrivals_total: number; detect_ms: number | null; stream_fps?: number
}
interface PreviewStatus { running: boolean; frames: number; error: string | null; source: string; perception: Perception | null; events: { t: number; event: string }[] }
interface CameraRow { approach: Approach; source: string | null; kind: 'synthetic' | 'stream' | 'webcam' | 'file' | 'none'; origin: 'dashboard' | 'vision.yaml'; preview: PreviewStatus | null }
interface CamerasResponse { cameras: CameraRow[]; detector: { kind: string; model: string; model_present: boolean }; locked: boolean; note: string }
interface TestResult { ok: boolean; width: number | null; height: number | null; ms: number; error: string | null; source: string }

const STATUS_TONE: Record<Perception['status'], Tone> = { ok: 'good', degraded: 'warn', stale: 'warn', disconnected: 'crit' }
const KIND_LABEL: Record<CameraRow['kind'], string> = { synthetic: 'Synthetic (demo)', stream: 'Phone / IP stream', webcam: 'USB / laptop webcam', file: 'Video file', none: 'No camera' }

/** Live video: one long-lived MJPEG stream at the camera's own frame rate; the browser plays it natively.
 *  If the stream breaks (camera or server restarted), it reconnects after a short pause. */
function LiveFeed({ approach, annotate }: { approach: Approach; annotate: boolean }) {
  const [k, setK] = useState(0)
  const [waiting, setWaiting] = useState(true)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  useEffect(() => { setWaiting(true) }, [annotate])
  return (
    <div className="relative">
      <img key={`${annotate}-${k}`} src={`/api/cameras/${approach}/live.mjpg?annotate=${annotate}&k=${k}`}
        alt={`${APPROACH_LABEL[approach]} camera live video${annotate ? ' with detected vehicles' : ' (raw)'}`}
        onLoad={() => setWaiting(false)}
        onError={() => { setWaiting(true); timer.current = setTimeout(() => setK((x) => x + 1), 1500) }}
        className={`aspect-video w-full rounded border border-line bg-bg object-contain ${waiting ? 'opacity-0' : ''}`} />
      {waiting && <div className="absolute inset-0 flex items-center justify-center text-sm text-ink-3">Connecting to camera…</div>}
    </div>
  )
}

function CameraCard({ cam, locked, onChanged }: { cam: CameraRow; locked: boolean; onChanged: () => void }) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [test, setTest] = useState<TestResult | null>(null)
  const [annotate, setAnnotate] = useState(true)
  const a = cam.approach
  const running = Boolean(cam.preview?.running)
  const per = cam.preview?.perception ?? null
  const lockReason = locked ? 'A live run is using the cameras. Stop it on Overview to change sources.' : null

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label); setErr(null)
    try { await fn(); onChanged() } catch (e) { setErr((e as Error).message) } finally { setBusy(null) }
  }
  // Saving stops the old live view on the server; if one was open, reopen it on the new source.
  const save = (source: string) => act('save', async () => {
    await api.put(`/api/cameras/${a}`, { source }); setValue(''); setTest(null)
    if (running && source !== 'none') await api.post(`/api/cameras/${a}/preview/start`)
  })
  const unsaved = value.trim() !== '' && value.trim() !== cam.source

  return (
    <Card title={<span className="flex items-center gap-2">{APPROACH_LABEL[a]} approach <span className="text-xs font-normal text-ink-3">({a})</span></span>}
      actions={<div className="flex flex-wrap gap-1"><Badge tone={cam.kind === 'none' ? 'neutral' : cam.kind === 'synthetic' ? 'info' : 'good'}>{KIND_LABEL[cam.kind]}</Badge></div>}>
      <div className="space-y-3">
        <div className="text-xs text-ink-3">
          Current source: <span className="break-all font-mono text-ink-2">{cam.source ?? '—'}</span>
          <span className="ml-2">({cam.origin === 'dashboard' ? 'set here' : 'from configs/vision.yaml'})</span>
        </div>

        <form className="flex flex-col gap-2 sm:flex-row" onSubmit={(e) => { e.preventDefault(); if (value.trim()) save(value.trim()) }}>
          <label className="sr-only" htmlFor={`src-${a}`}>Camera URL for {APPROACH_LABEL[a]}</label>
          <input id={`src-${a}`} className={`${inputCls} min-w-0 flex-1 font-mono placeholder:text-ink-3/60`} placeholder="http://192.168.0.110:8080/video"
            value={value} onChange={(e) => setValue(e.target.value)} disabled={locked} autoComplete="off" spellCheck={false} />
          <div className="flex gap-2">
            <Button type="submit" disabled={locked || !value.trim() || busy !== null} title={lockReason ?? 'Save this source for the approach'}>{busy === 'save' ? 'Saving…' : 'Save'}</Button>
            <Button variant="ghost" disabled={busy !== null || (!value.trim() && cam.kind === 'none')}
              title={value.trim() ? 'Try the URL typed here without saving it' : 'Try the saved source'}
              onClick={() => act('test', async () => setTest(await api.post<TestResult>(`/api/cameras/${a}/test`, { source: value.trim() || null })))}>
              {busy === 'test' ? 'Testing…' : 'Test'}
            </Button>
          </div>
        </form>
        {unsaved && <p className="text-xs text-warn">▲ Not saved yet: press Save to use this address{running ? ' (the live view still shows the current source)' : ''}.</p>}
        <div className="flex flex-wrap gap-2 text-xs">
          <Button variant="ghost" disabled={locked || busy !== null || cam.kind === 'synthetic'} title={lockReason ?? 'Use the simulated demo camera'} onClick={() => save('synthetic')}>Use synthetic</Button>
          <Button variant="ghost" disabled={locked || busy !== null || cam.kind === 'none'} title={lockReason ?? 'No camera on this approach: its demand uses the fallback rate'} onClick={() => save('none')}>No camera</Button>
          <Button variant="ghost" disabled={locked || busy !== null || cam.origin !== 'dashboard'} title={lockReason ?? (cam.origin === 'dashboard' ? 'Go back to the source in configs/vision.yaml' : 'Already using configs/vision.yaml')}
            onClick={() => act('reset', () => api.delete(`/api/cameras/${a}`))}>Reset to file default</Button>
        </div>
        {test && (test.ok
          ? <Banner tone="good" title="Camera reachable">Got a {test.width}×{test.height} frame in {test.ms} ms.</Banner>
          : <Banner tone="crit" title="Camera not reachable">{test.error} ({test.ms} ms). Check that the phone app is streaming, the phone and this computer are on the same Wi-Fi, and the URL ends in <code className="font-mono">/video</code>.</Banner>)}
        {err && <Banner tone="crit" title="Request failed">{err}</Banner>}

        <div className="flex items-center justify-between gap-2 border-t border-line pt-3">
          <span className="text-sm font-medium">Live view</span>
          {running
            ? <Button variant="ghost" disabled={busy !== null} onClick={() => act('stop', () => api.post(`/api/cameras/${a}/preview/stop`))}>Stop live view</Button>
            : <Button disabled={cam.kind === 'none' || busy !== null} title={cam.kind === 'none' ? 'No camera configured on this approach' : 'Show the camera feed with detected vehicles'}
                onClick={() => act('start', () => api.post(`/api/cameras/${a}/preview/start`))}>{busy === 'start' ? 'Starting…' : 'Start live view'}</Button>}
        </div>
        {running && (
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs text-ink-2">
              <input type="checkbox" className="accent-[#06B6D4]" checked={annotate} onChange={(e) => setAnnotate(e.target.checked)} />
              Show detections on the video (off: raw camera picture; detection keeps running either way)
            </label>
            <LiveFeed approach={a} annotate={annotate} />
            <div className="flex flex-wrap items-center gap-2 text-xs">
              {per ? <Badge tone={STATUS_TONE[per.status]}>{per.status.toUpperCase()}</Badge> : <Badge tone="neutral">Starting</Badge>}
              <span className="text-sm font-semibold tabular text-ink">{per?.vehicle_count ?? 0} vehicles</span>
              <span className="tabular text-ink-2">· {per?.queue_veh ?? 0} queued · {per?.arrivals_total ?? 0} counted since start</span>
            </div>
            <p className="text-xs tabular text-ink-3">Video {fmt(per?.stream_fps ?? null, 1)} fps · detection {fmt(per?.fps ?? null, 1)} per s, {fmt(per?.detect_ms ?? null, 0)} ms each · confidence {fmt(per?.confidence ?? null, 2)}</p>
            {per?.reason && <p className="text-xs text-warn">▲ {per.reason}</p>}
            {cam.preview?.error && <p className="text-xs text-[#F08A8A]">■ Live view stopped: {cam.preview.error}</p>}
            <p className="text-[11px] text-ink-3">Vehicles are counted inside the cyan region. Blue boxes: moving; amber: stopped (queued). The live view stops by itself 30 s after you leave this page.</p>
          </div>
        )}
      </div>
    </Card>
  )
}

export default function CameraSetup() {
  const res = useResource<CamerasResponse>('/api/cameras', 2000)
  if (res.loading) return <Loading />
  if (res.error || !res.data) return <ErrorState error={res.error ?? 'no data'} onRetry={res.reload} />
  const d = res.data
  const byApproach = Object.fromEntries(d.cameras.map((c) => [c.approach, c])) as Record<Approach, CameraRow>
  return (
    <div className="space-y-4">
      <Banner tone="info" title="Connect one camera per direction">
        On each phone, open <b>IP Webcam</b> and tap <b>Start server</b>, then paste the address it shows plus <code className="font-mono">/video</code> (for example <code className="font-mono">http://192.168.0.110:8080/video</code>) into the direction it watches. Phones and this computer must be on the same Wi-Fi. {d.note}
      </Banner>
      {d.locked && <Banner tone="warn" title="Sources locked">A live run is using the cameras. Live views still work; stop the run on Overview to change sources.</Banner>}
      {!d.detector.model_present && d.cameras.some((c) => !['synthetic', 'none'].includes(c.kind)) && (
        <Banner tone="crit" title="Vehicle detector missing">{d.detector.model} was not found; real cameras cannot be analysed. See models/README.md.</Banner>
      )}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {APPROACHES.map((a) => <CameraCard key={a} cam={byApproach[a]} locked={d.locked} onChanged={res.reload} />)}
      </div>
      <p className="text-xs text-ink-3">To drive a simulation from these cameras, tick “Camera-driven demand” on Overview. Directions with no camera, or with a camera that is not healthy, use fallback demand.</p>
    </div>
  )
}
