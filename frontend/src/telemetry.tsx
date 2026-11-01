import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { api, type HistoryPoint, type SimStatus, type Snapshot } from './api'

const STALE_MS = 3000
const MAX_POINTS = 900

export interface Telemetry {
  connected: boolean
  status: SimStatus | null
  snapshot: Snapshot | null
  history: HistoryPoint[]
  lastMessageAt: number | null
  /** No message for STALE_MS, or the backend reports the running sim has not advanced. */
  stale: boolean
  refresh: () => void
}

export const TelemetryContext = createContext<Telemetry | null>(null)

export function useTelemetry(): Telemetry {
  const t = useContext(TelemetryContext)
  if (!t) throw new Error('useTelemetry must be used inside <TelemetryProvider>')
  return t
}

/** Single shared WebSocket with exponential-backoff reconnect; seeds history over REST. */
export function TelemetryProvider({ children, url }: { children: ReactNode; url?: string }) {
  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState<SimStatus | null>(null)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [history, setHistory] = useState<HistoryPoint[]>([])
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const runKey = useRef<string>('')

  const seed = useCallback(() => {
    api.get<HistoryPoint[]>('/api/simulation/history').then(setHistory).catch(() => undefined)
    api.get<SimStatus>('/api/simulation/status').then(setStatus).catch(() => undefined)
  }, [])

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry = 0
    let timer: ReturnType<typeof setTimeout> | undefined
    let closed = false
    const wsUrl = url ?? `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/telemetry`

    const connect = () => {
      ws = new WebSocket(wsUrl)
      ws.onopen = () => { retry = 0; setConnected(true); seed() }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) timer = setTimeout(connect, Math.min(1000 * 2 ** retry++, 8000))
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        setLastMessageAt(Date.now())
        setStatus(msg.status)
        if (msg.type === 'snapshot') {
          const snap = msg.data as Snapshot
          const key = `${msg.status?.started_at}|${snap.controller}|${snap.scenario}|${snap.seed}`
          setSnapshot(snap)
          setHistory((h) => {
            const fresh = key !== runKey.current
            runKey.current = key
            const base = fresh ? [] : h
            if (!msg.point || (base.length && base[base.length - 1].t >= msg.point.t)) return base
            const next = [...base, msg.point]
            return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next
          })
        }
      }
    }
    connect()
    const tick = setInterval(() => setNow(Date.now()), 1000)
    return () => { closed = true; clearTimeout(timer); clearInterval(tick); ws?.close() }
  }, [url, seed])

  const silent = lastMessageAt !== null && now - lastMessageAt > STALE_MS
  const stale = connected && (silent || Boolean(status?.stale))

  return (
    <TelemetryContext.Provider value={{ connected, status, snapshot, history, lastMessageAt, stale, refresh: seed }}>
      {children}
    </TelemetryContext.Provider>
  )
}
