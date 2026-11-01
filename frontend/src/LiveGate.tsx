import type { ReactNode } from 'react'
import type { Snapshot } from './api'
import { useTelemetry } from './telemetry'
import { Banner, Empty, Loading } from './ui'

/** Shared disconnected / stale / empty handling for pages that show live data. */
export function LiveGate({ children }: { children: (snap: Snapshot) => ReactNode }) {
  const t = useTelemetry()
  if (!t.connected && t.lastMessageAt === null) return <Loading label="Connecting to backend…" />
  return (
    <div className="space-y-4">
      {!t.connected && (
        <Banner tone="crit" title="Disconnected from backend">
          Live data is paused and reconnecting automatically. Values below are the last received and may be outdated.
        </Banner>
      )}
      {t.connected && t.stale && (
        <Banner tone="warn" title="Stale data">
          No new simulation step for more than 3 s. The simulation may be stalled or overloaded.
        </Banner>
      )}
      {t.status?.status === 'error' && <Banner tone="crit" title="Simulation error">{t.status.error}</Banner>}
      {t.snapshot ? children(t.snapshot) : (
        <Empty title="No live simulation data">Start a simulation from the Overview page to see live data here.</Empty>
      )}
    </div>
  )
}
