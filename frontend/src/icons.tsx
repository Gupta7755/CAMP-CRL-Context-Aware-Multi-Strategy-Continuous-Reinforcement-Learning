/** Small inline stroke icons (24x24, currentColor), so the app needs no icon library. */
import type { ReactNode } from 'react'

function Svg({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden>{children}</svg>
  )
}

export const Icons: Record<string, ReactNode> = {
  overview: <Svg><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></Svg>,
  live: <Svg><path d="M12 3v18M3 12h18" /><circle cx="12" cy="12" r="3.5" /></Svg>,
  advisory: <Svg><rect x="8" y="2.5" width="8" height="19" rx="4" /><circle cx="12" cy="7.5" r="1.4" /><circle cx="12" cy="12" r="1.4" /><circle cx="12" cy="16.5" r="1.4" /></Svg>,
  cameras: <Svg><path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L17 6h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><circle cx="12" cy="12.5" r="3.5" /></Svg>,
  camp: <Svg><path d="M9.5 18h5M10 21h4" /><path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z" /></Svg>,
  lab: <Svg><path d="M9 3h6M10 3v6l-5.5 9.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3" /><path d="M7.5 15h9" /></Svg>,
  evaluation: <Svg><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></Svg>,
  config: <Svg><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></Svg>,
  health: <Svg><path d="M3 12h4l2.5-6 4 12 2.5-6H21" /></Svg>,
  methodology: <Svg><path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z" /><path d="M4 19V5M8 7h6M8 11h6" /></Svg>,
}

/** Brand mark: a signal head on the blue -> cyan gradient. */
export function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden>
      <defs>
        <linearGradient id="ss-logo" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#2563EB" /><stop offset="1" stopColor="#06B6D4" /></linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="9" fill="url(#ss-logo)" />
      <rect x="11" y="5.5" width="10" height="21" rx="5" fill="#0B1220" />
      <circle cx="16" cy="10.5" r="2.4" fill="#D03B3B" />
      <circle cx="16" cy="16" r="2.4" fill="#FAB219" />
      <circle cx="16" cy="21.5" r="2.4" fill="#22C55E" />
    </svg>
  )
}
