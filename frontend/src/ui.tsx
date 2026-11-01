import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { ApiError, api } from './api'

export function Card({ title, subtitle, actions, children, className = '' }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={`surface-card rounded-xl p-4 ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, unit, hint }: { label: string; value: ReactNode; unit?: string; hint?: string }) {
  return (
    <div className="surface-card relative overflow-hidden rounded-xl px-4 py-3" title={hint}>
      <div className="accent-rule absolute inset-x-0 top-0 opacity-70" aria-hidden />
      <div className="text-xs text-ink-3">{label}</div>
      <div className="mt-1 text-2xl font-semibold tracking-tight tabular text-ink">
        {value}{unit && <span className="ml-1 text-sm font-normal text-ink-3">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-ink-3">{hint}</div>}
    </div>
  )
}

export type Tone = 'good' | 'warn' | 'crit' | 'info' | 'neutral'
const TONE: Record<Tone, { cls: string; icon: string }> = {
  good: { cls: 'border-good/50 bg-good/10 text-[#7DDC7D]', icon: '●' },
  warn: { cls: 'border-warn/50 bg-warn/10 text-warn', icon: '▲' },
  crit: { cls: 'border-crit/60 bg-crit/15 text-[#F08A8A]', icon: '■' },
  info: { cls: 'border-accent/50 bg-accent/10 text-[#67E3F5]', icon: 'i' },
  neutral: { cls: 'border-line bg-surface-2 text-ink-2', icon: '○' },
}

/** Status badge: always icon + text, never colour alone. */
export function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  const t = TONE[tone]
  return (
    <span className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium ${t.cls}`}>
      <span aria-hidden className="text-[10px] leading-none">{t.icon}</span>{children}
    </span>
  )
}

export function Banner({ tone, title, children }: { tone: Tone; title: string; children?: ReactNode }) {
  const t = TONE[tone]
  return (
    <div role={tone === 'crit' ? 'alert' : 'status'} className={`rounded-lg border px-4 py-3 text-sm ${t.cls}`}>
      <div className="font-semibold"><span aria-hidden className="mr-2">{t.icon}</span>{title}</div>
      {children && <div className="mt-1 text-ink-2">{children}</div>}
    </div>
  )
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-2 py-6 text-sm text-ink-3">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-ink-3 border-t-transparent" aria-hidden />
      {label}
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-4 py-8 text-center">
      <div className="text-sm font-medium text-ink-2">{title}</div>
      {children && <div className="mt-1 text-xs text-ink-3">{children}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <Banner tone="crit" title="Could not load data">
      {error}
      {onRetry && <div className="mt-2"><Button onClick={onRetry} variant="ghost">Retry</Button></div>}
    </Banner>
  )
}

/** A feature that exists in the spec but not yet in this build: shown, disabled, with the reason. */
export function NotYet({ title, reason }: { title: string; reason: string }) {
  return (
    <div aria-disabled className="rounded-lg border border-dashed border-line bg-bg-2 px-4 py-3">
      <div className="flex items-center gap-2 text-sm font-medium text-ink-2">{title}<Badge tone="neutral">Not available</Badge></div>
      <p className="mt-1 text-xs text-ink-3">{reason}</p>
    </div>
  )
}

export function Button({ children, onClick, disabled, variant = 'primary', type = 'button', title }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; variant?: 'primary' | 'ghost' | 'danger'; type?: 'button' | 'submit'; title?: string
}) {
  const cls = {
    primary: 'bg-brand bg-brand-hover text-white shadow-[0_4px_14px_-6px_rgba(6,182,212,0.6)]',
    ghost: 'border border-line bg-surface-2/80 text-ink hover:border-accent/50',
    danger: 'border border-crit/70 bg-crit/15 text-[#F08A8A] hover:bg-crit/25',
  }[variant]
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title}
      className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none ${cls}`}>
      {children}
    </button>
  )
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-ink-2">
      <span>{label}</span>
      {children}
      {hint && <span className="text-[11px] text-ink-3">{hint}</span>}
    </label>
  )
}

export const inputCls = 'rounded-lg border border-line bg-bg-2/80 px-2.5 py-1.5 text-sm text-ink tabular transition focus:border-accent focus:ring-2 focus:ring-accent/25'

export function SplitBadge({ split }: { split: string }) {
  if (split === 'test') return <Badge tone="warn">held-out</Badge>
  if (split === 'validation') return <Badge tone="info">validation</Badge>
  return <Badge tone="neutral">train</Badge>
}

export function Table({ head, children, caption }: { head: ReactNode[]; children: ReactNode; caption?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        {caption && <caption className="mb-2 text-left text-xs text-ink-3">{caption}</caption>}
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-3">
            {head.map((h, i) => <th key={i} scope="col" className="px-2 py-2 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody className="tabular">{children}</tbody>
      </table>
    </div>
  )
}

export interface Resource<T> { data: T | null; error: string | null; loading: boolean; status: number | null; reload: () => void }

/** Fetch (and optionally poll) a GET endpoint with loading/error/404 handling. */
export function useResource<T>(path: string | null, pollMs?: number): Resource<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<number | null>(null)
  const [loading, setLoading] = useState<boolean>(Boolean(path))
  const alive = useRef(true)
  const load = useCallback(() => {
    if (!path) return
    api.get<T>(path)
      .then((d) => { if (alive.current) { setData(d); setError(null); setStatus(200) } })
      .catch((e: ApiError) => { if (alive.current) { setError(e.message); setStatus(e.status); if (e.status === 404) setData(null) } })
      .finally(() => { if (alive.current) setLoading(false) })
  }, [path])
  useEffect(() => {
    alive.current = true
    setLoading(Boolean(path))
    load()
    const id = pollMs ? setInterval(load, pollMs) : undefined
    return () => { alive.current = false; if (id) clearInterval(id) }
  }, [load, path, pollMs])
  return { data, error, loading, status, reload: load }
}
