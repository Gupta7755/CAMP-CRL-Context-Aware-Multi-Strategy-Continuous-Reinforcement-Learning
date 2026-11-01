import { useEffect, useState } from 'react'
import { api } from '../api'
import { useTelemetry } from '../telemetry'
import { Badge, Banner, Button, Card, ErrorState, Field, Loading, NotYet, inputCls, useResource } from '../ui'

type Cfg = Record<string, Record<string, unknown>>
interface ConfigResponse { config: Cfg; source: string; modified: boolean; note: string }

// [section, key path, label, step]
const FIELDS: { title: string; subtitle: string; items: [string, string, string, number][] }[] = [
  { title: 'Signal timing limits', subtitle: 'Enforced by the independent safety filter (seconds)', items: [
    ['timing', 'min_green_s', 'Min green', 1], ['timing', 'max_green_s', 'Max green', 1], ['timing', 'yellow_s', 'Yellow (≥ 3)', 0.5],
    ['timing', 'all_red_s', 'All-red', 0.5], ['timing', 'max_red_s', 'Max red (starvation bound)', 1]] },
  { title: 'Actuated baseline', subtitle: 'Virtual stop-line detector', items: [
    ['actuated', 'detection_zone_m', 'Detection zone (m)', 1], ['actuated', 'gap_s', 'Gap-out (s)', 0.5], ['actuated', 'max_green_s', 'Max green (s)', 1]] },
  { title: 'CAMP composite weights', subtitle: 'C = P + λM·M + λF·F + λS·S', items: [
    ['camp', 'lambda_momentum', 'λ momentum', 0.05], ['camp', 'lambda_risk', 'λ risk', 0.05], ['camp', 'lambda_fairness', 'λ fairness', 0.05],
    ['camp', 'risk_horizon_s', 'Risk horizon τ (s)', 1], ['camp', 'fairness_ref_wait_s', 'Fairness W_ref (s)', 5]] },
  { title: 'Pressure weights', subtitle: 'Must sum to 1', items: [
    ['camp.pressure', 'queue', 'Queue', 0.05], ['camp.pressure', 'arrival', 'Arrival', 0.05], ['camp.pressure', 'wait', 'Wait', 0.05],
    ['camp.pressure', 'density', 'Density', 0.05], ['camp.pressure', 'spillback', 'Spillback', 0.05]] },
  { title: 'Strategy rules', subtitle: 'Thresholds on normalised CAMP terms + hysteresis', items: [
    ['strategy', 'fairness_threshold', 'Fairness trigger', 0.05], ['strategy', 'congestion_pressure', 'Congestion trigger', 0.05],
    ['strategy', 'dominance_gap', 'Dominance gap', 0.05], ['strategy', 'overflow_congestion', 'Overflow exit congestion', 0.05],
    ['strategy', 'confirm_steps', 'Confirm steps', 1], ['strategy', 'min_hold_s', 'Min hold (s)', 1]] },
  { title: 'Simulation', subtitle: 'Episode length (seconds)', items: [
    ['simulation', 'duration_s', 'Duration', 60], ['simulation', 'warmup_s', 'Warm-up', 30]] },
]

function getIn(cfg: Cfg, section: string, key: string): number {
  const parts = section.split('.')
  let node: unknown = cfg
  for (const p of parts) node = (node as Record<string, unknown>)[p]
  return Number((node as Record<string, unknown>)[key])
}
function setIn(cfg: Cfg, section: string, key: string, value: number): Cfg {
  const copy = structuredClone(cfg)
  let node: Record<string, unknown> = copy
  for (const p of section.split('.')) node = node[p] as Record<string, unknown>
  node[key] = value
  return copy
}

function RLSettings() {
  const rl = useResource<{ config: Record<string, Record<string, unknown> | string[]>; source: string; note: string }>('/api/rl/config')
  if (rl.loading) return <Card title="RL settings"><Loading /></Card>
  if (rl.error || !rl.data) return <NotYet title="RL settings" reason={rl.error ?? 'RL configuration unavailable.'} />
  const c = rl.data.config
  const rows = (o: Record<string, unknown>) => Object.entries(o).map(([k, v]) => (
    <div key={k} className="flex justify-between gap-2 border-b border-line/40 py-1"><dt className="text-ink-3">{k}</dt><dd className="tabular">{Array.isArray(v) ? v.join(', ') : String(v)}</dd></div>
  ))
  return (
    <Card title="RL settings (read-only)" subtitle={`${rl.data.source}: ${rl.data.note}`}>
      <div className="grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <dl><h3 className="mb-1 font-semibold text-ink-2">Algorithm</h3>{rows(c.algo as Record<string, unknown>)}</dl>
        <dl><h3 className="mb-1 font-semibold text-ink-2">Reward weights</h3>{rows(c.reward as Record<string, unknown>)}</dl>
        <dl><h3 className="mb-1 font-semibold text-ink-2">Action / scenarios</h3>
          {rows({ ...(c.action as Record<string, unknown>), train: (c.train_scenarios as string[]).join(', '), validation: (c.validation_scenarios as string[]).join(', ') })}</dl>
      </div>
    </Card>
  )
}

export default function Configuration() {
  const t = useTelemetry()
  const res = useResource<ConfigResponse>('/api/config')
  const [draft, setDraft] = useState<Cfg | null>(null)
  const [errors, setErrors] = useState<{ loc: string; msg: string }[]>([])
  const [msg, setMsg] = useState<{ tone: 'good' | 'crit' | 'info'; text: string } | null>(null)
  const live = ['starting', 'running', 'stopping'].includes(t.status?.status ?? '')

  useEffect(() => { if (res.data) setDraft(res.data.config) }, [res.data])
  if (res.loading) return <Loading />
  if (res.error || !res.data || !draft) return <ErrorState error={res.error ?? 'no config'} onRetry={res.reload} />

  const validate = async () => {
    const v = await api.post<{ valid: boolean; errors: { loc: string; msg: string }[] }>('/api/config/validate', { config: draft })
    setErrors(v.errors)
    setMsg(v.valid ? { tone: 'good', text: 'Configuration is valid.' } : { tone: 'crit', text: `${v.errors.length} validation error(s).` })
    return v.valid
  }
  const apply = async () => {
    if (!(await validate())) return
    try { await api.put('/api/config', { config: draft }); setMsg({ tone: 'good', text: 'Applied. New runs in this session use it.' }); res.reload() }
    catch (e) { setMsg({ tone: 'crit', text: (e as Error).message }) }
  }
  const reset = async () => { await api.post('/api/config/reset'); setErrors([]); setMsg({ tone: 'info', text: `Reset to ${res.data!.source}.` }); res.reload() }

  return (
    <div className="space-y-4">
      <Banner tone="info" title={`Source: ${res.data.source}${res.data.modified ? ' (modified in memory)' : ''}`}>{res.data.note} Values are starting defaults, not tuned results.</Banner>
      {live && <Banner tone="warn" title="Live simulation running">Stop it before applying changes.</Banner>}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {FIELDS.map((sec) => (
          <Card key={sec.title} title={sec.title} subtitle={sec.subtitle}>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              {sec.items.map(([section, key, label, step]) => {
                const err = errors.find((e) => e.loc.includes(key) || e.loc === section.split('.').pop())
                return (
                  <Field key={section + key} label={label}>
                    <input className={`${inputCls} ${err ? 'border-crit' : ''}`} type="number" step={step} aria-invalid={Boolean(err)}
                      value={Number.isFinite(getIn(draft, section, key)) ? getIn(draft, section, key) : ''}
                      onChange={(e) => setDraft(setIn(draft, section, key, e.target.value === '' ? NaN : Number(e.target.value)))} />
                  </Field>
                )
              })}
            </div>
          </Card>
        ))}
        <Card title="Seeds and scenarios">
          <p className="text-sm text-ink-2">Seeds are chosen per run (Overview) or per experiment (Experiment Lab). Scenarios are defined in <code>configs/default.yaml</code>; editing them here is disabled so the train / validation / held-out split cannot be changed accidentally.</p>
          <div className="mt-2 flex flex-wrap gap-1">{Object.keys(draft.scenarios ?? {}).map((s) => <Badge key={s} tone="neutral">{s}</Badge>)}</div>
        </Card>
        <RLSettings />
      </div>
      {errors.length > 0 && (
        <Banner tone="crit" title="Validation errors">
          <ul className="list-disc pl-5">{errors.map((e, i) => <li key={i}><span className="font-mono text-xs">{e.loc}</span>: {e.msg}</li>)}</ul>
        </Banner>
      )}
      {msg && <Banner tone={msg.tone} title={msg.text} />}
      <div className="flex gap-2">
        <Button variant="ghost" onClick={validate}>Validate</Button>
        <Button onClick={apply} disabled={live} title={live ? 'Stop the live simulation first' : ''}>Apply</Button>
        <Button variant="ghost" onClick={reset}>Reset to file</Button>
      </div>
    </div>
  )
}
