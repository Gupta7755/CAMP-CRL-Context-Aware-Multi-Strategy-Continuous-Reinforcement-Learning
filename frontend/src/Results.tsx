import { useState } from 'react'
import { ctrlLabel, fmt, type Summary } from './api'
import { EntityBars } from './charts'
import { Badge, Card, Field, SplitBadge, Table, inputCls } from './ui'

const METRIC_LABEL: Record<string, string> = {
  delay_mean_s: 'Mean delay (s)', delay_median_s: 'Median delay (s)', delay_p95_s: 'p95 delay (s)',
  queue_mean_veh: 'Mean queue (veh)', queue_max_veh: 'Max queue (veh)', throughput_completed: 'Completed trips',
  unfinished: 'Unfinished trips', travel_time_mean_s: 'Mean travel time (s)', stops_mean: 'Stops per trip',
  max_wait_s: 'Max wait (s)', approach_delay_std_s: 'Approach delay std (s) — fairness',
  spillback_steps: 'Spillback steps', downstream_blocking_steps: 'Downstream blocking steps',
  phase_switches: 'Phase switches', safety_interventions_total: 'Safety interventions', teleports: 'Teleports',
}
export const metricLabel = (m: string) => METRIC_LABEL[m] ?? m

const ci = (c: [number, number] | null) => (c ? `[${c[0].toFixed(2)}, ${c[1].toFixed(2)}]` : '')

export function ResultsView({ summary, defaultScenario }: { summary: Summary; defaultScenario?: string }) {
  const scenarios = Object.keys(summary.scenarios)
  const [metric, setMetric] = useState('delay_mean_s')
  const [scenario, setScenario] = useState(defaultScenario && scenarios.includes(defaultScenario) ? defaultScenario : scenarios[0])
  const ctrls = summary.controllers
  const sc = summary.scenarios[scenario]
  const lower = summary.metrics.find((m) => m.name === metric)?.lower_is_better ?? true
  if (!sc) return null
  const bars = ctrls.map((c) => {
    const d = sc.controllers[c]?.[metric]
    return { name: ctrlLabel(c), value: d?.mean ?? null, err: d?.ci95 ?? undefined }
  })
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <Field label="Metric">
          <select className={inputCls} value={metric} onChange={(e) => setMetric(e.target.value)}>
            {summary.metrics.map((m) => <option key={m.name} value={m.name}>{metricLabel(m.name)}</option>)}
          </select>
        </Field>
        <Field label="Scenario">
          <select className={inputCls} value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {scenarios.map((s) => <option key={s} value={s}>{s} ({summary.scenarios[s].split})</option>)}
          </select>
        </Field>
        <p className="self-end text-xs text-ink-3">{lower ? 'Lower is better.' : 'Higher is better.'} Bars: mean over seeds; whiskers: 95% bootstrap CI.</p>
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title={<>{metricLabel(metric)} · {scenario} <SplitBadge split={sc.split} /></>}
          subtitle={`Grey bar = baseline (${ctrlLabel(summary.baseline)})`}>
          <EntityBars data={bars} unit="" highlight={ctrlLabel(summary.baseline)} />
        </Card>
        <Card title={`Paired vs baseline: ${ctrlLabel(summary.baseline)}`} subtitle="Per-seed difference (candidate − baseline), bootstrap 95% CI">
          <Table head={['Controller', 'Mean diff', '95% CI', 'Change vs baseline', 'CI excludes 0']}>
            {ctrls.filter((c) => c !== summary.baseline).map((c) => {
              const p = sc.paired_vs_baseline[c]?.[metric]
              if (!p) return null
              const better = p.improvement_pct_of_means !== null && p.improvement_pct_of_means > 0
              return (
                <tr key={c} className="border-b border-line/50">
                  <td className="px-2 py-1.5">{ctrlLabel(c)}</td>
                  <td className="px-2 py-1.5">{fmt(p.mean_diff, 2)}</td>
                  <td className="px-2 py-1.5 text-ink-3">{ci(p.ci95)}</td>
                  <td className="px-2 py-1.5">{p.improvement_pct_of_means === null ? '—' : `${p.improvement_pct_of_means > 0 ? '+' : ''}${p.improvement_pct_of_means.toFixed(1)} % ${better ? 'better' : 'worse'}`}</td>
                  <td className="px-2 py-1.5">{p.ci_excludes_zero ? <Badge tone="info">Yes</Badge> : <span className="text-ink-3">No</span>}</td>
                </tr>
              )
            })}
          </Table>
          <p className="mt-2 text-xs text-ink-3">“CI excludes 0” is not a corrected significance test; many metrics and scenarios are compared.</p>
        </Card>
      </div>
      <Card title="All scenarios" subtitle={`${metricLabel(metric)} — mean [95% CI] over seeds`}>
        <Table head={['Scenario', 'Split', ...ctrls.map((c) => ctrlLabel(c))]}>
          {scenarios.map((s) => (
            <tr key={s} className="border-b border-line/50">
              <td className="px-2 py-1.5">{s}</td>
              <td className="px-2 py-1.5"><SplitBadge split={summary.scenarios[s].split} /></td>
              {ctrls.map((c) => {
                const d = summary.scenarios[s].controllers[c]?.[metric]
                return <td key={c} className="px-2 py-1.5">{fmt(d?.mean, 2)} <span className="text-xs text-ink-3">{ci(d?.ci95 ?? null)}</span></td>
              })}
            </tr>
          ))}
        </Table>
      </Card>
    </div>
  )
}
