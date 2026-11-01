import { useEffect, useMemo, useState } from 'react'
import { CONTROLLER_LABEL, api, type ControllerInfo, type ControllerName, type ExperimentManifest, type LogRecord, type ScenarioInfo, type Summary } from '../api'
import { ResultsView } from '../Results'
import { RLTraining } from '../RLTraining'
import { Badge, Banner, Button, Card, Empty, ErrorState, Field, Loading, SplitBadge, Table, inputCls, useResource, type Tone } from '../ui'

const STATUS_TONE: Record<string, Tone> = {
  queued: 'neutral', running: 'info', completed: 'good', completed_with_failures: 'warn', cancelled: 'neutral', failed: 'crit',
}

/** "1-5, 8" -> [1,2,3,4,5,8]; null when invalid. */
export function parseSeeds(text: string): number[] | null {
  const out: number[] = []
  for (const part of text.split(',').map((p) => p.trim()).filter(Boolean)) {
    const m = part.match(/^(\d+)\s*-\s*(\d+)$/)
    if (m) {
      const [a, b] = [Number(m[1]), Number(m[2])]
      if (b < a || b - a > 29) return null
      for (let i = a; i <= b; i++) out.push(i)
    } else if (/^\d+$/.test(part)) out.push(Number(part))
    else return null
  }
  const uniq = [...new Set(out)]
  return uniq.length && uniq.length <= 30 ? uniq : null
}

function NewExperiment({ running, onStarted }: { running: boolean; onStarted: (id: string) => void }) {
  const scenarios = useResource<ScenarioInfo[]>('/api/scenarios')
  const controllers = useResource<ControllerInfo[]>('/api/controllers')
  const [selS, setSelS] = useState<string[]>(['balanced'])
  const [selC, setSelC] = useState<ControllerName[]>(['fixed_time', 'camp_rule'])
  const [baseline, setBaseline] = useState<ControllerName>('fixed_time')
  const [seedText, setSeedText] = useState('1-3')
  const [allowTest, setAllowTest] = useState(false)
  const [workers, setWorkers] = useState(8)
  const [err, setErr] = useState<string | null>(null)
  const seeds = parseSeeds(seedText)
  const runs = (seeds?.length ?? 0) * selS.length * selC.length
  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v])

  useEffect(() => { if (!selC.includes(baseline) && selC.length) setBaseline(selC[0]) }, [selC, baseline])
  useEffect(() => { if (!allowTest) setSelS((s) => s.filter((n) => scenarios.data?.find((x) => x.name === n)?.split !== 'test')) }, [allowTest, scenarios.data])

  if (scenarios.loading || controllers.loading) return <Loading />
  if (scenarios.error) return <ErrorState error={scenarios.error} onRetry={scenarios.reload} />
  const submit = async () => {
    setErr(null)
    try {
      const m = await api.post<ExperimentManifest>('/api/experiments/run', { scenarios: selS, controllers: selC, seeds, baseline, allow_test: allowTest, workers })
      onStarted(m.id)
    } catch (e) { setErr((e as Error).message) }
  }
  const disabledReason = running ? 'Another experiment is running.' : !seeds ? 'Seeds are invalid.' : !selS.length ? 'Select at least one scenario.' : !selC.length ? 'Select at least one controller.' : null

  return (
    <Card title="New experiment" subtitle="Every controller runs the same demand for each (scenario, seed) pair">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <fieldset>
          <legend className="mb-2 text-xs text-ink-2">Scenarios</legend>
          <div className="space-y-1">
            {scenarios.data?.map((s) => {
              const locked = s.split === 'test' && !allowTest
              return (
                <label key={s.name} className={`flex items-center gap-2 text-sm ${locked ? 'opacity-50' : ''}`} title={locked ? 'Held-out: enable final evaluation below' : s.description}>
                  <input type="checkbox" disabled={locked} checked={selS.includes(s.name)} onChange={() => setSelS(toggle(selS, s.name))} />
                  {s.name} <SplitBadge split={s.split} />
                </label>
              )
            })}
          </div>
          <label className="mt-3 flex items-center gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={allowTest} onChange={(e) => setAllowTest(e.target.checked)} /> Final evaluation: allow held-out scenarios
          </label>
          {allowTest && <p className="mt-1 text-[11px] text-warn">Never tune anything on held-out results.</p>}
        </fieldset>
        <fieldset>
          <legend className="mb-2 text-xs text-ink-2">Controllers</legend>
          <div className="space-y-1">
            {controllers.data?.map((c) => (
              <label key={c.name} className={`flex items-center gap-2 text-sm ${c.available ? '' : 'opacity-50'}`} title={c.reason ?? ''}>
                <input type="checkbox" disabled={!c.available} checked={selC.includes(c.name)} onChange={() => setSelC(toggle(selC, c.name))} />
                {CONTROLLER_LABEL[c.name]}{!c.available && <span className="text-xs text-ink-3">— {c.reason}</span>}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="space-y-3">
          <Field label="Seeds" hint="e.g. 1-10 or 1,2,5 (max 30)">
            <input className={inputCls} value={seedText} onChange={(e) => setSeedText(e.target.value)} aria-invalid={!seeds} />
          </Field>
          <Field label="Baseline for paired comparison">
            <select className={inputCls} value={baseline} onChange={(e) => setBaseline(e.target.value as ControllerName)}>
              {selC.map((c) => <option key={c} value={c}>{CONTROLLER_LABEL[c]}</option>)}
            </select>
          </Field>
          <Field label="Parallel SUMO processes">
            <input className={inputCls} type="number" min={1} max={16} value={workers} onChange={(e) => setWorkers(Math.min(16, Math.max(1, Number(e.target.value) || 1)))} />
          </Field>
          <p className="text-xs text-ink-3">{runs} runs · each 1800 simulated seconds.</p>
          <Button disabled={Boolean(disabledReason)} title={disabledReason ?? ''} onClick={submit}>Run experiment</Button>
          {disabledReason && <p className="text-xs text-ink-3">{disabledReason}</p>}
          {err && <Banner tone="crit" title="Could not start">{err}</Banner>}
        </div>
      </div>
    </Card>
  )
}

function Logs() {
  const logs = useResource<LogRecord[]>('/api/logs/recent?limit=60', 2000)
  return (
    <Card title="Backend log" subtitle="Most recent first; refreshes every 2 s">
      {logs.error ? <ErrorState error={logs.error} /> : !logs.data?.length ? <Empty title="No log entries yet" /> : (
        <ol className="max-h-64 space-y-0.5 overflow-y-auto font-mono text-[11px] text-ink-2">
          {[...logs.data].reverse().map((r) => <li key={r.seq}><span className="text-ink-3">{r.time.slice(11)}</span> {r.level.padEnd(7)} {r.message}</li>)}
        </ol>
      )}
    </Card>
  )
}

export default function ExperimentLab() {
  const list = useResource<ExperimentManifest[]>('/api/experiments', 2000)
  const [selected, setSelected] = useState<string | null>(null)
  const running = useMemo(() => list.data?.find((m) => m.status === 'running' || m.status === 'queued') ?? null, [list.data])
  const current = selected ?? list.data?.[0]?.id ?? null
  const manifest = list.data?.find((m) => m.id === current) ?? null
  const done = manifest && !['running', 'queued'].includes(manifest.status)
  const results = useResource<{ summary: Summary }>(done ? `/api/experiments/${current}/results` : null)

  return (
    <div className="space-y-4">
      <NewExperiment running={Boolean(running)} onStarted={(id) => { setSelected(id); list.reload() }} />
      {running && (
        <Card title={`Running: ${running.id}`} subtitle={running.note}
          actions={<Button variant="danger" disabled={!running.cancellable} title={running.cancellable ? '' : running.note}
            onClick={() => api.post(`/api/experiments/${running.id}/cancel`).then(list.reload).catch(() => list.reload())}>Cancel</Button>}>
          <div className="h-2 w-full rounded bg-bg-2" role="progressbar" aria-valuemin={0} aria-valuemax={running.total_runs} aria-valuenow={running.completed_runs}>
            <div className="h-2 rounded bg-accent transition-all" style={{ width: `${(100 * running.completed_runs) / Math.max(running.total_runs, 1)}%` }} />
          </div>
          <p className="mt-2 text-xs text-ink-3 tabular">{running.completed_runs} / {running.total_runs} runs complete{running.failed_runs ? `, ${running.failed_runs} failed` : ''}</p>
        </Card>
      )}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Experiments" className="xl:col-span-2">
          {list.loading ? <Loading /> : list.error ? <ErrorState error={list.error} onRetry={list.reload} /> : !list.data?.length ? (
            <Empty title="No experiments yet">Configure one above, or run <code>python -m camp_crl.cli evaluate</code>.</Empty>
          ) : (
            <Table head={['ID', 'Kind', 'Status', 'Runs', 'Scenarios', '']}>
              {list.data.slice(0, 15).map((m) => (
                <tr key={m.id} className={`border-b border-line/50 ${m.id === current ? 'bg-surface-2' : ''}`}>
                  <td className="px-2 py-1.5 font-mono text-xs">{m.id}</td>
                  <td className="px-2 py-1.5">{m.params.kind}</td>
                  <td className="px-2 py-1.5"><Badge tone={STATUS_TONE[m.status]}>{m.status.replace(/_/g, ' ')}</Badge></td>
                  <td className="px-2 py-1.5">{m.completed_runs}/{m.total_runs}</td>
                  <td className="px-2 py-1.5 text-xs text-ink-3">{m.params.scenarios.length} · {m.params.controllers.length} ctrl · {m.params.seeds.length} seeds</td>
                  <td className="px-2 py-1.5"><Button variant="ghost" onClick={() => setSelected(m.id)}>View</Button></td>
                </tr>
              ))}
            </Table>
          )}
        </Card>
        <Logs />
      </div>
      {manifest && (
        <Card title={`Results: ${manifest.id}`}
          actions={done ? (
            <div className="flex gap-2">
              <a className="rounded-md border border-line bg-surface-2 px-3 py-1.5 text-sm hover:border-ink-3" href={`/api/experiments/${manifest.id}/results?format=csv`} download>Export CSV</a>
              <a className="rounded-md border border-line bg-surface-2 px-3 py-1.5 text-sm hover:border-ink-3" href={`/api/experiments/${manifest.id}/results`} download={`camp_crl_${manifest.id}.json`}>Export JSON</a>
            </div>
          ) : undefined}>
          {!done ? <Loading label="Experiment in progress…" /> : results.loading ? <Loading /> : results.error ? <ErrorState error={results.error} /> :
            results.data?.summary && Object.keys(results.data.summary).length ? <ResultsView summary={results.data.summary} /> : <Empty title="No results">All runs failed or the experiment was cancelled before any run finished.</Empty>}
        </Card>
      )}
      <RLTraining />
    </div>
  )
}
