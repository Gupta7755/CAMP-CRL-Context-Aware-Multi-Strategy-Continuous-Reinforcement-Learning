import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ComposedChart, Tooltip, XAxis, YAxis } from 'recharts'
import { fmt } from './api'
import { Badge, Card, Empty, ErrorState, Field, Loading, Table, inputCls, useResource, type Tone } from './ui'

interface RLRun { id: string; algo: string; obs: string; seed: number; timesteps: number; status: string; created: string; wall_time_s?: number; error?: string }
interface Curves {
  episodes: { episode: number; timesteps: number; reward: number; reward_avg: number; scenario: string }[]
  validation: { timesteps: number; mean_reward: number; mean_delay_s: number | null }[]
  window: number
}

const TONE: Record<string, Tone> = { running: 'info', completed: 'good', failed: 'crit' }
const AXIS = { stroke: '#7F8DA6', fontSize: 12, tickLine: false }
const TIP = { contentStyle: { background: '#0B1220', border: '1px solid #2B3A55', borderRadius: 6, fontSize: 12, color: '#E6EDF7' }, labelStyle: { color: '#A9B6CC' } }
const steps = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v))

export function RLTraining() {
  const runs = useResource<RLRun[]>('/api/rl/runs', 10000)
  const [sel, setSel] = useState<string | null>(null)
  const runId = sel ?? runs.data?.[0]?.id ?? null
  const run = runs.data?.find((r) => r.id === runId)
  const curves = useResource<Curves>(runId ? `/api/rl/runs/${runId}/curves` : null, run?.status === 'running' ? 15000 : undefined)

  if (runs.loading) return <Card title="RL training curves"><Loading /></Card>
  if (runs.error) return <Card title="RL training curves"><ErrorState error={runs.error} onRetry={runs.reload} /></Card>
  if (!runs.data?.length) {
    return (
      <Card title="RL training curves">
        <Empty title="No RL training runs yet">Start one with <code>python -m camp_crl.rl.batch</code> or <code>python -m camp_crl.rl.train --algo td3 --obs camp</code>.</Empty>
      </Card>
    )
  }
  const eps = curves.data?.episodes ?? []
  const val = curves.data?.validation ?? []
  return (
    <Card title="RL training curves" subtitle="Training reward on train scenarios; validation on validation scenarios only (fixed SUMO seeds)"
      actions={
        <Field label="Run">
          <select className={inputCls} value={runId ?? ''} onChange={(e) => setSel(e.target.value)}>
            {runs.data.map((r) => <option key={r.id} value={r.id}>{r.algo.toUpperCase()} {r.obs} · seed {r.seed} · {r.status}</option>)}
          </select>
        </Field>}>
      {run && (
        <p className="mb-3 flex flex-wrap items-center gap-2 text-xs text-ink-3">
          <Badge tone={TONE[run.status] ?? 'neutral'}>{run.status}</Badge>
          <span>target {steps(run.timesteps)} steps</span>
          {run.wall_time_s !== undefined && run.wall_time_s !== null && <span>· {(run.wall_time_s / 60).toFixed(0)} min</span>}
          {run.error && <span className="text-[#F08A8A]">· {run.error}</span>}
        </p>
      )}
      {curves.loading ? <Loading /> : curves.error ? <ErrorState error={curves.error} /> : !eps.length ? (
        <Empty title="No completed training episodes yet">Curves appear after the first full episode.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <div>
            <h3 className="mb-1 text-xs text-ink-2">Episode reward ({curves.data?.window}-episode moving average)</h3>
            <div style={{ height: 240 }} role="img" aria-label="Training episode reward over timesteps">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={eps} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="#2B3A55" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="timesteps" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={steps} />
                  <YAxis {...AXIS} width={48} />
                  <Tooltip {...TIP} labelFormatter={(v) => `${v} steps`} formatter={(v, n) => [Number(v).toFixed(3), n]} />
                  <Scatter dataKey="reward" name="episode" fill="#7F8DA6" shape="circle" isAnimationActive={false} />
                  <Line dataKey="reward_avg" name="moving average" stroke="#3987E5" strokeWidth={2} dot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div>
            <h3 className="mb-1 text-xs text-ink-2">Validation mean reward</h3>
            {val.length < 2 ? <p className="py-8 text-sm text-ink-3">{val.length ? 'One validation point so far.' : 'No validation yet.'}</p> : (
              <div style={{ height: 240 }} role="img" aria-label="Validation reward over timesteps">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={val} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#2B3A55" strokeDasharray="2 4" vertical={false} />
                    <XAxis dataKey="timesteps" type="number" domain={['dataMin', 'dataMax']} {...AXIS} tickFormatter={steps} />
                    <YAxis {...AXIS} width={48} />
                    <Tooltip {...TIP} labelFormatter={(v) => `${v} steps`} formatter={(v) => [Number(v).toFixed(3), 'validation reward']} />
                    <Line dataKey="mean_reward" stroke="#199E70" strokeWidth={2} dot={{ r: 4, stroke: '#172033', strokeWidth: 2 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            <Table head={['Steps', 'Validation reward', 'Validation mean delay (s)']}>
              {val.map((v) => (
                <tr key={v.timesteps} className="border-b border-line/50">
                  <td className="px-2 py-1">{steps(v.timesteps)}</td><td className="px-2 py-1">{v.mean_reward.toFixed(3)}</td><td className="px-2 py-1">{fmt(v.mean_delay_s, 2)}</td>
                </tr>
              ))}
            </Table>
          </div>
        </div>
      )}
    </Card>
  )
}
