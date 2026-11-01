import { ctrlLabel, fmt, type ExperimentManifest, type Summary } from '../api'
import { ResultsView } from '../Results'
import { Banner, Card, Empty, ErrorState, Loading, Table, useResource } from '../ui'

const SPLIT_TITLE: Record<string, string> = { train: 'Seen: train', validation: 'Seen: validation', test: 'Unseen: held-out test' }
const KEY_METRICS = ['delay_mean_s', 'delay_p95_s', 'queue_mean_veh', 'max_wait_s', 'approach_delay_std_s', 'throughput_completed', 'phase_switches']
const SHORT: Record<string, string> = {
  delay_mean_s: 'Mean delay', delay_p95_s: 'p95 delay', queue_mean_veh: 'Mean queue', max_wait_s: 'Max wait',
  approach_delay_std_s: 'Fairness (delay std)', throughput_completed: 'Throughput', phase_switches: 'Switches',
}

function RLCard({ summary }: { summary: Summary }) {
  const spread = summary.training_seed_spread ?? {}
  const alpha = summary.fusion_alpha ?? {}
  const scenarios = Object.keys(summary.scenarios)
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <Card title="RL: variability across training seeds" subtitle="Mean delay over all scenarios, per independently trained model">
        <Table head={['Controller', 'Per training seed (s)', 'Std across seeds']}>
          {Object.entries(spread).map(([c, s]) => (
            <tr key={c} className="border-b border-line/50">
              <td className="px-2 py-1.5">{ctrlLabel(c)}</td>
              <td className="px-2 py-1.5">{Object.entries(s.delay_mean_s_by_training_seed).map(([k, v]) => `s${k}: ${v.toFixed(1)}`).join('  ·  ')}</td>
              <td className="px-2 py-1.5">{fmt(s.std_across_training_seeds, 2)}</td>
            </tr>
          ))}
        </Table>
      </Card>
      <Card title="Fusion gate: mean α by scenario" subtitle="α = share of the executed action taken from RL (0 = CAMP prior only, 1 = RL only)">
        <Table head={['Scenario', ...Object.keys(alpha).map(ctrlLabel)]}>
          {scenarios.map((sc) => (
            <tr key={sc} className="border-b border-line/50">
              <td className="px-2 py-1.5">{sc}</td>
              {Object.keys(alpha).map((c) => <td key={c} className="px-2 py-1.5">{fmt(alpha[c]?.[sc], 2)}</td>)}
            </tr>
          ))}
        </Table>
      </Card>
    </div>
  )
}

export default function Evaluation() {
  const ev = useResource<{ manifest: ExperimentManifest; summary: Summary }>('/api/evaluation/latest')
  if (ev.loading) return <Loading />
  if (ev.status === 404) {
    return <Empty title="No evaluation has been run yet">Run <code>python -m camp_crl.cli evaluate --splits train validation</code> (and, once, with <code>--splits test --allow-test</code>).</Empty>
  }
  if (ev.error || !ev.data) return <ErrorState error={ev.error ?? 'unknown error'} onRetry={ev.reload} />
  const { manifest, summary } = ev.data
  const ctrls = summary.controllers
  return (
    <div className="space-y-4">
      <Banner tone="info" title="How to read this">
        Experiment {manifest.id}: {manifest.params.seeds.length} seeds per scenario, finished {manifest.finished?.replace('T', ' ')}.
        CAMP uses untuned default weights. Results are evidence about this simulated intersection only; no claim of general superiority is made.
      </Banner>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {Object.entries(summary.by_split).map(([split, data]) => (
          <Card key={split} title={SPLIT_TITLE[split] ?? split} subtitle={`Unweighted mean over ${data._scenarios.length} scenario(s): ${data._scenarios.join(', ')}`}>
            <Table head={['Controller', ...KEY_METRICS.slice(0, 3).map((m) => SHORT[m])]}>
              {ctrls.map((c) => (
                <tr key={c} className="border-b border-line/50">
                  <td className="px-2 py-1.5">{ctrlLabel(c)}</td>
                  {KEY_METRICS.slice(0, 3).map((m) => <td key={m} className="px-2 py-1.5">{fmt((data as Record<string, Record<string, number | null>>)[c]?.[m], 2)}</td>)}
                </tr>
              ))}
            </Table>
          </Card>
        ))}
      </div>
      {!summary.by_split.test && (
        <Banner tone="warn" title="Unseen scenarios not evaluated in this experiment">The held-out test split has not been included; generalisation is unknown.</Banner>
      )}
      {(summary.fusion_alpha || summary.training_seed_spread) && <RLCard summary={summary} />}
      <Card title="Detail by scenario and metric">
        <ResultsView summary={summary} />
      </Card>
      <Card title="Notes">
        <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">{summary.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      </Card>
    </div>
  )
}
