import { Badge, Card, Table } from '../ui'

const STAGES = ['SUMO + TraCI (ground truth)', 'Sensor model (noise / dropout)', 'Traffic state', 'CAMP context engine', 'Strategy selection', 'Controller request', 'Safety filter', 'Signal state → SUMO', 'Metrics + experiment store → UI']

const STATUS: [string, 'done' | 'partial' | 'not'][] = [
  ['Four-arm SUMO intersection, 11 scenarios (train / validation / held-out)', 'done'],
  ['TraCI state collector, sensor degradation model', 'done'],
  ['Fixed-Time (untuned + tuned), Actuated, Max-Pressure baselines', 'done'],
  ['CAMP: pressure, momentum, risk, fairness, composite', 'done'],
  ['CAMP: trend, pattern, CUSUM anomaly, data-confidence score (explanation only)', 'done'],
  ['Rule-based strategy selection with hysteresis', 'done'],
  ['Independent safety filter with logged interventions', 'done'],
  ['Multi-seed evaluation with bootstrap CIs; FastAPI backend; this dashboard', 'done'],
  ['Protected left-turn phases, pedestrians', 'not'],
  ['Phase 2: Gymnasium environment, reward, DDPG / TD3 training pipeline (Stable-Baselines3)', 'done'],
  ['Phase 2: CAMP action prior and adaptive fusion gate (deployment-time), with ablation modes', 'done'],
  ['Phase 2: trained policies (3 configurations x 3 seeds) and their evaluation', 'partial'],
  ['Phase 2: training-time fusion, learned gate weights, SAC', 'not'],
  ['Phase 3: camera adapters (phone stream / webcam / video file / synthetic), YOLOX detector (Apache-2.0), ByteTrack-style tracker', 'done'],
  ['Phase 3: approach features, health (stale / low fps / obstructed / disconnected), reconnect with backoff, camera -> SUMO demand with fallback', 'done'],
  ['Phase 3: repeatable offline demo with scripted faults; overlays; this System Health panel', 'done'],
  ['Phase 3: camera-driven demand in live dashboard runs (synthetic: simulation clock; phone streams: wall clock, real time)', 'done'],
  ['Phase 3: Cameras page: paste a camera URL per direction, test it, live view with detections', 'done'],
  ['Phase 3: tested with a real phone stream; counting validated on hand-labelled real footage', 'partial'],
]

const GLOSSARY: [string, string][] = [
  ['Q', 'Halting vehicles on an approach (speed < 0.1 m/s), veh'],
  ['A', 'EWMA-smoothed arrival rate onto the approach, veh/h'],
  ['W', 'Longest accumulated stopped time of a vehicle on the approach, s'],
  ['D', 'Approach edge occupancy, %'],
  ['R', 'Estimated queue length / approach storage length (spillback ratio)'],
  ['x̂', 'Normalised value, clip((x − min)/(max − min), 0, 1); NaN or zero range → safe value 0'],
  ['Delay', 'SUMO timeLoss per completed trip (time lost vs. driving at desired speed), s'],
  ['Warm-up', 'First 300 s; trips departing earlier are excluded from metrics'],
]

const LIMITS = [
  'One isolated four-arm intersection with a two-phase plan and permissive left turns; results may not transfer to networks or other phase plans.',
  'All weights, thresholds and bounds are untuned starting defaults; the Fixed-Time plan is tuned on training scenarios only.',
  'Risk F is a linear extrapolation heuristic, not a validated forecast. The CUSUM detector has not been validated for accuracy.',
  'The confidence value is a data-quality score, not a calibrated probability.',
  'Delay statistics include completed trips only; read them together with the unfinished count.',
  'Confidence intervals are percentile bootstrap over seeds, without multiple-comparison correction.',
  'No claim of novelty or general superiority is made; that requires broader experiments and a systematic literature review.',
]

const REFS = [
  'P. A. Lopez et al., “Microscopic Traffic Simulation using SUMO,” IEEE ITSC, 2018.',
  'P. Varaiya, “Max pressure control of a network of signalized intersections,” Transportation Research Part C, vol. 36, pp. 177–195, 2013.',
  'F. V. Webster, “Traffic Signal Settings,” Road Research Technical Paper No. 39, HMSO, London, 1958.',
  'E. S. Page, “Continuous Inspection Schemes,” Biometrika, vol. 41, pp. 100–115, 1954.',
  'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993.',
  'T. P. Lillicrap et al., “Continuous control with deep reinforcement learning,” ICLR, 2016 (DDPG; Phase 2).',
  'S. Fujimoto, H. van Hoof, D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” ICML, 2018 (TD3; Phase 2).',
]

function Eq({ children }: { children: string }) {
  return <pre className="overflow-x-auto rounded bg-bg-2 px-3 py-2 font-mono text-sm text-ink">{children}</pre>
}

export default function Methodology() {
  return (
    <div className="max-w-5xl space-y-4">
      <Card title="SignalSense AI and CAMP-CRL">
        <p className="text-sm text-ink-2">
          <span className="font-semibold text-ink">SignalSense AI</span> is this application: simulation, camera perception and live timing advice.
          The method inside it is <span className="text-gradient font-semibold">CAMP-CRL</span>, Context-Aware Multi-Strategy Continuous
          Reinforcement Learning: CAMP context features and strategy selection, a continuous RL policy, adaptive fusion and an independent
          safety filter. It is a research hypothesis under evaluation, not a proven improvement.
        </p>
      </Card>
      <Card title="Architecture" subtitle="Each step's output feeds the next; the safety filter is the only component that sets signals">
        <ol className="flex flex-wrap items-center gap-2 text-sm">
          {STAGES.map((s, i) => (
            <li key={s} className="flex items-center gap-2">
              <span className={`rounded-md border px-2.5 py-1 ${s.startsWith('Safety') ? 'border-accent text-ink' : 'border-line bg-surface-2 text-ink-2'}`}>{s}</span>
              {i < STAGES.length - 1 && <span aria-hidden className="text-ink-3">→</span>}
            </li>
          ))}
        </ol>
      </Card>
      <Card title="CAMP equations (as implemented)" subtitle="Proposed heuristics under evaluation, not validated innovations">
        <div className="space-y-2">
          <Eq>{'P_i = 0.35 Q̂ + 0.15 Â + 0.20 Ŵ + 0.15 D̂ + 0.15 R̂            (weights sum to 1 → P ∈ [0,1])'}</Eq>
          <Eq>{'M_i = 0.5 ĝ_Q + 0.3 ĝ_A − 0.2 ĝ_V    ĝ = clip(EWMA(Δx/Δt) / scale, −1, 1)'}</Eq>
          <Eq>{'F_i = clip(P_i + τ·k_M·M_i, 0, 1)    τ = 10 s, k_M = 0.03 s⁻¹    (heuristic)'}</Eq>
          <Eq>{'S_i = clip(W_i / 90 s, 0, 1)'}</Eq>
          <Eq>{'C_i = P_i + 0.3 M_i + 0.3 F_i + 0.5 S_i                       (ranking score)'}</Eq>
          <Eq>{'CUSUM: S_t = max(0, S_{t−1} + (A_t − μ_t)/150 − 0.5); alarm when S_t ≥ 5'}</Eq>
          <Eq>{'Max-Pressure: phase weight = Σ_served (vehicles on approach − vehicles on its through exit)'}</Eq>
          <Eq>{'improvement % = 100·(baseline − candidate)/|baseline|   (lower-is-better metrics)'}</Eq>
          <Eq>{'Fusion: a_mix = (1 − α)·a_CAMP + α·a_RL,   α = sigmoid(w·u + b),  w ≤ 0,  u = [1 − confidence, anomaly, novelty, critic disagreement]'}</Eq>
          <Eq>{'RL reward per decision = 0.1·Σ_seconds(−2·q + 0.5·thr − 0.5·w_max − 0.25·f − spill) − 0.1·switch − 0.5·interventions'}</Eq>
        </div>
      </Card>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Variable glossary">
          <Table head={['Symbol', 'Meaning and unit']}>{GLOSSARY.map(([k, v]) => <tr key={k} className="border-b border-line/50"><td className="px-2 py-1.5 font-mono">{k}</td><td className="px-2 py-1.5 text-ink-2">{v}</td></tr>)}</Table>
        </Card>
        <Card title="Experiment protocol">
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            <li>Every controller runs identical demand for each (scenario, seed); SUMO seed drives arrivals.</li>
            <li>Scenarios are split into train (tuning), validation (selection) and held-out test (final report only).</li>
            <li>1800 s episodes, 300 s warm-up, 1 s control step, 10 seeds per cell.</li>
            <li>Paired per-seed differences against the tuned Fixed-Time baseline with bootstrap 95 % CIs.</li>
            <li>Raw tripinfo, per-step logs, configuration and software versions are saved for every run.</li>
          </ul>
        </Card>
      </div>
      <Card title="Implementation status">
        <ul className="space-y-1.5 text-sm">
          {STATUS.map(([s, st]) => (
            <li key={s} className="flex items-center gap-2">
              {st === 'done' ? <Badge tone="good">Done</Badge> : st === 'partial' ? <Badge tone="warn">Partial</Badge> : <Badge tone="neutral">Not started</Badge>}
              <span className="text-ink-2">{s}</span>
            </li>
          ))}
        </ul>
      </Card>
      <Card title="Limitations">
        <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul>
      </Card>
      <Card title="References" subtitle="Methods used or planned; not a literature review">
        <ol className="list-decimal space-y-1 pl-5 text-sm text-ink-2">{REFS.map((r) => <li key={r}>{r}</li>)}</ol>
      </Card>
    </div>
  )
}
