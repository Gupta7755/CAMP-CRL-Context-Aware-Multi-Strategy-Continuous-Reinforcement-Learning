import { NavLink, Route, Routes } from 'react-router-dom'
import { CONTROLLER_LABEL } from './api'
import CameraSetup from './pages/CameraSetup'
import Configuration from './pages/Configuration'
import Evaluation from './pages/Evaluation'
import ExperimentLab from './pages/ExperimentLab'
import Explainability from './pages/Explainability'
import LiveAdvisory from './pages/LiveAdvisory'
import LiveIntersection from './pages/LiveIntersection'
import Methodology from './pages/Methodology'
import Overview from './pages/Overview'
import SystemHealth from './pages/SystemHealth'
import { Icons, Logo } from './icons'
import { useTelemetry } from './telemetry'
import { Badge } from './ui'

export const APP_NAME = 'SignalSense AI'

export const PAGES = [
  { path: '/', label: 'Overview', group: 'Monitor', icon: 'overview', subtitle: 'Simulation status, approaches and live KPIs', element: <Overview /> },
  { path: '/live', label: 'Live Intersection', group: 'Monitor', icon: 'live', subtitle: 'Signal state, queues and what the safety filter applied', element: <LiveIntersection /> },
  { path: '/advisory', label: 'Live Advisory', group: 'Monitor', icon: 'advisory', subtitle: 'Signal-timing advice from your live cameras (advisory only)', element: <LiveAdvisory /> },
  { path: '/cameras', label: 'Cameras', group: 'Monitor', icon: 'cameras', subtitle: 'Connect a camera per direction and watch it live', element: <CameraSetup /> },
  { path: '/camp', label: 'CAMP Explainability', group: 'Analyze', icon: 'camp', subtitle: 'Pressure, momentum, risk, fairness and the chosen strategy', element: <Explainability /> },
  { path: '/lab', label: 'Experiment Lab', group: 'Analyze', icon: 'lab', subtitle: 'Run multi-seed experiments and watch RL training', element: <ExperimentLab /> },
  { path: '/evaluation', label: 'Evaluation', group: 'Analyze', icon: 'evaluation', subtitle: 'Controller comparison with uncertainty, seen vs held-out traffic', element: <Evaluation /> },
  { path: '/config', label: 'Configuration', group: 'System', icon: 'config', subtitle: 'Validated timing limits, CAMP weights and RL settings', element: <Configuration /> },
  { path: '/system', label: 'System Health', group: 'System', icon: 'health', subtitle: 'Backend, SUMO, cameras, latency and logs', element: <SystemHealth /> },
  { path: '/methodology', label: 'Methodology', group: 'System', icon: 'methodology', subtitle: 'How it works: the CAMP-CRL method, equations and limitations', element: <Methodology /> },
]
const GROUPS = ['Monitor', 'Analyze', 'System'] as const

function StatusLine() {
  const t = useTelemetry()
  const s = t.status
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <Badge tone={t.connected ? 'good' : 'crit'}>{t.connected ? 'Live link' : 'Offline'}</Badge>
      {s && s.status !== 'idle' && s.params && (
        <span className="text-ink-3 tabular">
          {CONTROLLER_LABEL[s.params.controller]} · {s.params.scenario} · {s.status}
          {s.sim_time_s !== null && ` · t=${s.sim_time_s.toFixed(0)} s`}
        </span>
      )}
      {t.stale && <Badge tone="warn">Stale</Badge>}
    </div>
  )
}

export default function App() {
  return (
    <div className="flex min-h-full flex-col md:flex-row">
      <nav aria-label="Main" className="border-b border-line/70 bg-bg-2/80 backdrop-blur md:sticky md:top-0 md:h-screen md:w-60 md:shrink-0 md:overflow-y-auto md:border-b-0 md:border-r">
        <div className="flex items-center gap-3 px-4 py-4">
          <Logo />
          <div>
            <div className="text-[15px] font-bold tracking-tight"><span className="text-ink">SignalSense</span> <span className="text-gradient">AI</span></div>
            <div className="text-[11px] leading-tight text-ink-3">Context-aware adaptive signals</div>
          </div>
        </div>
        <div className="accent-rule mx-4 mb-2 opacity-60" />
        <div className="flex gap-3 overflow-x-auto px-2 pb-2 md:block md:overflow-visible">
          {GROUPS.map((g) => (
            <div key={g} className="shrink-0 md:mb-3">
              <div className="hidden px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3 md:block">{g}</div>
              <ul className="flex gap-1 md:flex-col">
                {PAGES.filter((p) => p.group === g).map((p) => (
                  <li key={p.path} className="shrink-0">
                    <NavLink to={p.path} end={p.path === '/'}
                      className={({ isActive }) => `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${isActive ? 'nav-active font-medium text-ink' : 'text-ink-2 hover:bg-surface/70 hover:text-ink'}`}>
                      <span className="text-[#67E3F5]/80">{Icons[p.icon]}</span>{p.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="hidden px-4 py-4 text-[11px] leading-snug text-ink-3 md:block">
          Simulation &amp; advisory only. Never connected to real signals.
          <div className="mt-1 opacity-80">Method: CAMP-CRL</div>
        </div>
      </nav>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-2 border-b border-line/70 bg-bg/75 px-4 py-3 backdrop-blur md:px-6">
          <Routes>{PAGES.map((p) => <Route key={p.path} path={p.path} element={
            <div>
              <h1 className="text-lg font-semibold tracking-tight">{p.label}</h1>
              <p className="text-xs text-ink-3">{p.subtitle}</p>
            </div>} />)}</Routes>
          <StatusLine />
        </header>
        <main className="min-w-0 flex-1 px-4 py-4 md:px-6">
          <Routes>
            {PAGES.map((p) => <Route key={p.path} path={p.path} element={p.element} />)}
            <Route path="*" element={<p className="text-sm text-ink-3">Page not found.</p>} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
