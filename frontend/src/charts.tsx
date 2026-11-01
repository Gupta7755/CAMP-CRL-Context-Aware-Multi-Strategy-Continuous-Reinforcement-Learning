import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { APPROACHES, APPROACH_COLOR, APPROACH_LABEL, type HistoryPoint } from './api'

const AXIS = { stroke: '#7F8DA6', fontSize: 12, tickLine: false }
const GRID = <CartesianGrid stroke="#2B3A55" strokeDasharray="2 4" vertical={false} />
const TOOLTIP = {
  contentStyle: { background: '#0B1220', border: '1px solid #2B3A55', borderRadius: 6, fontSize: 12, color: '#E6EDF7' },
  labelStyle: { color: '#A9B6CC' },
  itemStyle: { color: '#E6EDF7' },
  cursor: { stroke: '#7F8DA6', strokeDasharray: '3 3' },
}
const LEGEND = { wrapperStyle: { fontSize: 12, color: '#A9B6CC' }, iconType: 'plainline' as const }
// Keep legend in N, E, S, W order (Recharts sorts by name otherwise).
const LEGEND_ORDER = APPROACHES.map((a) => APPROACH_LABEL[a])

/** One line per approach over simulation time. `prefix` picks the series, e.g. q_ (queue) or w_ (max wait). */
export function ApproachLines({ data, prefix, unit, height = 240 }: { data: HistoryPoint[]; prefix: string; unit: string; height?: number }) {
  return (
    <div style={{ height }} role="img" aria-label={`Per-approach ${unit} over simulation time`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          {GRID}
          <XAxis dataKey="t" {...AXIS} tickFormatter={(v) => `${v}s`} minTickGap={40} />
          <YAxis {...AXIS} width={40} allowDecimals={false} />
          <Tooltip {...TOOLTIP} labelFormatter={(v) => `t = ${v} s`} formatter={(v, name) => [`${Number(v).toFixed(prefix === 'q_' ? 0 : 2)} ${unit}`, name]} />
          <Legend {...LEGEND} itemSorter={(item) => LEGEND_ORDER.indexOf(String(item.value))} />
          {APPROACHES.map((a) => (
            <Line key={a} dataKey={`${prefix}${a}`} name={APPROACH_LABEL[a]} stroke={APPROACH_COLOR[a]} strokeWidth={2}
              dot={false} isAnimationActive={false} activeDot={{ r: 4, stroke: '#172033', strokeWidth: 2 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export interface BarDatum { name: string; value: number | null; err?: [number, number] }

/** One measure per controller as horizontal bars (names on the axis, so identity never relies on colour).
 *  All bars share one hue; the baseline is drawn in neutral grey. Whiskers: asymmetric 95% CI. */
export function EntityBars({ data, unit, highlight }: { data: BarDatum[]; unit: string; highlight?: string }) {
  const rows = data.map((d) => ({ ...d, errRange: d.err && d.value !== null ? [d.value - d.err[0], d.err[1] - d.value] : undefined }))
  const height = Math.max(160, rows.length * 34 + 40)
  return (
    <div style={{ height }} role="img" aria-label={`Bar chart of ${unit || 'metric'} per controller`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }} barCategoryGap="28%">
          <CartesianGrid stroke="#2B3A55" strokeDasharray="2 4" horizontal={false} />
          <XAxis type="number" {...AXIS} />
          <YAxis type="category" dataKey="name" {...AXIS} width={190} interval={0} />
          <Tooltip {...TOOLTIP} cursor={{ fill: '#1E293B' }}
            formatter={(v, _n, item) => {
              const e = (item?.payload as { err?: [number, number] })?.err
              return [`${Number(v).toFixed(2)} ${unit}${e ? `  (95% CI ${e[0].toFixed(2)}–${e[1].toFixed(2)})` : ''}`, 'mean']
            }} />
          <Bar dataKey="value" isAnimationActive={false}
            shape={(props: unknown) => {
              const p = props as { x: number; y: number; width: number; height: number; payload: BarDatum }
              const w = Math.max(0, p.width)
              const r = Math.min(4, w, p.height / 2)
              const fill = p.payload.name === highlight ? '#7F8DA6' : '#3987E5'
              return <path d={`M${p.x},${p.y} H${p.x + w - r} Q${p.x + w},${p.y} ${p.x + w},${p.y + r} V${p.y + p.height - r} Q${p.x + w},${p.y + p.height} ${p.x + w - r},${p.y + p.height} H${p.x} Z`} fill={fill} />
            }}>
            <ErrorBar dataKey="errRange" width={6} strokeWidth={1.5} stroke="#E6EDF7" direction="x" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
