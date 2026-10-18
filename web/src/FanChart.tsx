import * as d3 from "d3";
import { bands, formatValue, type Fan } from "./lib";

const W = 520;
const H = 240;
const M = { top: 12, right: 16, bottom: 30, left: 64 };

type Series = { label: string; fan: Fan; color: string };

/** Fan chart: 5-95% and 25-75% bands with the median, one or more series. */
export function FanChart({ t, series, unit, title }: { t: number[]; series: Series[]; unit: string; title: string }) {
  const all = series.flatMap((s) => [...s.fan.p05, ...s.fan.p95]);
  const x = d3.scaleLinear().domain(d3.extent(t) as [number, number]).range([M.left, W - M.right]);
  const y = d3.scaleLinear().domain(d3.extent(all) as [number, number]).nice().range([H - M.bottom, M.top]);
  const area = d3.area<{ t: number; lo: number; hi: number }>().x((d) => x(d.t)).y0((d) => y(d.lo)).y1((d) => y(d.hi));
  const line = d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v));

  return (
    <figure className="fan">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {y.ticks(5).map((v) => (
          <g key={v} transform={`translate(0,${y(v)})`}>
            <line x1={M.left} x2={W - M.right} className="grid" />
            <text x={M.left - 6} dy={4} textAnchor="end">{formatValue(v, unit)}</text>
          </g>
        ))}
        {x.ticks(6).map((v) => (
          <text key={v} x={x(v)} y={H - 10} textAnchor="middle">{v}</text>
        ))}
        {series.map((s) => {
          const b = bands(t, s.fan);
          return (
            <g key={s.label}>
              <path d={area(b.outer)!} fill={s.color} opacity={0.15} />
              <path d={area(b.inner)!} fill={s.color} opacity={0.3} />
              <path d={line(b.median)!} stroke={s.color} fill="none" strokeWidth={2} />
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.label}><i style={{ background: s.color }} />{s.label}</span>
        ))}
        <span className="muted">bands: 50% and 90% of runs</span>
      </div>
    </figure>
  );
}
