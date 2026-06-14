import { type ReactNode } from "react";

export interface Series {
  label: string;
  color: string;
  values: number[];
}

interface LineChartProps {
  series: Series[];
  height?: number;
  yDomain?: [number, number];
  thresholds?: { value: number; color: string; label?: string }[];
  band?: { from: number; to: number; color: string };
  cursor?: number;
  stops?: number[];
}

const W = 640;
const PAD = 28;

/** A small dependency-free SVG line chart over step indices. */
export function LineChart({
  series,
  height = 160,
  yDomain,
  thresholds = [],
  band,
  cursor,
  stops = [],
}: LineChartProps) {
  const n = Math.max(...series.map((s) => s.values.length), 1);
  const allVals = series.flatMap((s) => s.values).concat(thresholds.map((t) => t.value));
  const lo = yDomain ? yDomain[0] : Math.min(0, ...allVals);
  const hi = yDomain ? yDomain[1] : Math.max(1, ...allVals);
  const span = hi - lo || 1;

  const x = (i: number) => PAD + (i / Math.max(n - 1, 1)) * (W - 2 * PAD);
  const y = (v: number) => PAD + (1 - (v - lo) / span) * (height - 2 * PAD);

  return (
    <svg
      viewBox={`0 0 ${W} ${height}`}
      width="100%"
      preserveAspectRatio="none"
      role="img"
      className="chart"
    >
      {band && (
        <rect
          x={PAD}
          y={y(band.to)}
          width={W - 2 * PAD}
          height={Math.abs(y(band.from) - y(band.to))}
          fill={band.color}
          opacity={0.12}
        />
      )}
      {/* axes */}
      <line x1={PAD} y1={height - PAD} x2={W - PAD} y2={height - PAD} stroke="#2a3550" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={height - PAD} stroke="#2a3550" />
      {thresholds.map((t, i) => (
        <line
          key={i}
          x1={PAD}
          y1={y(t.value)}
          x2={W - PAD}
          y2={y(t.value)}
          stroke={t.color}
          strokeDasharray="4 4"
          opacity={0.7}
        />
      ))}
      {stops.map((s, i) => (
        <line key={`stop-${i}`} x1={x(s)} y1={PAD} x2={x(s)} y2={height - PAD} stroke="#ff5d6c" strokeWidth={2} opacity={0.8} />
      ))}
      {series.map((s, si) => (
        <polyline
          key={si}
          fill="none"
          stroke={s.color}
          strokeWidth={2}
          points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
        />
      ))}
      {cursor !== undefined && cursor >= 0 && (
        <line x1={x(cursor)} y1={PAD} x2={x(cursor)} y2={height - PAD} stroke="#9fb4ff" strokeWidth={1.5} opacity={0.9} />
      )}
    </svg>
  );
}

interface BarProps {
  value: number; // 0..1
  color?: string;
  height?: number;
  label?: ReactNode;
}

/** A 0..1 horizontal bar (intensity, scores). */
export function Bar({ value, color = "#5b8cff", height = 10, label }: BarProps) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="bar-wrap">
      {label && <div className="bar-label">{label}</div>}
      <div className="bar-track" style={{ height }}>
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

interface LegendProps {
  items: { label: string; color: string }[];
}
export function Legend({ items }: LegendProps) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          <span className="legend-swatch" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}
