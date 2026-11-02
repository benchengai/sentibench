"use client";

import { useEffect, useRef, useState } from "react";
import { GROUPS, fmt, groupOf, type Classifier } from "@/lib/results";
import { Tooltip, useTooltip } from "./Tooltip";

export type CostPoint = { c: Classifier; cost: number; f1: number; latency: string };

const H = 300;
const PAD = { l: 44, r: 16, t: 14, b: 40 };
const FREE_LANE = 96; // px reserved at the left for classifiers with no per-item price
const LANE_GAP = 28;

const SHORT: Record<string, string> = {
  vader: "VADER", "vader-tuned": "VADER tuned", roberta: "RoBERTa", "tfidf-lr": "TF-IDF", "embed-lr": "Embeddings + LR",
};
const shortName = (key: string) =>
  SHORT[key] ?? key.replace("gpt-6.1-", "").replace("gpt-6-", "").replace("qwen3:8b", "qwen").replace(/-(zs|fs)$/, " $1");

type Box = { x1: number; x2: number; y1: number; y2: number };
const overlaps = (a: Box, b: Box) => a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;

// Quality against price: mean macro-F1 on the LLM test sample (y) against USD per 1,000 items
// on a log scale (x). Local classifiers have no per-item price and sit in their own lane.
export function CostScatter({ points }: { points: CostPoint[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const { tip, show, hide } = useTooltip();
  const [view, setView] = useState<"chart" | "table">("chart");

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(300, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const paid = points.filter((p) => p.cost > 0);
  const lo = Math.floor(Math.log10(Math.min(...paid.map((p) => p.cost))));
  const hi = Math.ceil(Math.log10(Math.max(...paid.map((p) => p.cost))));
  const yMin = Math.floor(Math.min(...points.map((p) => p.f1)) * 10) / 10;
  const yMax = 1.0;
  const laneEnd = PAD.l + FREE_LANE;
  const x0 = laneEnd + LANE_GAP;
  const x = (cost: number) =>
    cost > 0 ? x0 + ((Math.log10(cost) - lo) / (hi - lo)) * (width - x0 - PAD.r) : PAD.l + FREE_LANE / 2;
  const y = (f1: number) => PAD.t + (1 - (f1 - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const yTicks = Array.from({ length: Math.round((yMax - yMin) * 10) + 1 }, (_, i) => +(yMin + i / 10).toFixed(1));
  const xTicks = Array.from({ length: hi - lo + 1 }, (_, i) => 10 ** (lo + i));

  // Free classifiers share one x position; spread them a little so dots do not sit on top of each other.
  const free = points.filter((p) => p.cost === 0).sort((a, b) => a.f1 - b.f1);
  const jitter = new Map(free.map((p, i) => [p.c.key, (i % 2 === 0 ? -1 : 1) * 9]));

  // Direct labels where they fit: right of the dot, else left, else none (tooltip and table carry it).
  const dots: Box[] = points.map((p) => {
    const cx = x(p.cost) + (jitter.get(p.c.key) ?? 0);
    return { x1: cx - 6, x2: cx + 6, y1: y(p.f1) - 6, y2: y(p.f1) + 6 };
  });
  const placed: Box[] = [];
  const labels = new Map<string, { x: number; y: number; anchor: "start" | "end" }>();
  [...points].sort((a, b) => b.f1 - a.f1).forEach((p) => {
    const cx = x(p.cost) + (jitter.get(p.c.key) ?? 0);
    const cy = y(p.f1);
    const w = shortName(p.c.key).length * 6.4;
    const options: { box: Box; x: number; anchor: "start" | "end" }[] = [
      { box: { x1: cx + 9, x2: cx + 9 + w, y1: cy - 7, y2: cy + 7 }, x: cx + 9, anchor: "start" },
      { box: { x1: cx - 9 - w, x2: cx - 9, y1: cy - 7, y2: cy + 7 }, x: cx - 9, anchor: "end" },
    ];
    const ok = options.find((o) => o.box.x1 >= 2 && o.box.x2 <= width - 2 &&
      !placed.some((b) => overlaps(b, o.box)) && !dots.some((d) => overlaps(d, o.box)));
    if (ok) {
      placed.push(ok.box);
      labels.set(p.c.key, { x: ok.x, y: cy + 4, anchor: ok.anchor });
    }
  });

  return (
    <div>
      <div className="dot-controls">
        <div className="legend">
          {GROUPS.map((g) => <span key={g.id}><i className="swatch" style={{ background: g.color }} />{g.label}</span>)}
        </div>
        <div className="seg" role="group" aria-label="View">
          <button aria-pressed={view === "chart"} onClick={() => setView("chart")}>Chart</button>
          <button aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
        </div>
      </div>
      {view === "chart" ? (
        <div ref={ref} style={{ width: "100%" }}>
          <svg width={width} height={H} role="img" aria-label="Mean macro-F1 against cost per 1,000 items">
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={PAD.l} x2={width - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                <text x={PAD.l - 8} y={y(t) + 4} textAnchor="end" fontSize="11" fill="var(--text-muted)">{t.toFixed(1)}</text>
              </g>
            ))}
            <line x1={laneEnd + LANE_GAP / 2} x2={laneEnd + LANE_GAP / 2} y1={PAD.t} y2={H - PAD.b} stroke="var(--axis)" />
            <text x={PAD.l + FREE_LANE / 2} y={H - PAD.b + 16} textAnchor="middle" fontSize="11" fill="var(--text-muted)">local, $0</text>
            {xTicks.map((t) => (
              <text key={t} x={x(t)} y={H - PAD.b + 16} textAnchor="middle" fontSize="11" fill="var(--text-muted)">
                ${t >= 1 ? t.toFixed(0) : t.toPrecision(1)}
              </text>
            ))}
            <text x={(x0 + width - PAD.r) / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--text-secondary)">
              USD per 1,000 items (log scale)
            </text>
            <text x={12} y={PAD.t + 4} fontSize="11" fill="var(--text-secondary)" transform={`rotate(-90 12 ${PAD.t + 4})`} textAnchor="end">
              mean macro-F1
            </text>
            {points.map((p) => {
              const cx = x(p.cost) + (jitter.get(p.c.key) ?? 0);
              const cy = y(p.f1);
              const content = (
                <div>
                  <strong>{fmt(p.f1)}</strong> mean macro-F1
                  <div className="muted">{p.c.name}</div>
                  <div className="muted">{p.cost > 0 ? `$${p.cost.toFixed(3)} per 1,000 items` : "no per-item price (runs locally)"} · {p.latency}</div>
                </div>
              );
              return (
                <g key={p.c.key} tabIndex={0} role="button" aria-label={`${p.c.name}: mean macro-F1 ${fmt(p.f1)}`}
                   onPointerMove={(e) => show(e, content)} onPointerLeave={hide}
                   onFocus={(e) => show(e, content)} onBlur={hide}>
                  <circle cx={cx} cy={cy} r={12} fill="transparent" />
                  <circle cx={cx} cy={cy} r={5} fill={groupOf(p.c).color} stroke="var(--surface-1)" strokeWidth={2} />
                  {labels.has(p.c.key) && (
                    <text x={labels.get(p.c.key)!.x} y={labels.get(p.c.key)!.y} textAnchor={labels.get(p.c.key)!.anchor}
                          fontSize="11" fill="var(--text-secondary)">{shortName(p.c.key)}</text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      ) : (
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>Classifier</th><th className="num">Mean macro-F1</th><th className="num">USD per 1,000 items</th><th className="num">Latency</th></tr></thead>
            <tbody>
              {[...points].sort((a, b) => b.f1 - a.f1).map((p) => (
                <tr key={p.c.key}><td>{p.c.name}</td><td className="num">{fmt(p.f1)}</td>
                  <td className="num">{p.cost > 0 ? `$${p.cost.toFixed(3)}` : "0 (local)"}</td><td className="num">{p.latency}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Tooltip tip={tip} />
    </div>
  );
}
