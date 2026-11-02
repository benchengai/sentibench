"use client";

import type { ReactNode } from "react";
import { Tooltip, useTooltip } from "./Tooltip";

const STEPS = [1, 2, 3, 4, 5, 6, 7];

type Props = {
  rows: string[];
  cols: string[];
  values: (number | null)[][];
  domain: [number, number];
  corner?: string;
  showValues?: boolean;
  describe: (r: number, c: number, v: number) => ReactNode;
  label: (r: number, c: number, v: number) => string;
  highlightDiagonal?: boolean;
  compact?: boolean;
};

function step(v: number, [lo, hi]: [number, number]) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  return STEPS[Math.round(t * (STEPS.length - 1))];
}

// Sequential one-hue heatmap (light = low, dark = high). Cells are buttons so keyboard
// focus gives the same readout as hover.
export function Heatmap({ rows, cols, values, domain, corner, showValues = true, describe, label, highlightDiagonal, compact }: Props) {
  const { tip, show, hide } = useTooltip();
  const labelWidth = compact ? "minmax(70px, 120px)" : "minmax(90px, 130px)";
  return (
    <div>
      <div className="heat" style={{ gridTemplateColumns: `${labelWidth} repeat(${cols.length}, minmax(${compact ? 26 : 44}px, 1fr))` }}>
        <div className="h-lab">{corner}</div>
        {cols.map((c) => <div key={c} className="h-lab h-col">{c}</div>)}
        {rows.map((r, i) => (
          <div key={r} style={{ display: "contents" }}>
            <div className="h-lab">{r}</div>
            {cols.map((c, j) => {
              const v = values[i][j];
              if (v === null) return <div key={c} className="h-cell" style={{ background: "var(--surface-2)" }}>–</div>;
              const s = step(v, domain);
              const diag = highlightDiagonal && i === j;
              return (
                <button
                  key={c}
                  aria-label={label(i, j, v)}
                  className={`h-cell heat-s${s}`}
                  style={{
                    fontWeight: diag ? 700 : 400,
                    minHeight: compact ? 26 : 40,
                    fontSize: compact ? "0.7rem" : undefined,
                  }}
                  onPointerMove={(e) => show(e, describe(i, j, v))}
                  onPointerLeave={hide}
                  onFocus={(e) => show(e, describe(i, j, v))}
                  onBlur={hide}
                >
                  {showValues ? v.toFixed(2) : ""}
                </button>
              );
            })}
          </div>
        ))}
      </div>
      <div className="heat-scale" aria-hidden="true">
        <span>{domain[0].toFixed(1)}</span>
        {STEPS.map((s) => <i key={s} className={`heat-s${s}`} />)}
        <span>{domain[1].toFixed(1)}</span>
      </div>
      <Tooltip tip={tip} />
    </div>
  );
}
