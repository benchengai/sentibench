"use client";

import { useState } from "react";
import { DATASETS, GROUPS, fmt, groupOf, mean, type Classifier, type Dataset, type Scores } from "@/lib/results";
import { Tooltip, useTooltip } from "./Tooltip";

type Props = {
  classifiers: Classifier[];
  scores: Record<string, Partial<Record<Dataset, Scores>>>;
  names: Record<Dataset, string>;
};

const X_MIN = 0.4;
const X_MAX = 1.0;
const GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const TICKS = [0.4, 0.6, 0.8, 1.0];
const pos = (v: number) => `${(100 * (Math.max(X_MIN, Math.min(X_MAX, v)) - X_MIN)) / (X_MAX - X_MIN)}%`;

// Dot-and-whisker small multiples: macro-F1 with its 95 % bootstrap interval, one column per
// dataset, classifiers in rows sorted by mean macro-F1. HTML grid rather than SVG so the
// layout reflows; on narrow screens one dataset is shown at a time.
export function DotPlot({ classifiers, scores, names }: Props) {
  const { tip, show, hide } = useTooltip();
  const [view, setView] = useState<"chart" | "table">("chart");
  const [focusDs, setFocusDs] = useState<Dataset>("github");
  const rows = classifiers
    .filter((c) => DATASETS.every((ds) => scores[c.key]?.[ds]))
    .map((c) => ({ c, m: mean(DATASETS.map((ds) => scores[c.key][ds]!.macro_f1)) }))
    .sort((a, b) => b.m - a.m);

  return (
    <div>
      <div className="dot-controls">
        <div className="legend" aria-label="Legend">
          {GROUPS.map((g) => (
            <span key={g.id}><i className="swatch" style={{ background: g.color }} />{g.label}</span>
          ))}
          <span><i className="whisker-key" />95% bootstrap interval</span>
        </div>
        <div className="seg" role="group" aria-label="View">
          <button aria-pressed={view === "chart"} onClick={() => setView("chart")}>Chart</button>
          <button aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
        </div>
      </div>
      {view === "chart" ? (
        <>
          <div className="seg ds-switch" role="group" aria-label="Dataset">
            {DATASETS.map((ds) => (
              <button key={ds} aria-pressed={focusDs === ds} onClick={() => setFocusDs(ds)}>{names[ds]}</button>
            ))}
          </div>
          <div className="dotplot" style={{ ["--cols" as string]: DATASETS.length }}>
            <div className="dp-label dp-head" />
            {DATASETS.map((ds) => (
              <div key={ds} className={`dp-track dp-head ${ds === focusDs ? "focus" : ""}`}>
                <div className="dp-title">{names[ds]}</div>
                <div className="dp-axis">
                  {TICKS.map((t) => (
                    <span key={t} style={{ left: pos(t) }}>{t.toFixed(1)}</span>
                  ))}
                </div>
              </div>
            ))}
            {rows.map(({ c }) => {
              const g = groupOf(c);
              return (
                <div className="dp-row" key={c.key} style={{ display: "contents" }}>
                  <div className="dp-label">{c.name}</div>
                  {DATASETS.map((ds) => {
                    const s = scores[c.key][ds]!;
                    const content = (
                      <div>
                        <strong>{fmt(s.macro_f1)}</strong> macro-F1 <span className="muted">[{fmt(s.ci.macro_f1[0])}, {fmt(s.ci.macro_f1[1])}]</span>
                        <div className="muted">{c.name} · {names[ds]}</div>
                        <div className="muted">κ {fmt(s.kappa)} · n = {s.n}{s.failed ? ` · ${s.failed} failed` : ""}</div>
                      </div>
                    );
                    return (
                      <div key={ds} className={`dp-track ${ds === focusDs ? "focus" : ""}`}>
                        {GRID.map((t) => <i key={t} className="dp-grid" style={{ left: pos(t) }} />)}
                        <button
                          className="dp-hit"
                          aria-label={`${c.name}, ${names[ds]}: macro-F1 ${fmt(s.macro_f1)}, 95% interval ${fmt(s.ci.macro_f1[0])} to ${fmt(s.ci.macro_f1[1])}`}
                          style={{ left: `calc(${pos(s.ci.macro_f1[0])} - 10px)`, width: `calc(${pos(s.ci.macro_f1[1])} - ${pos(s.ci.macro_f1[0])} + 20px)` }}
                          onPointerMove={(e) => show(e, content)}
                          onPointerLeave={hide}
                          onFocus={(e) => show(e, content)}
                          onBlur={hide}
                        >
                          <i className="dp-whisker" style={{ background: g.color }} />
                        </button>
                        <i className="dp-dot" style={{ left: pos(s.macro_f1), background: g.color }} />
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <div className="scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th>Classifier</th>
                {DATASETS.map((ds) => <th key={ds} className="num">{names[ds]} macro-F1 [95% CI]</th>)}
                {DATASETS.map((ds) => <th key={ds} className="num">{names[ds]} κ</th>)}
                <th className="num">Mean</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ c, m }) => (
                <tr key={c.key}>
                  <td>{c.name}</td>
                  {DATASETS.map((ds) => {
                    const s = scores[c.key][ds]!;
                    return <td key={ds} className="num">{fmt(s.macro_f1)} [{fmt(s.ci.macro_f1[0])}, {fmt(s.ci.macro_f1[1])}]</td>;
                  })}
                  {DATASETS.map((ds) => <td key={ds} className="num">{fmt(scores[c.key][ds]!.kappa)}</td>)}
                  <td className="num">{fmt(m)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Tooltip tip={tip} />
    </div>
  );
}
