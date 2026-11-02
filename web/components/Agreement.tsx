"use client";

import { useState } from "react";
import { fmt, type Classifier, type Results } from "@/lib/results";
import { Heatmap } from "./Heatmap";

export function Agreement({ agreement, classifiers }: { agreement: Results["agreement"]; classifiers: Classifier[] }) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const name = Object.fromEntries(classifiers.map((c) => [c.key, c.name]));
  const keys = agreement.keys;
  const cols = [...keys, "gold"];
  const values = keys.map((a) => cols.map((b) => agreement.kappa[a][b]));
  return (
    <div>
      <div className="seg" role="group" aria-label="View" style={{ marginBottom: 10 }}>
        <button aria-pressed={view === "chart"} onClick={() => setView("chart")}>Chart</button>
        <button aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
      </div>
      {view === "chart" ? (
        <div className="scroll-x">
          <div style={{ minWidth: 640 }}>
            <Heatmap
              rows={keys}
              cols={cols}
              values={values}
              domain={[0.2, 1.0]}
              showValues={false}
              compact
              label={(i, j, v) => `${name[keys[i]]} vs ${cols[j] === "gold" ? "gold" : name[cols[j]]}: kappa ${fmt(v)}`}
              describe={(i, j, v) => (
                <div>
                  <strong>κ {fmt(v)}</strong>
                  <div className="muted">{name[keys[i]]}</div>
                  <div className="muted">vs {cols[j] === "gold" ? "gold labels" : name[cols[j]]}</div>
                </div>
              )}
            />
          </div>
        </div>
      ) : (
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>κ</th>{cols.map((c) => <th key={c} className="num">{c}</th>)}</tr></thead>
            <tbody>
              {keys.map((a) => (
                <tr key={a}><td>{name[a]}</td>{cols.map((b) => <td key={b} className="num">{fmt(agreement.kappa[a][b])}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
