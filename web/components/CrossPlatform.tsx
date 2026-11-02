"use client";

import { DATASETS, fmt, type Dataset, type Results } from "@/lib/results";
import { Heatmap } from "./Heatmap";

export function CrossPlatform({ cross, names }: { cross: Results["cross"]; names: Record<Dataset, string> }) {
  return (
    <div className="multiples">
      {Object.entries(cross).map(([family, m]) => {
        const values = DATASETS.map((src) => DATASETS.map((tgt) => m[src]?.[tgt]?.macro_f1 ?? null));
        return (
          <div key={family}>
            <h3>{family}</h3>
            <Heatmap
              rows={DATASETS.map((d) => names[d])}
              cols={DATASETS.map((d) => names[d])}
              corner="train ↓ test →"
              values={values}
              domain={[0.4, 0.9]}
              highlightDiagonal
              label={(i, j, v) => `${family}, trained on ${names[DATASETS[i]]}, tested on ${names[DATASETS[j]]}: macro-F1 ${fmt(v)}`}
              describe={(i, j, v) => (
                <div>
                  <strong>{fmt(v)}</strong> macro-F1
                  <div className="muted">train {names[DATASETS[i]]} → test {names[DATASETS[j]]}</div>
                  <div className="muted">κ {fmt(m[DATASETS[i]][DATASETS[j]].kappa)} · n = {m[DATASETS[i]][DATASETS[j]].n}</div>
                </div>
              )}
            />
          </div>
        );
      })}
    </div>
  );
}
