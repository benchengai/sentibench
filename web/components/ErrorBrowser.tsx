"use client";

import { useEffect, useMemo, useState } from "react";
import type { Classifier, Dataset } from "@/lib/results";

type Item = {
  id: string;
  dataset: Dataset;
  text: string;
  gold: string;
  tags: string[];
  vader_compound: number;
  preds: Record<string, string | null>;
  all_strong_disagree_with_gold: boolean;
};

const PAGE = 30;
const TAG_LABELS: Record<string, string> = {
  technical: "Technical vocabulary",
  politeness: "Politeness",
  sarcasm: "Sarcasm candidate",
};

export function ErrorBrowser({ classifiers, names }: { classifiers: Classifier[]; names: Record<Dataset, string> }) {
  const [items, setItems] = useState<Item[] | null>(null);
  const [keys, setKeys] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [ds, setDs] = useState<string>("all");
  const [tag, setTag] = useState<string>("all");
  const [clf, setClf] = useState<string>("gpt-6.1-sol-fs");
  const [wrongOnly, setWrongOnly] = useState(true);
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);
  const name = Object.fromEntries(classifiers.map((c) => [c.key, c.name]));

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_BASE_PATH}/data/errors.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => {
        setItems(d.items);
        setKeys(d.keys);
        if (!d.keys.includes("gpt-6.1-sol-fs")) setClf(d.keys[0]);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const filtered = useMemo(() => {
    if (!items) return [];
    const q = query.trim().toLowerCase();
    return items.filter((it) =>
      (ds === "all" || it.dataset === ds) &&
      (tag === "all" || (tag === "noise" ? it.all_strong_disagree_with_gold : tag === "split" ? new Set(Object.values(it.preds)).size >= 3 : it.tags.includes(tag))) &&
      (!wrongOnly || it.preds[clf] !== it.gold) &&
      (!q || it.text.toLowerCase().includes(q)),
    );
  }, [items, ds, tag, clf, wrongOnly, query]);

  useEffect(() => setShown(PAGE), [ds, tag, clf, wrongOnly, query]);

  if (error) return <p>Could not load the error data ({error}).</p>;
  if (!items) return <p className="note">Loading items…</p>;

  return (
    <div>
      <div className="filters" role="search">
        <label>Dataset
          <select value={ds} onChange={(e) => setDs(e.target.value)}>
            <option value="all">All three</option>
            {(Object.keys(names) as Dataset[]).map((d) => <option key={d} value={d}>{names[d]}</option>)}
          </select>
        </label>
        <label>Case
          <select value={tag} onChange={(e) => setTag(e.target.value)}>
            <option value="all">All items</option>
            <option value="technical">Technical vocabulary, gold not negative</option>
            <option value="politeness">Politeness (thanks, please, sorry)</option>
            <option value="sarcasm">Sarcasm candidates (positive words, gold negative)</option>
            <option value="split">Classifiers split three ways</option>
            <option value="noise">Strong models agree, gold differs</option>
          </select>
        </label>
        <label>Classifier
          <select value={clf} onChange={(e) => setClf(e.target.value)}>
            {keys.map((k) => <option key={k} value={k}>{name[k] ?? k}</option>)}
          </select>
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={wrongOnly} onChange={(e) => setWrongOnly(e.target.checked)} />
          only where it disagrees with gold
        </label>
        <label>Search text
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. kill" />
        </label>
      </div>
      <p className="note">
        {filtered.length} of {items.length} items. &quot;Strong models&quot; are TF-IDF + LR, embeddings + LR, gpt-6.1-terra and
        gpt-6.1-sol; when all of them agree with each other and not with gold, the gold label may follow a dataset
        convention or be debatable.
      </p>
      {filtered.slice(0, shown).map((it) => (
        <article key={it.id} className="card" style={{ padding: 14, marginBottom: 12 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 6 }}>
            <span className="chip">{names[it.dataset]}</span>
            <span className="chip">gold: <strong>{it.gold}</strong></span>
            {it.tags.map((t) => <span key={t} className="chip">{TAG_LABELS[t] ?? t}</span>)}
            {it.all_strong_disagree_with_gold && <span className="chip">strong models disagree with gold</span>}
            <span className="note">VADER compound {it.vader_compound.toFixed(3)}</span>
          </div>
          <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", margin: "0 0 8px" }}>{it.text}</p>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {keys.map((k) => {
              const p = it.preds[k];
              const ok = p === it.gold;
              return (
                <span key={k} className={`chip ${ok ? "right" : "wrong"}`} title={name[k]}
                      style={k === clf ? { borderColor: "var(--text-secondary)" } : undefined}>
                  <span aria-hidden="true">{ok ? "✓" : "✗"}</span>
                  <span style={{ color: "var(--text-secondary)" }}>{k}</span> {p ?? "failed"}
                </span>
              );
            })}
          </div>
        </article>
      ))}
      {shown < filtered.length && (
        <button className="primary" onClick={() => setShown(shown + PAGE)}>Show {Math.min(PAGE, filtered.length - shown)} more</button>
      )}
    </div>
  );
}
