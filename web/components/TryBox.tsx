"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Classifier, Dataset } from "@/lib/results";
import { Tfidf, type TfidfModel } from "@/lib/tfidf";
import { Vader, vaderLabel, type VaderData } from "@/lib/vader";

const NOT_DONE = "not done: OpenAI credits ran out";

export type Example = {
  id: string;
  text: string;
  phenomenon: string;
  look_for: string;
  llm: Record<string, Partial<Record<Dataset, string | null>>>;
  roberta: string | null;
  embed_lr: Partial<Record<Dataset, string | null>>;
};

type Props = {
  examples: Example[];
  note: string;
  names: Record<Dataset, string>;
  tuned: Record<Dataset, { thresholds: [number, number] }>;
  classifiers: Classifier[];
};

export function TryBox({ examples, note, names, tuned, classifiers }: Props) {
  const [text, setText] = useState(examples[0]?.text ?? "");
  const [domain, setDomain] = useState<Dataset>("jira");
  const [vader, setVader] = useState<Vader | null>(null);
  const models = useRef<Partial<Record<Dataset, Tfidf>>>({});
  const [, setLoaded] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_BASE_PATH}/data/models/vader.json`).then((r) => r.json()).then((d: VaderData) => setVader(new Vader(d)))
      .catch((e) => setLoadError(String(e)));
  }, []);
  useEffect(() => {
    if (models.current[domain]) return;
    fetch(`${process.env.NEXT_PUBLIC_BASE_PATH}/data/models/tfidf_${domain}.json`).then((r) => r.json()).then((m: TfidfModel) => {
      models.current[domain] = new Tfidf(m);
      setLoaded((n) => n + 1);
    }).catch((e) => setLoadError(String(e)));
  }, [domain]);

  const compound = useMemo(() => (vader && text.trim() ? vader.compound(text) : null), [vader, text]);
  const tfidf = models.current[domain];
  const tf = useMemo(() => (tfidf && text.trim() ? tfidf.score(text) : null), [tfidf, text]);
  const example = examples.find((e) => e.text === text.trim());
  const llms = classifiers.filter((c) => c.family === "LLM");

  return (
    <div>
      <div className="card">
        <label htmlFor="sentence" style={{ fontWeight: 600 }}>A developer message</label>
        <textarea id="sentence" value={text} onChange={(e) => setText(e.target.value)} maxLength={2000} />
        <div className="filters" style={{ marginTop: 10, marginBottom: 0 }}>
          <label>Training domain (TF-IDF model, tuned VADER thresholds, few-shot examples)
            <select value={domain} onChange={(e) => setDomain(e.target.value as Dataset)}>
              {(Object.keys(names) as Dataset[]).map((d) => <option key={d} value={d}>{names[d]}</option>)}
            </select>
          </label>
        </div>
        <p className="note" style={{ marginTop: 10, marginBottom: 6 }}>Example sentences (synthetic, written for this demo):</p>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {examples.map((e) => (
            <button key={e.id} className="chip" style={{ cursor: "pointer" }} onClick={() => setText(e.text)}
                    aria-pressed={text.trim() === e.text}>
              {e.text.length > 38 ? e.text.slice(0, 36) + "…" : e.text}
            </button>
          ))}
        </div>
      </div>

      {loadError && <p>Could not load the browser models ({loadError}).</p>}

      <div className="grid-2">
        <section className="card">
          <h2>Runs live in your browser</h2>
          <table className="data">
            <tbody>
              <tr>
                <td>VADER, default thresholds (±0.05)</td>
                <td><strong>{compound === null ? "–" : vaderLabel(compound)}</strong></td>
                <td className="num">compound {compound === null ? "–" : compound.toFixed(4)}</td>
              </tr>
              <tr>
                <td>VADER, thresholds tuned on {names[domain]} dev ({tuned[domain].thresholds.map((t) => t.toFixed(2)).join(" / ")})</td>
                <td><strong>{compound === null ? "–" : vaderLabel(compound, tuned[domain].thresholds)}</strong></td>
                <td />
              </tr>
              <tr>
                <td>TF-IDF + LR trained on {names[domain]}</td>
                <td><strong>{tf?.label ?? "–"}</strong></td>
                <td className="num">{tf ? `p = ${tf.probs[tf.label].toFixed(2)}` : ""}</td>
              </tr>
            </tbody>
          </table>
          {tf && tf.terms.length > 0 && (
            <p className="note" style={{ marginTop: 10 }}>
              Terms that moved TF-IDF + LR towards <em>{tf.label}</em> most:{" "}
              {tf.terms.map((t) => `${t.term} (${t.weight >= 0 ? "+" : ""}${t.weight.toFixed(2)})`).join(", ")}
            </p>
          )}
          <p className="note">
            Both are re-implementations of the Python classifiers, checked against them on all 5,221 test items
            (identical VADER scores and TF-IDF labels; <code>npm run parity</code>).
          </p>
        </section>

        <section className="card">
          <h2>Precomputed model outputs</h2>
          {example ? (
            <>
              <p className="note">
                <strong>{example.phenomenon}</strong>: {example.look_for} No gold label: this sentence is not from a gold set.
              </p>
              <table className="data">
                <tbody>
                  <tr><td>RoBERTa-Twitter (general domain)</td><td><strong>{example.roberta ?? "–"}</strong></td></tr>
                  <tr><td>Embeddings + LR trained on {names[domain]}</td><td><strong>{example.embed_lr[domain] ?? NOT_DONE}</strong></td></tr>
                  {llms.map((c) => {
                    const label = example.llm[c.key]?.[domain];
                    return (
                      <tr key={c.key}>
                        <td>{c.name}{c.kind === "few-shot" ? `, examples from ${names[domain]}` : ""}</td>
                        <td><strong>{label === undefined ? NOT_DONE : label ?? "failed"}</strong></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="note" style={{ marginTop: 8 }}>
                {note} Rows marked &quot;{NOT_DONE}&quot; have no output: those calls were never made, and nothing is shown in their place.
              </p>
            </>
          ) : (
            <p className="note">
              RoBERTa, the embedding model and the LLMs need a model download or an API call, so in demo mode their
              outputs are shown only for the example sentences, from real pipeline runs. Pick an example to see them, or
              run <code>sentibench classify &quot;…&quot; -m openai:gpt-6-luna/zs-guided</code> locally with an API key.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
