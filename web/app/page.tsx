import Link from "next/link";
import { CostScatter, type CostPoint } from "@/components/CostScatter";
import { CrossPlatform } from "@/components/CrossPlatform";
import { Agreement } from "@/components/Agreement";
import { ResultsSwitch } from "@/components/ResultsSwitch";
import { loadResults } from "@/lib/load";
import { DATASETS, fmt, mean } from "@/lib/results";

export default function Home() {
  const r = loadResults();
  const byKey = Object.fromEntries(r.classifiers.map((c) => [c.key, c]));
  const meanF1 = (key: string) => mean(DATASETS.map((ds) => r.sample[key][ds]!.macro_f1));
  const ranked = r.classifiers.filter((c) => r.sample[c.key]).sort((a, b) => meanF1(b.key) - meanF1(a.key));
  const best = ranked[0];
  const bestNonLlm = ranked.find((c) => c.family !== "LLM")!;
  const totalItems = DATASETS.reduce((n, ds) => n + r.datasets[ds].splits.train.n + r.datasets[ds].splits.test.n, 0);

  const cost = Object.fromEntries(r.cost.map((c) => [c.key, c]));
  const points: CostPoint[] = ranked.map((c) => {
    const k = cost[c.key];
    const latency = k?.latency_median_s !== undefined ? `median ${k.latency_median_s.toFixed(2)} s per call`
      : k?.ms_per_item ? `${k.ms_per_item.toFixed(2)} ms per item (batch)` : "batch";
    const local = c.model?.startsWith("ollama") || !["embed-lr"].includes(c.key) && c.family !== "LLM";
    return { c, f1: meanF1(c.key), cost: local ? 0 : k?.cost_per_1k ?? 0, latency };
  });

  return (
    <>
      <h1>Sentiment classifiers on software-engineering text, measured on one harness</h1>
      <p className="lede">
        SentiBench compares a lexicon (VADER), a general pre-trained model, supervised baselines trained on SE data,
        and four LLMs prompted zero- and few-shot, on three published gold sets: GitHub, Jira and Stack Overflow.
        Prompts and settings were chosen on dev splits; test was scored once.
      </p>

      <div className="kpis">
        <div className="tile hero">
          <div className="label">Best mean macro-F1 on the LLM test sample</div>
          <div className="value">{fmt(meanF1(best.key))}</div>
          <div className="sub">{best.name}</div>
        </div>
        <div className="tile">
          <div className="label">Best without an LLM</div>
          <div className="value">{fmt(meanF1(bestNonLlm.key))}</div>
          <div className="sub">{bestNonLlm.name}</div>
        </div>
        <div className="tile">
          <div className="label">Gold-standard items</div>
          <div className="value">{totalItems.toLocaleString("en-US")}</div>
          <div className="sub">3 datasets, 3 polarity classes</div>
        </div>
        <div className="tile">
          <div className="label">API spend, all runs</div>
          <div className="value">${r.spend.total_usd.toFixed(2)}</div>
          <div className="sub">dev tuning, test, variance runs</div>
        </div>
      </div>

      {r.incomplete.length > 0 && <Incomplete gaps={r.incomplete} />}

      <section className="card">
        <h2>Macro-F1 by classifier and dataset</h2>
        <ResultsSwitch classifiers={r.classifiers} sample={r.sample} full={r.full} names={r.names} sampleN={r.sample_n} />
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <h2>Cross-platform: train or tune on one dataset, test on another</h2>
            <p>
              Macro-F1 on the target test split; the diagonal (bold) is within-platform. Moved to another dataset, TF-IDF + LR
              loses {dropRange(r.cross["TF-IDF + LR"])} and Embeddings + LR {dropRange(r.cross["Embeddings + LR"])}.
              Cells marked – were not run (the API credits ran out).
            </p>
          </div>
        </div>
        <CrossPlatform cross={r.cross} names={r.names} />
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <h2>Quality against cost</h2>
            <p>
              Mean macro-F1 over the three datasets on the LLM test sample, against the published price per 1,000
              classified items (prices as of {r.prices.as_of}). Local classifiers have no per-item price.
            </p>
          </div>
        </div>
        <CostScatter points={points} />
      </section>

      <div className="grid-2">
        <section className="card">
          <h2>Run-to-run variance</h2>
          <p className="note">
            Three runs of each LLM on the first {r.variance_n} items per dataset of the sample, same prompt, default sampling.
            {r.variance.length < r.classifiers.filter((c) => c.family === "LLM").length &&
              " Only the runs that completed are shown; the API models' repeated runs were cut short when the credits ran out."}
          </p>
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Classifier</th><th className="num">Macro-F1, 3 runs</th><th className="num">Labels changed</th><th className="num">κ between runs</th></tr></thead>
              <tbody>
                {r.variance.map((v) => (
                  <tr key={v.key}>
                    <td>{byKey[v.key]?.name ?? v.key}</td>
                    <td className="num">{fmt(mean(v.macro_f1_runs), 3)} ({fmt(Math.min(...v.macro_f1_runs), 3)}–{fmt(Math.max(...v.macro_f1_runs), 3)})</td>
                    <td className="num">{(100 * v.flip_rate).toFixed(1)}% ({v.items_changed} of {v.items})</td>
                    <td className="num">{fmt(v.mean_pairwise_kappa)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section className="card">
          <h2>Cost, latency and failures</h2>
          <p className="note">Per item on test. LLM latency is per call with 6 to 8 calls in flight; local timings are batch throughput on a laptop CPU.</p>
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Classifier</th><th className="num">USD / 1k</th><th className="num">Latency</th><th className="num">Failed</th></tr></thead>
              <tbody>
                {r.cost.map((c) => {
                  const cl = byKey[c.key];
                  const local = cl.model?.startsWith("ollama") || (cl.family !== "LLM" && c.key !== "embed-lr");
                  return (
                    <tr key={c.key}>
                      <td>{cl.name}</td>
                      <td className="num">{local ? "0 (local)" : c.cost_per_1k !== null ? `$${c.cost_per_1k.toFixed(c.cost_per_1k < 0.01 ? 4 : 3)}` : "–"}</td>
                      <td className="num">{c.latency_median_s !== undefined ? `${c.latency_median_s.toFixed(2)} s (p95 ${c.latency_p95_s!.toFixed(2)})` : c.ms_per_item ? `${c.ms_per_item.toFixed(2)} ms` : "–"}</td>
                      <td className="num">{c.calls ? `${c.failed} of ${c.calls}` : "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="card">
        <div className="card-head">
          <div>
            <h2>Agreement between classifiers</h2>
            <p>Cohen&apos;s κ between every pair on the LLM test sample (all three datasets pooled); the last column is agreement with gold.</p>
          </div>
        </div>
        <Agreement agreement={r.agreement} classifiers={r.classifiers} />
      </section>

      <section className="card">
        <h2>Where classifiers fail</h2>
        <p className="note">
          Surface tags on the LLM test sample. They select candidates, not verified cases: read the examples in the{" "}
          <Link href="/errors/">error browser</Link>.
        </p>
        <ErrorSummary r={r} names={byKey} />
      </section>

      <section className="card">
        <h2>Prompt selection on dev</h2>
        <p className="note">
          Mean dev macro-F1 per prompt variant; per model the best zero-shot and the best few-shot variant were taken to
          test. {r.prompt_dev["openai:gpt-6.1-sol"] ? "gpt-6.1-sol and gpt-6.1-terra were tuned on 60 dev items per dataset, the cheaper models on 150." : ""}
        </p>
        <div className="scroll-x">
          <table className="data">
            <thead>
              <tr><th>Model</th><th>Prompt</th>{DATASETS.map((ds) => <th key={ds} className="num">{r.names[ds]}</th>)}<th className="num">Mean</th><th>Selected</th></tr>
            </thead>
            <tbody>
              {Object.entries(r.prompt_dev).flatMap(([model, prompts]) =>
                Object.entries(prompts).map(([p, v]) => {
                  const sel = Object.entries(r.selected_prompts[model]).filter(([, sp]) => sp === p).map(([k]) => k);
                  return (
                    <tr key={model + p}>
                      <td>{model.split(":").slice(1).join(":")}</td>
                      <td><code>{p}</code></td>
                      {DATASETS.map((ds) => <td key={ds} className="num">{fmt(v.datasets[ds].macro_f1, 3)}</td>)}
                      <td className="num">{fmt(v.mean, 3)}</td>
                      <td>{sel.join(", ")}</td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function ErrorSummary({ r, names }: { r: ReturnType<typeof loadResults>; names: Record<string, { name: string }> }) {
  const e = r.errors;
  const keys = r.agreement.keys;
  return (
    <div className="scroll-x">
      <table className="data">
        <thead>
          <tr>
            <th>Classifier</th>
            <th className="num">Technical vocabulary: called negative ({e.technical.n} items)</th>
            <th className="num">Same, other non-negative items ({e.technical.baseline_n})</th>
            <th className="num">Politeness: accuracy ({e.politeness.n})</th>
            <th className="num">Sarcasm candidates: called positive ({e.sarcasm.n})</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k}>
              <td>{names[k]?.name ?? k}</td>
              <td className="num">{fmt(e.technical.classifiers[k].called_negative)}</td>
              <td className="num">{fmt(e.technical.baseline_called_negative[k])}</td>
              <td className="num">{fmt(e.politeness.classifiers[k].accuracy)}</td>
              <td className="num">{fmt(e.sarcasm.classifiers[k].called_positive)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function dropRange(m: ReturnType<typeof loadResults>["cross"][string] | undefined) {
  if (!m) return "–";
  const drops: number[] = [];
  for (const src of DATASETS) for (const tgt of DATASETS)
    if (src !== tgt && m[src]?.[tgt] && m[tgt]?.[tgt]) drops.push(m[tgt][tgt].macro_f1 - m[src][tgt].macro_f1);
  return `${Math.min(...drops).toFixed(2)} to ${Math.max(...drops).toFixed(2)} macro-F1`;
}

function Incomplete({ gaps }: { gaps: ReturnType<typeof loadResults>["incomplete"] }) {
  const groups = new Map<string, { done: number; needed: number }>();
  for (const g of gaps) {
    const key = `${g.classifier}: ${g.what.replace(/ \d+$/, "")}`;
    const cur = groups.get(key) ?? { done: 0, needed: 0 };
    groups.set(key, { done: cur.done + g.done, needed: cur.needed + g.needed });
  }
  return (
    <details className="panel" style={{ marginBottom: 20 }}>
      <summary>Not completed: some planned runs were cut short when the shared API account ran out of credits</summary>
      <p className="note" style={{ marginTop: 8 }}>
        These runs are missing or partial. Classifiers and tables that depend on them are left out rather than shown on
        fewer items; everything shown is from completed runs.
      </p>
      <ul className="note" style={{ margin: 0, paddingLeft: 18 }}>
        {[...groups.entries()].map(([k, v]) => (
          <li key={k}>{k} ({v.done} of {v.needed} items)</li>
        ))}
      </ul>
    </details>
  );
}
