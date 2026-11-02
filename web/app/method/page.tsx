import { loadResults } from "@/lib/load";
import { DATASETS } from "@/lib/results";

export default function MethodPage() {
  const r = loadResults() as ReturnType<typeof loadResults> & {
    prompts: { "zs-basic": string; "zs-guided": string; shots: Record<string, { text: string; label: string }[]> };
  };
  const sources: Record<string, string> = {
    github: "Novielli et al. 2020 (MSR), figshare 10.6084/m9.figshare.11604597, CC BY 4.0",
    jira: "Ortu et al. 2016 (MSR); polarity version from collab-uniba/EMTK_datasets, MIT",
    so: "Calefato et al. 2018 (EMSE, Senti4SD); collab-uniba/EMTK_datasets, MIT",
  };
  const splitNote: Record<string, string> = {
    github: "seeded stratified 70/30 (no official split)",
    jira: "official train/test",
    so: "official train/test",
  };
  return (
    <>
      <h1>Method</h1>
      <p className="lede">
        How the numbers were produced. The full report, with every table and the error analysis, is{" "}
        <code>docs/benchmark.md</code> in the repository; it and this site are generated from the same run files.
      </p>

      <section className="card">
        <h2>Datasets and splits</h2>
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>Dataset</th><th>Source and licence</th><th>Split</th><th className="num">Train</th><th className="num">Dev (in train)</th><th className="num">Test</th><th className="num">Test neg / neu / pos</th></tr></thead>
            <tbody>
              {DATASETS.map((ds) => {
                const s = r.datasets[ds].splits;
                const t = s.test.classes;
                return (
                  <tr key={ds}>
                    <td>{r.names[ds]}</td><td>{sources[ds]}</td><td>{splitNote[ds]}</td>
                    <td className="num">{s.train.n}</td><td className="num">{s.dev.n}</td><td className="num">{s.test.n}</td>
                    <td className="num">{t.negative} / {t.neutral} / {t.positive}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="note" style={{ marginTop: 8 }}>
          Dev is a stratified 20% of each train split (seed 2026). Hyperparameters, VADER thresholds and prompts were chosen
          on dev; supervised models were then refitted on the full train split and test was scored once. API models ran on
          the first {r.sample_n} items per dataset of a fixed stratified order of the test split; the variance runs use the
          first {r.variance_n} of those.
        </p>
      </section>

      <section className="card">
        <h2>Classifiers</h2>
        <ul>
          <li><strong>VADER</strong> (Hutto &amp; Gilbert 2014): compound score with the authors&apos; ±0.05 thresholds, and with thresholds tuned on each dataset&apos;s dev split.</li>
          <li><strong>RoBERTa-Twitter</strong>: <code>cardiffnlp/twitter-roberta-base-sentiment-latest</code> (Loureiro et al. 2022), general-domain, off the shelf on CPU.</li>
          <li><strong>TF-IDF + LR</strong>: word uni- and bigrams, sublinear tf, logistic regression; C and class weighting chosen on dev. The same family as SentiCR (bag of words).</li>
          <li><strong>Embeddings + LR</strong>: OpenAI <code>text-embedding-3-small</code> vectors with a logistic regression head trained per dataset.</li>
          <li><strong>LLMs</strong>: <code>gpt-6-luna</code>, <code>gpt-6.1-terra</code>, <code>gpt-6.1-sol</code> (OpenAI Responses API, strict JSON schema, reasoning effort low) and <code>qwen3:8b</code> locally through Ollama (schema-constrained decoding, reasoning off). One item per call.</li>
        </ul>
      </section>

      <section className="card">
        <h2>Prompts</h2>
        <p className="note">
          Four variants. zs-basic, zs-guided and fs-guided were written before any model ran; fs-basic was added after the
          first dev round showed the guided rules clashing with Jira&apos;s conventions. The few-shot variants append nine
          labelled examples (three per class) drawn at random, with a fixed seed, from the scored dataset&apos;s fit split.
        </p>
        <h3>zs-basic</h3>
        <pre className="prompt">{r.prompts["zs-basic"]}</pre>
        <h3>zs-guided</h3>
        <pre className="prompt">{r.prompts["zs-guided"]}</pre>
        <h3>Few-shot examples (Jira)</h3>
        <pre className="prompt">{r.prompts.shots.jira.map((s) => `Message: """${s.text}"""\nLabel: ${s.label}`).join("\n\n")}</pre>
      </section>

      <section className="card">
        <h2>Metrics</h2>
        <ul>
          <li>Macro-F1 (mean of the three per-class F1 scores) and Cohen&apos;s κ against gold, each with a 95% percentile bootstrap interval over items (1,000 resamples).</li>
          <li>Paired bootstrap intervals for differences between two classifiers on the same items.</li>
          <li>A failed LLM call (invalid or truncated JSON, refusal, API error after retries) counts as a wrong answer, never as a class.</li>
          <li>Cost from published prices ({r.prices.as_of}): {Object.entries(r.prices.usd_per_million).map(([m, p]) => `${m} $${p[0]} / $${p[1]} / $${p[2]}`).join("; ")} per million input / cached input / output tokens.</li>
        </ul>
      </section>
    </>
  );
}
