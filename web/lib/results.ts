// Types and helpers for public/data/results.json, written by `sentibench report`.
// No Node imports here: client components use these helpers too (the loader is in load.ts).

export const DATASETS = ["github", "jira", "so"] as const;
export type Dataset = (typeof DATASETS)[number];
export const LABELS = ["negative", "neutral", "positive"] as const;

export type Scores = {
  n: number;
  macro_f1: number;
  kappa: number;
  accuracy: number;
  per_class: Record<string, { precision: number; recall: number; f1: number }>;
  severe_error_rate: number;
  failed: number;
  confusion: number[][];
  ci: { macro_f1: [number, number]; kappa: [number, number] };
};

export type Classifier = {
  key: string;
  name: string;
  family: string;
  model?: string;
  prompt?: string;
  kind?: string;
};

export type Results = {
  generated: string;
  names: Record<Dataset, string>;
  datasets: Record<Dataset, { splits: Record<string, { n: number; classes: Record<string, number> }>; test_texts_also_in_train: number }>;
  classifiers: Classifier[];
  selected_prompts: Record<string, { "zero-shot": string; "few-shot": string }>;
  prompt_dev: Record<string, Record<string, { datasets: Record<Dataset, { n: number; macro_f1: number; cost_per_1k: number }>; mean: number }>>;
  full: Record<string, Partial<Record<Dataset, Scores>>>;
  sample: Record<string, Partial<Record<Dataset, Scores>>>;
  sample_n: number;
  variance_n: number;
  paired: { a: string; b: string; dataset: Dataset; diff: number; ci: [number, number]; p_sign: number }[];
  cross: Record<string, Record<Dataset, Record<Dataset, { macro_f1: number; kappa: number; n: number }>>>;
  cost: { key: string; cost_per_1k: number | null; latency_median_s?: number; latency_p95_s?: number; ms_per_item?: number | null; calls?: number; failed?: number; failure_rate?: number; input_tokens?: number; output_tokens?: number }[];
  variance: { key: string; runs: number; items: number; items_changed: number; flip_rate: number; mean_pairwise_kappa: number; macro_f1_runs: number[] }[];
  agreement: { keys: string[]; kappa: Record<string, Record<string, number>> };
  errors: Record<string, any>;
  tags: Record<string, string>;
  spend: { total_usd: number; embeddings_usd: number };
  incomplete: { classifier: string; what: string; dataset: Dataset; done: number; needed: number }[];
  vader_tuned: Record<Dataset, { thresholds: [number, number]; dev_macro_f1: number }>;
  prices: { as_of: string; source: string; usd_per_million: Record<string, [number, number, number]> };
};

// Three colour groups: what the classifier learned from.
export const GROUPS = [
  { id: "llm", label: "LLM prompting", color: "var(--series-1)" },
  { id: "supervised", label: "Trained on SE data", color: "var(--series-2)" },
  { id: "none", label: "No SE training", color: "var(--series-3)" },
] as const;

export function groupOf(c: Classifier): (typeof GROUPS)[number] {
  if (c.family === "LLM") return GROUPS[0];
  if (c.family.startsWith("Supervised")) return GROUPS[1];
  return GROUPS[2];
}

export const fmt = (x: number | null | undefined, digits = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(digits);

export const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
