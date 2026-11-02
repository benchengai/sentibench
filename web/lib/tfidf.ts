// Browser re-implementation of the TF-IDF + logistic regression baseline (scikit-learn
// TfidfVectorizer with word uni- and bigrams, sublinear tf, l2 norm; multinomial
// LogisticRegression). Weights come from `sentibench report` (public/data/models/tfidf_*.json);
// scripts/parity.ts checks the labels against the Python predictions on every test item.

export type TfidfModel = {
  classes: string[];
  ngram_range: [number, number];
  sublinear_tf: boolean;
  terms: string[];
  idf: number[];
  coef: number[][];
  intercept: number[];
};

// scikit-learn's default token_pattern (?u)\b\w\w+\b: runs of two or more word characters.
const WORD = /[\p{L}\p{N}_]+/gu;

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(WORD) || []).filter((t) => Array.from(t).length >= 2);
}

export class Tfidf {
  private model: TfidfModel;
  private index: Map<string, number>;

  constructor(model: TfidfModel) {
    this.model = model;
    this.index = new Map(model.terms.map((t, i) => [t, i]));
  }

  /** Per-class probabilities plus the terms that pushed the winning class most. */
  score(text: string): { label: string; probs: Record<string, number>; terms: { term: string; weight: number }[] } {
    const tokens = tokenize(text);
    const [minN, maxN] = this.model.ngram_range;
    const counts = new Map<number, number>();
    for (let n = minN; n <= maxN; n++) {
      for (let i = 0; i + n <= tokens.length; i++) {
        const j = this.index.get(tokens.slice(i, i + n).join(" "));
        if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
      }
    }
    const values = new Map<number, number>();
    let norm = 0;
    for (const [j, c] of counts) {
      const tf = this.model.sublinear_tf ? 1 + Math.log(c) : c;
      const v = tf * this.model.idf[j];
      values.set(j, v);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    const decision = this.model.intercept.slice();
    for (const [j, v] of values) {
      const x = norm > 0 ? v / norm : 0;
      for (let k = 0; k < decision.length; k++) decision[k] += x * this.model.coef[k][j];
    }
    const best = decision.indexOf(Math.max(...decision));
    const max = decision[best];
    const exp = decision.map((d) => Math.exp(d - max));
    const total = exp.reduce((a, b) => a + b, 0);
    const probs: Record<string, number> = {};
    this.model.classes.forEach((c, k) => (probs[c] = exp[k] / total));
    const terms = [...values.entries()]
      .map(([j, v]) => ({ term: this.model.terms[j], weight: (norm > 0 ? v / norm : 0) * this.model.coef[best][j] }))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 6);
    return { label: this.model.classes[best], probs, terms };
  }
}
