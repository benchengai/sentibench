// Checks the browser classifiers against the Python ones on every test item:
// VADER compound scores (must match to 4 decimals) and TF-IDF + LR labels for the three
// trained models. Run with `npm run parity` after `sentibench report`.
import { readFileSync } from "node:fs";
import { Tfidf, type TfidfModel } from "../lib/tfidf.ts";
import { Vader, type VaderData } from "../lib/vader.ts";

type Row = { dataset: string; text: string; vader: number; tfidf: Record<string, string> };

const here = new URL(".", import.meta.url);
const read = (rel: string) => JSON.parse(readFileSync(new URL(rel, here), "utf-8"));
const rows: Row[] = read("parity_fixture.json");
const vader = new Vader(read("../public/data/models/vader.json") as VaderData);
const datasets = ["github", "jira", "so"];
const tfidf = Object.fromEntries(
  datasets.map((ds) => [ds, new Tfidf(read(`../public/data/models/tfidf_${ds}.json`) as TfidfModel)]),
);

let vaderMismatch = 0;
const examples: string[] = [];
for (const row of rows) {
  const got = vader.compound(row.text);
  if (Math.abs(got - row.vader) > 1e-9) {
    vaderMismatch++;
    if (examples.length < 5) examples.push(`  python ${row.vader} ts ${got}: ${row.text.slice(0, 100)}`);
  }
}
console.log(`VADER: ${rows.length - vaderMismatch} of ${rows.length} compound scores identical`);
examples.forEach((e) => console.log(e));

let failed = vaderMismatch > 0;
for (const ds of datasets) {
  let same = 0;
  for (const row of rows) if (tfidf[ds].score(row.text).label === row.tfidf[ds]) same++;
  const share = same / rows.length;
  console.log(`TF-IDF + LR trained on ${ds}: ${same} of ${rows.length} labels identical (${(100 * share).toFixed(2)}%)`);
  // Weights are rounded to 4 decimals for the browser, so a near-tie can flip; allow 0.2 %.
  if (share < 0.998) failed = true;
}
process.exit(failed ? 1 : 0);
