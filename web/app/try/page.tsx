import { readFileSync } from "node:fs";
import path from "node:path";
import { TryBox, type Example } from "@/components/TryBox";
import { loadResults } from "@/lib/load";

export default function TryPage() {
  const r = loadResults();
  const examples = JSON.parse(readFileSync(path.join(process.cwd(), "public", "data", "examples.json"), "utf-8")) as {
    note: string;
    sentences: Example[];
  };
  return (
    <>
      <h1>Try a sentence</h1>
      <p className="lede">
        Type a developer message. VADER and the TF-IDF baseline run in your browser with no API key; the larger models&apos;
        outputs are precomputed for the example sentences. This classifies the text, not the person who wrote it.
      </p>
      <TryBox examples={examples.sentences} note={examples.note} names={r.names} tuned={r.vader_tuned} classifiers={r.classifiers} />
    </>
  );
}
