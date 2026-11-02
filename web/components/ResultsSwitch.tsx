"use client";

import { useState } from "react";
import type { Classifier, Dataset, Scores } from "@/lib/results";
import { DotPlot } from "./DotPlot";

type Props = {
  classifiers: Classifier[];
  sample: Record<string, Partial<Record<Dataset, Scores>>>;
  full: Record<string, Partial<Record<Dataset, Scores>>>;
  names: Record<Dataset, string>;
  sampleN: number;
};

export function ResultsSwitch({ classifiers, sample, full, names, sampleN }: Props) {
  const [which, setWhich] = useState<"sample" | "full">("sample");
  return (
    <div>
      <div className="card-head">
        <p>
          {which === "sample"
            ? `Every classifier on the same stratified sample of ${sampleN} test items per dataset, so rows are directly comparable.`
            : "Classifiers cheap enough to run on every test item (the API models other than gpt-6-luna were sampled)."}
        </p>
        <div className="seg" role="group" aria-label="Test items">
          <button aria-pressed={which === "sample"} onClick={() => setWhich("sample")}>LLM test sample</button>
          <button aria-pressed={which === "full"} onClick={() => setWhich("full")}>Full test sets</button>
        </div>
      </div>
      <DotPlot classifiers={classifiers} scores={which === "sample" ? sample : full} names={names} />
    </div>
  );
}
