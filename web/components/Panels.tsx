// "Research basis" and "Limitations" panels, shown on every page.
export function ResearchPanels() {
  return (
    <section aria-label="Research basis and limitations" style={{ marginTop: 28 }}>
      <details className="panel" open>
        <summary>Research basis</summary>
        <ul>
          <li>
            Grounded in a systematic mapping study of 57 papers on emotions in software engineering. Its
            emotion-category × SDLC-phase map has 10 studies on <em>emotion recognition tools × cross-phase</em>,
            and that cell is fragmented: tools are proposed and compared one or two at a time on different data
            (e.g. Jongeling et al. 2017; Calefato et al. 2018 (Senti4SD); Novielli et al. 2018; Imtiaz et al. 2018;
            Lin et al. 2018; Novielli et al. 2020). SentiBench puts lexicon, supervised and LLM classifiers on the
            same splits with the same metrics.
          </li>
          <li>
            Gold sets: GitHub comments (Novielli et al. 2020), Jira comments (Ortu et al. 2016, polarity version
            used by Novielli et al.) and Stack Overflow posts (Calefato et al. 2018). All three label polarity
            from Shaver et al.&apos;s (1987) emotion framework: love and joy are positive; anger, sadness and fear
            negative; no emotion neutral.
          </li>
          <li>
            Prior work found that general-purpose tools misread technical vocabulary (Jongeling et al. 2017;
            Novielli et al. 2018), that SE-specific tools lose accuracy across platforms (Novielli et al. 2020),
            and that prompted LLMs do best where training data is limited or imbalanced while fine-tuned smaller
            models can still win where it is ample and balanced (Zhang et al. 2025). SentiBench measures these
            effects on one harness, with cost and run-to-run variance alongside accuracy.
          </li>
          <li>
            Classifiers: VADER (Hutto &amp; Gilbert 2014); RoBERTa-Twitter (Loureiro et al. 2022); TF-IDF and
            embedding baselines trained per dataset; four LLMs prompted zero- and few-shot with strict JSON output.
          </li>
        </ul>
      </details>
      <details className="panel">
        <summary>Limitations</summary>
        <ul>
          <li>
            This analyses <strong>text artefacts</strong>, not people. It gives no per-person scores and must not
            be used to infer how individual developers feel: emotion recognition of workers is prohibited in the EU
            workplace (AI Act, Art. 5(1)(f)), and expressed sentiment is a cue, not a reliable readout of an inner
            state (Barrett et al. 2019).
          </li>
          <li>
            The API models were scored on a stratified sample of 300 test items per dataset, not every item;
            intervals are 95 % bootstrap intervals over items and are wide for small differences.
          </li>
          <li>
            Gold labels follow each dataset&apos;s own conventions (thanks for a patch is positive in Jira, routine
            politeness neutral elsewhere). A classifier can be &quot;wrong&quot; by a convention, not by reading.
          </li>
          <li>
            The GitHub split is a seeded 70/30 split, not the authors&apos; (none is published). Some Jira test
            texts also occur in the train split (111 of 1,758), which flatters the supervised baselines there.
          </li>
          <li>
            LLM outputs vary between runs; numbers are from one run except in the variance table (3 runs on 100
            items per dataset, completed for the local model only). Model behaviour can change when the provider
            updates a model.
          </li>
          <li>
            The example sentences in the try-box are synthetic, written for this demo; they have no gold label.
          </li>
          <li>
            Not done: OpenAI credits ran out. Some planned runs (gpt-6-luna few-shot, the API models&apos; repeated runs,
            half of the cross-platform LLM cells, the API models&apos; outputs on the example sentences) are missing; the
            Results page lists them. Nothing is estimated in their place.
          </li>
        </ul>
      </details>
    </section>
  );
}
