import { ErrorBrowser } from "@/components/ErrorBrowser";
import { loadResults } from "@/lib/load";

export default function ErrorsPage() {
  const r = loadResults();
  return (
    <>
      <h1>Error browser</h1>
      <p className="lede">
        The {3 * r.sample_n} items of the LLM test sample with every classifier&apos;s label. Filter by the hard cases the
        literature names: technical vocabulary (&quot;kill the process&quot;), politeness, and sarcasm. Tags are surface
        patterns that pick candidates; whether a case really is sarcasm is for the reader to judge.
      </p>
      <ErrorBrowser classifiers={r.classifiers} names={r.names} />
    </>
  );
}
