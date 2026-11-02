import type { Metadata } from "next";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { ResearchPanels } from "@/components/Panels";
import "./globals.css";

export const metadata: Metadata = {
  title: "SentiBench",
  description:
    "Lexicon, supervised and LLM sentiment classifiers compared on software-engineering gold sets, with cost, agreement and error analysis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site">
          <div className="wrap">
            <Link href="/" className="brand">SentiBench</Link>
            <Nav />
          </div>
        </header>
        <div className="demo-banner">
          <div className="wrap">
            Demo mode: no API key needed. Every number and model output on this site comes from real runs of the
            SentiBench pipeline, precomputed and shipped as static data.
          </div>
        </div>
        <main>
          <div className="wrap">
            {children}
            <ResearchPanels />
          </div>
        </main>
        <footer className="site">
          <div className="wrap">
            Gold sets: GitHub (Novielli et al. 2020, CC BY 4.0), Jira (Ortu et al. 2016) and Stack Overflow (Calefato et al.
            2018) via the collab-uniba EMTk datasets (MIT). Texts are shown with @handles and e-mail addresses masked.
          </div>
        </footer>
      </body>
    </html>
  );
}
