// Server-side loader: pages read the report data at build time, so the static export
// ships it pre-rendered.
import { readFileSync } from "node:fs";
import path from "node:path";
import type { Results } from "./results";

export function loadResults(): Results {
  const file = path.join(process.cwd(), "public", "data", "results.json");
  return JSON.parse(readFileSync(file, "utf-8"));
}
