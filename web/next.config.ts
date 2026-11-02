import type { NextConfig } from "next";

// DEMO_BASE_PATH serves the static demo from a sub-path, e.g. /demos/sentibench on a
// personal site. Runtime fetches read NEXT_PUBLIC_BASE_PATH to find the data files.
const basePath = process.env.DEMO_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
