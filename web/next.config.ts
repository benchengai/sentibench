import type { NextConfig } from "next";

// Static export: `npm run build` writes a self-contained site to out/, deployable to Vercel or any static host.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
