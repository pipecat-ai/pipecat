import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: new URL(".", import.meta.url).pathname,
  },
};

export default nextConfig;
