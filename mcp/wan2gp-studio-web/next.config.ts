import type { NextConfig } from "next";

const KIMODO_URL =
  process.env.KIMODO_URL || "http://gpu-worker-subprocess.ai-services:18470";

const nextConfig: NextConfig = {
  output: "standalone",
  basePath: "/studio",
  async rewrites() {
    return [
      // Proxy /studio/kimodo/* → Viser 3D server on GPU worker
      {
        source: "/kimodo/:path*",
        destination: `${KIMODO_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
