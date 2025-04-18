/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",

  async rewrites() {
    return [
      {
        source: "/:path*",
        destination: "http://localhost:7860/:path*",
      },
    ];
  },
};

export default nextConfig;
