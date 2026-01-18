/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['ai'],

  // Environment variables available on the client
  env: {
    STORM_API_URL: process.env.STORM_API_URL || '',
  },

  // API rewrites for proxying to backend
  // If STORM_API_URL is set, proxy to external backend
  // If not set, requests go to same domain (for combined deployments)
  async rewrites() {
    const apiUrl = process.env.STORM_API_URL;

    // If no external API URL configured, don't set up rewrites
    // This allows the app to work when backend is on same domain
    if (!apiUrl) {
      return [];
    }

    return [
      {
        source: '/api/v1/:path*',
        destination: `${apiUrl}/api/v1/:path*`,
      },
      {
        source: '/health',
        destination: `${apiUrl}/health`,
      },
      {
        source: '/docs',
        destination: `${apiUrl}/docs`,
      },
      {
        source: '/openapi.json',
        destination: `${apiUrl}/openapi.json`,
      },
    ];
  },
};

module.exports = nextConfig;
