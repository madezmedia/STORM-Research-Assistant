/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['ai'],

  // Environment variables available on the client
  env: {
    STORM_API_URL: process.env.STORM_API_URL || 'http://localhost:8000',
  },

  // API rewrites for proxying to backend
  async rewrites() {
    const apiUrl = process.env.STORM_API_URL || 'http://localhost:8000';

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
