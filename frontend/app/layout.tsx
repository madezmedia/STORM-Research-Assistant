import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { Providers } from "./providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "STORM Research Assistant",
  description: "AI-powered content generation with multi-perspective research",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <Providers>
          {/* Background gradient */}
          <div className="fixed inset-0 bg-gradient-radial pointer-events-none" />

          {/* Sidebar */}
          <Sidebar />

          {/* Main content area */}
          <div className="min-h-screen transition-all duration-300 pl-64">
            <Header />
            <main className="pt-16 p-6">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
