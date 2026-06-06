import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mini Attention Notebook | Q/K/V self-attention visualizer",
  description: "Interactive visualizer playground for Transformer Self-Attention mechanics. Analyze token embeddings, matrices projections (Q, K, V), causal masking, and multi-head attention calculations step-by-step.",
  keywords: ["self-attention", "transformer", "QKV matrices", "deep learning visualizer", "causal mask", "multi-head attention", "neural networks", "linear algebra"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <body className={`${geistSans.variable} ${geistMono.variable} min-h-full bg-[#070b13] font-sans antialiased selection:bg-indigo-500/30 selection:text-indigo-200`}>
        {children}
      </body>
    </html>
  );
}
