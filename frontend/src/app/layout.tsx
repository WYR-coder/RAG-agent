import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Agent",
  description: "可私有化部署的 RAG 知识库问答系统",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <head>
        <meta httpEquiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src * data:; connect-src *; font-src 'self'" />
      </head>
      <body className="antialiased h-screen bg-gray-50">{children}</body>
    </html>
  );
}
