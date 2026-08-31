"use client";

import Link from "next/link";
import {
  Cpu,
  Search,
  Zap,
  BarChart3,
  Sparkles,
  ExternalLink,
  Play,
  FileText,
  Layers,
} from "lucide-react";

export default function AboutPage() {
  const techStack = [
    {
      category: "Frontend",
      items: ["Next.js 14 (App Router)", "TypeScript", "Tailwind CSS", "TanStack Query"],
    },
    {
      category: "Backend",
      items: ["Python & FastAPI", "Pydantic V2", "Server-Sent Events (SSE)", "ARQ Task Queue"],
    },
    {
      category: "Data & Retrieval",
      items: ["PostgreSQL 16 & pgvector", "BM25 Keyword Engine", "Redis 7 Pub/Sub", "Tesseract OCR"],
    },
    {
      category: "Infrastructure & Observability",
      items: ["Docker & Compose", "OpenTelemetry Tracing", "Prometheus Metrics", "Circuit Breakers"],
    },
  ];

  const technologies = [
    {
      icon: FileText,
      title: "Document Ingestion & Retrieval",
      description:
        "Extracts, parses, and chunks content from PDFs, DOCX files, CSVs, PowerPoints, scanned images (via OCR), and web URLs into high-dimensional vector embeddings and keyword indices.",
    },
    {
      icon: Search,
      title: "Hybrid Search & Reranking",
      description:
        "Fuses dense semantic similarity (pgvector) and sparse keyword frequency (BM25) using Reciprocal Rank Fusion, followed by cross-encoder reranking to ensure high-signal context assembly.",
    },
    {
      icon: Zap,
      title: "Real-Time Generation & Citations",
      description:
        "Streams token-by-token responses from multi-provider LLMs over Server-Sent Events, complete with deterministic chunk-level citations and inline document previews.",
    },
    {
      icon: BarChart3,
      title: "Automatic Evaluation & Improvement",
      description:
        "Asynchronously evaluates every generation across Faithfulness, Relevance, Precision, and Recall using LLM judges, automatically curating top-scoring pairs for SFT and DPO fine-tuning.",
    },
  ];

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 overflow-x-hidden flex flex-col justify-between selection:bg-indigo-600 selection:text-white">
      {/* ── Background Grid & Subtle Ambient Glow ────────────────────── */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(99,102,241,0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(99,102,241,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "64px 64px",
        }}
      />
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.10) 0%, transparent 70%)",
        }}
      />

      {/* ── Main Content Area ─────────────────────────────────────────── */}
      <div className="relative z-10 w-full">
        {/* ── Navigation Header ──────────────────────────────────────── */}
        <nav className="w-full border-b border-slate-800/60 bg-[#0b0f19]/80 backdrop-blur-xl sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-6 sm:px-8 lg:px-12 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 group-hover:scale-105 transition-transform">
                <Cpu className="h-5 w-5" />
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-white text-base tracking-tight leading-none">
                  NeuroFlow
                </span>
                <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                  RAG Lifecycle Platform
                </span>
              </div>
            </Link>

            <div className="flex items-center gap-3 sm:gap-6">
              <div className="hidden sm:flex items-center gap-1">
                <Link
                  href="/"
                  className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors"
                >
                  Overview
                </Link>
                <Link
                  href="/about"
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-600/10 text-indigo-400 border border-indigo-500/30"
                >
                  About
                </Link>
                <Link
                  href="/pipelines"
                  className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors"
                >
                  Pipelines
                </Link>
                <Link
                  href="/evaluations"
                  className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors"
                >
                  Live Feed
                </Link>
              </div>

              <Link
                href="/playground"
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-md shadow-indigo-600/25 hover:shadow-indigo-500/30"
              >
                <Play className="h-3.5 w-3.5" />
                Launch Playground
              </Link>
            </div>
          </div>
        </nav>

        {/* ── Main Container: Wide, Clean, Direct ─────────────────────── */}
        <main className="max-w-6xl mx-auto px-6 sm:px-8 lg:px-12 py-12 sm:py-16 space-y-12 sm:space-y-16">
          {/* ── 1. About NeuroFlow ───────────────────────────────────── */}
          <section className="space-y-4">
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white tracking-tight leading-tight">
              About{" "}
              <span className="text-indigo-400 font-bold">
                NeuroFlow
              </span>
            </h1>

            <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-5xl">
              NeuroFlow is an end-to-end RAG lifecycle and optimization platform designed to turn
              experimental retrieval into observable, measurable intelligence. It unifies multi-format
              document ingestion, hybrid search, real-time generation with exact citation attribution,
              automated LLM-as-judge evaluation, and fine-tuning dataset curation into a single cohesive system.
            </p>
          </section>

          {/* ── 2. Tech Stack ────────────────────────────────────────── */}
          <section className="space-y-5 pt-6 border-t border-slate-800/60">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Tech Stack
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {techStack.map((group) => (
                <div
                  key={group.category}
                  className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-3"
                >
                  <h3 className="font-semibold text-indigo-400 text-xs font-mono uppercase tracking-wider">
                    {group.category}
                  </h3>
                  <ul className="space-y-1.5">
                    {group.items.map((item) => (
                      <li key={item} className="text-xs text-slate-300 font-medium flex items-center gap-1.5">
                        <span className="h-1 w-1 rounded-full bg-slate-600" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          {/* ── 3. Our Technology ────────────────────────────────────── */}
          <section className="space-y-6 pt-6 border-t border-slate-800/60">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Our Technology
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {technologies.map((tech) => (
                <div
                  key={tech.title}
                  className="p-5 sm:p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-2.5 hover:border-slate-700 transition-colors"
                >
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-indigo-950/60 border border-indigo-800/40 text-indigo-400">
                      <tech.icon className="h-4 w-4" />
                    </div>
                    <h3 className="font-semibold text-slate-100 text-sm">{tech.title}</h3>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed pl-1">{tech.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ── 4. Why NeuroFlow ─────────────────────────────────────── */}
          <section className="space-y-4 pt-6 border-t border-slate-800/60">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Why NeuroFlow
            </h2>

            <p className="text-sm sm:text-base text-slate-300 leading-relaxed max-w-5xl">
              NeuroFlow is not just about generating an answer from documents—it makes the entire RAG
              lifecycle observable, measurable, and continuously improvable. While orchestration libraries
              help script prompt chains in code, NeuroFlow provides the operational state, automated
              evaluation telemetry, and A/B benchmarking needed to systematically measure and improve
              retrieval quality in production.
            </p>
          </section>

          {/* ── 5. Built by ──────────────────────────────────────────── */}
          <section className="space-y-4 pt-6 border-t border-slate-800/60">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Built by
            </h2>

            <div className="p-5 sm:p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-5 max-w-5xl hover:border-slate-700/80 transition-colors">
              <div className="flex items-center gap-4">
                <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white font-bold text-sm shadow-md shadow-indigo-500/20 flex-shrink-0">
                  SP
                </div>
                <div className="space-y-0.5">
                  <div className="text-base font-bold text-white leading-none">
                    Sanjana Patil
                  </div>
                  <div className="text-xs text-indigo-400 font-mono mt-1">
                    Creator of NeuroFlow
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2.5 flex-wrap">
                <a
                  href="https://github.com/Sanjana0019"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700/90 text-xs font-semibold text-slate-200 transition-colors border border-slate-700/80"
                >
                  <svg className="h-3.5 w-3.5 fill-current" viewBox="0 0 24 24">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                  </svg>
                  <span>GitHub</span>
                  <ExternalLink className="h-2.5 w-2.5 text-slate-400" />
                </a>

                <a
                  href="https://www.linkedin.com/in/sanjana-patil-dev/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700/90 text-xs font-semibold text-slate-200 transition-colors border border-slate-700/80"
                >
                  <svg className="h-3.5 w-3.5 fill-current text-[#0A66C2]" viewBox="0 0 24 24">
                    <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
                  </svg>
                  <span>LinkedIn</span>
                  <ExternalLink className="h-2.5 w-2.5 text-slate-400" />
                </a>

                <a
                  href="https://x.com/sanjana_p0019"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700/90 text-xs font-semibold text-slate-200 transition-colors border border-slate-700/80"
                >
                  <svg className="h-3.5 w-3.5 fill-current text-white" viewBox="0 0 24 24">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                  <span>X</span>
                  <ExternalLink className="h-2.5 w-2.5 text-slate-400" />
                </a>
              </div>
            </div>
          </section>
        </main>
      </div>

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800/60 py-8 px-6 sm:px-8 lg:px-12 relative z-10 mt-12">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600/80 text-white">
              <Cpu className="h-4 w-4" />
            </div>
            <span className="text-xs font-semibold text-slate-400">NeuroFlow</span>
            <span className="text-slate-700">·</span>
            <span className="text-xs text-slate-600 font-mono">
              End-to-End RAG Lifecycle Platform
            </span>
          </div>

          <div className="text-xs text-slate-500 flex items-center gap-4">
            <Link href="/" className="hover:text-slate-300 transition-colors">
              Home
            </Link>
            <Link href="/about" className="text-indigo-400 font-medium">
              About
            </Link>
            <Link href="/playground" className="hover:text-slate-300 transition-colors">
              Playground
            </Link>
            <Link href="/pipelines" className="hover:text-slate-300 transition-colors">
              Pipelines
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
