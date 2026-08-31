"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import {
  Cpu,
  ArrowRight,
  Play,
  Workflow,
  Activity,
  FileText,
  Sparkles,
  ChevronRight,
  Database,
  Zap,
  BarChart3,
  GitCompare,
  Download,
  Shield,
  Layers,
  Search,
  Bot,
  CheckCircle,
  Circle,
  ArrowDown,
  Code2,
  RefreshCw,
  Eye,
  Target,
  TrendingUp,
  BookOpen,
  Upload,
  ScanSearch,
  ListChecks,
} from "lucide-react";

// ─── Rolling Odometer Digit Column ──────────────────────────────────────────
function RollingDigit({
  targetDigit,
  isRolling,
  delay = 0,
  cycles = 2,
}: {
  targetDigit: number;
  isRolling: boolean;
  delay?: number;
  cycles?: number;
}) {
  // Build a vertical sequence that spins 'cycles' times then stops at targetDigit
  const numbers: number[] = [];
  for (let c = 0; c < cycles; c++) {
    for (let d = 0; d <= 9; d++) {
      numbers.push(d);
    }
  }
  for (let d = 0; d <= targetDigit; d++) {
    numbers.push(d);
  }

  const finalIndex = numbers.length - 1;

  return (
    <span className="inline-block h-[1.15em] overflow-hidden leading-[1.15em] align-top">
      <span
        className="flex flex-col transition-transform duration-[1800ms]"
        style={{
          transform: isRolling
            ? `translateY(-${(finalIndex / numbers.length) * 100}%)`
            : "translateY(0%)",
          transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
          transitionDelay: `${delay}ms`,
        }}
      >
        {numbers.map((num, idx) => (
          <span
            key={idx}
            className="h-[1.15em] flex items-center justify-center font-mono font-extrabold"
          >
            {num}
          </span>
        ))}
      </span>
    </span>
  );
}

function RollingNumber({
  value,
  suffix = "",
}: {
  value: number;
  suffix?: string;
}) {
  const [isRolling, setIsRolling] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsRolling(true);
        } else {
          // Re-arm so it rolls whenever sliding back down
          setIsRolling(false);
        }
      },
      { threshold: 0.25 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const digits = value.toString().split("");

  return (
    <span
      ref={containerRef}
      className="inline-flex items-baseline font-mono font-extrabold select-none tracking-tight"
    >
      {digits.map((ch, idx) => {
        const d = parseInt(ch, 10);
        if (isNaN(d)) {
          return <span key={idx}>{ch}</span>;
        }
        return (
          <RollingDigit
            key={idx}
            targetDigit={d}
            isRolling={isRolling}
            delay={idx * 100}
            cycles={1 + idx}
          />
        );
      })}
      {suffix && <span className="ml-0.5">{suffix}</span>}
    </span>
  );
}

// ─── Pipeline Flow Step ───────────────────────────────────────────────────────
function FlowStep({
  icon: Icon,
  label,
  sublabel,
  color,
  delay,
}: {
  icon: React.ElementType;
  label: string;
  sublabel: string;
  color: string;
  delay: number;
}) {
  return (
    <div
      className="flex flex-col items-center gap-2 group"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div
        className={`w-12 h-12 rounded-2xl flex items-center justify-center border ${color} transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg`}
      >
        <Icon className="h-5 w-5" />
      </div>
      <div className="text-center">
        <div className="text-xs font-semibold text-slate-200">{label}</div>
        <div className="text-[10px] text-slate-500 mt-0.5 font-mono">{sublabel}</div>
      </div>
    </div>
  );
}

// ─── Capability Card ──────────────────────────────────────────────────────────
function CapabilityCard({
  icon: Icon,
  title,
  description,
  badge,
  href,
  accent,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  badge: string;
  href: string;
  accent: string;
}) {
  return (
    <Link
      href={href}
      className="group relative flex flex-col p-6 rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-indigo-500/40 transition-all duration-300 hover:bg-slate-900/80 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-indigo-950/20"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`p-2.5 rounded-xl border ${accent}`}>
          <Icon className="h-5 w-5" />
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
          {badge}
        </span>
      </div>
      <h3 className="font-semibold text-slate-100 text-sm mb-2 group-hover:text-white">
        {title}
      </h3>
      <p className="text-xs text-slate-400 leading-relaxed flex-1">{description}</p>
      <div className="mt-4 flex items-center gap-1.5 text-[11px] font-medium text-indigo-400 group-hover:text-indigo-300">
        Open
        <ArrowRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
      </div>
    </Link>
  );
}

// ─── Terminal Code Block ──────────────────────────────────────────────────────
function TerminalBlock({ lines }: { lines: { type: string; text: string }[] }) {
  return (
    <div className="rounded-xl bg-[#0a0e1a] border border-slate-800 overflow-hidden font-mono text-xs">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-slate-900/80 border-b border-slate-800">
        <div className="h-2.5 w-2.5 rounded-full bg-rose-500/70" />
        <div className="h-2.5 w-2.5 rounded-full bg-amber-500/70" />
        <div className="h-2.5 w-2.5 rounded-full bg-emerald-500/70" />
        <span className="ml-2 text-slate-500 text-[10px]">NeuroFlow · RAG Query</span>
      </div>
      <div className="p-4 space-y-1">
        {lines.map((line, i) => (
          <div key={i} className={
            line.type === "prompt" ? "text-indigo-400" :
            line.type === "output" ? "text-emerald-300" :
            line.type === "metric" ? "text-amber-300" :
            line.type === "comment" ? "text-slate-500" :
            "text-slate-300"
          }>
            {line.text}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Metrics Stat ─────────────────────────────────────────────────────────────
function StatCard({
  value,
  suffix = "",
  unit = "",
  label,
  description,
}: {
  value: number;
  suffix?: string;
  unit?: string;
  label: string;
  description: string;
}) {
  return (
    <div className="flex flex-col items-center justify-between text-center h-full px-2 py-1">
      <div className="flex items-baseline justify-center gap-1.5 whitespace-nowrap">
        <span className="text-3xl sm:text-4xl font-extrabold text-white font-mono tracking-tight">
          <RollingNumber value={value} suffix={suffix} />
        </span>
        {unit && (
          <span className="text-xs sm:text-sm font-semibold text-indigo-400 font-sans uppercase tracking-wider">
            {unit}
          </span>
        )}
      </div>
      <div className="mt-2 space-y-0.5">
        <div className="text-sm font-semibold text-slate-200">{label}</div>
        <div className="text-xs text-slate-400 leading-snug">{description}</div>
      </div>
    </div>
  );
}

// ─── Main Landing Page ────────────────────────────────────────────────────────
export default function LandingPage() {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const pipelineFlow = [
    { icon: Upload, label: "Ingest", sublabel: "PDF · DOCX · URL · Image", color: "bg-violet-950/60 border-violet-800/50 text-violet-400" },
    { icon: Layers, label: "Chunk", sublabel: "Recursive · Semantic", color: "bg-blue-950/60 border-blue-800/50 text-blue-400" },
    { icon: Database, label: "Embed", sublabel: "pgvector · BM25", color: "bg-indigo-950/60 border-indigo-800/50 text-indigo-400" },
    { icon: ScanSearch, label: "Retrieve", sublabel: "Dense + Sparse Hybrid", color: "bg-cyan-950/60 border-cyan-800/50 text-cyan-400" },
    { icon: Target, label: "Rerank", sublabel: "Cross-encoder", color: "bg-teal-950/60 border-teal-800/50 text-teal-400" },
    { icon: Bot, label: "Generate", sublabel: "GPT-4o · Claude · OpenRouter", color: "bg-emerald-950/60 border-emerald-800/50 text-emerald-400" },
    { icon: ListChecks, label: "Evaluate", sublabel: "LLM-as-Judge · 4 metrics", color: "bg-amber-950/60 border-amber-800/50 text-amber-400" },
    { icon: TrendingUp, label: "Improve", sublabel: "Analytics · Fine-tuning", color: "bg-rose-950/60 border-rose-800/50 text-rose-400" },
  ];

  const capabilities = [
    {
      icon: Play,
      title: "Query Playground",
      description: "Ask questions against your knowledge base with real-time streaming responses, inline citations, source inspection, and A/B pipeline comparison.",
      badge: "Verified",
      href: "/playground",
      accent: "bg-indigo-950/60 border-indigo-800/50 text-indigo-400",
    },
    {
      icon: Workflow,
      title: "Pipeline Manager",
      description: "Create, version, and configure retrieval pipelines declaratively. Each pipeline has independent analytics: latency percentiles, quality scores, and 7-day query trends.",
      badge: "Verified",
      href: "/pipelines",
      accent: "bg-violet-950/60 border-violet-800/50 text-violet-400",
    },
    {
      icon: Activity,
      title: "Live Evaluation Feed",
      description: "Watch every query get automatically scored in real-time via LLM-as-judge. Faithfulness, answer relevance, context precision, and context recall — streamed over SSE.",
      badge: "Verified",
      href: "/evaluations",
      accent: "bg-emerald-950/60 border-emerald-800/50 text-emerald-400",
    },
    {
      icon: FileText,
      title: "Document Ingestion",
      description: "Upload PDFs, DOCX files, CSVs, PowerPoints, images (with OCR), and URLs. Documents are chunked, embedded, and stored in PostgreSQL with pgvector.",
      badge: "Verified",
      href: "/documents",
      accent: "bg-amber-950/60 border-amber-800/50 text-amber-400",
    },
    {
      icon: Sparkles,
      title: "Fine-Tuning & Datasets",
      description: "NeuroFlow automatically extracts high-quality training pairs from evaluation history. Review, inspect, and export datasets in JSONL format for SFT and DPO training.",
      badge: "Infrastructure",
      href: "/finetuning",
      accent: "bg-rose-950/60 border-rose-800/50 text-rose-400",
    },
  ];

  const terminalLines = [
    { type: "comment", text: "# Query your knowledge base" },
    { type: "prompt", text: "POST /query { pipeline: 'production-hybrid-rag' }" },
    { type: "output", text: "" },
    { type: "output", text: '> "Based on your technical documents, the architecture..."' },
    { type: "output", text: '  [1] document_a.pdf · page 12' },
    { type: "output", text: '  [2] architecture_spec.docx · page 4' },
    { type: "output", text: "" },
    { type: "metric", text: "  Latency: 1.48s  Faithfulness: 0.91  Relevance: 0.88" },
    { type: "comment", text: "# Automatically logged → Evaluation Feed" },
  ];

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 overflow-x-hidden">
      {/* ── Subtle animated background grid ─────────────────────────── */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(99,102,241,0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(99,102,241,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "64px 64px",
          transform: `translateY(${scrollY * 0.1}px)`,
        }}
      />
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.10) 0%, transparent 70%)",
        }}
      />

      {/* ── Landing Navigation ───────────────────────────────────────── */}
      <nav className="relative z-50 w-full border-b border-slate-800/60 bg-[#0b0f19]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-600/30">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <span className="font-bold text-white text-lg tracking-tight">NeuroFlow</span>
              <span className="hidden sm:inline text-slate-400 text-xs font-mono ml-2">v1.0 · RAG Lifecycle Platform</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-1 mr-2">
              <Link
                href="/"
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-600/10 text-indigo-400 border border-indigo-500/30"
              >
                Overview
              </Link>
              <Link
                href="/about"
                className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors"
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

            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/40 text-emerald-400 text-xs font-mono">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Live
            </div>
            <Link
              href="/playground"
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition-all shadow-lg shadow-indigo-600/25 hover:shadow-indigo-500/30"
            >
              <Play className="h-3.5 w-3.5" />
              Try Now
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ─────────────────────────────────────────────── */}
      <section className="relative pt-20 pb-16 sm:pt-28 sm:pb-24 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto text-center space-y-8">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-indigo-500/30 bg-indigo-950/50 text-indigo-300 text-xs font-mono">
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
            End-to-End RAG Lifecycle Platform
            <ChevronRight className="h-3 w-3 text-indigo-500" />
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-tight tracking-tight">
            Your Knowledge Base.
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-violet-400 to-indigo-400">
              Retrieval that actually works.
            </span>
          </h1>

          {/* Subheadline */}
          <p className="max-w-2xl mx-auto text-base sm:text-lg text-slate-400 leading-relaxed">
            NeuroFlow is a complete RAG infrastructure platform — from document ingestion to LLM generation, 
            automatic quality evaluation, pipeline versioning, A/B comparison, and training data extraction.
            <span className="text-slate-300"> One system. Every step observed.</span>
          </p>

          {/* CTA Row */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
            <Link
              href="/playground"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-xl shadow-indigo-600/30 hover:shadow-indigo-500/40 hover:-translate-y-0.5"
            >
              <Play className="h-4 w-4" />
              Open Query Playground
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
            <Link
              href="/pipelines"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-6 py-3 rounded-xl border border-slate-700 hover:border-slate-500 text-slate-300 hover:text-white font-semibold text-sm transition-all hover:bg-slate-800/50"
            >
              <Workflow className="h-4 w-4" />
              Pipeline Manager
            </Link>
          </div>

          {/* Scroll hint */}
          <div className="flex flex-col items-center gap-2 pt-6 text-slate-600 text-xs animate-bounce">
            <ArrowDown className="h-4 w-4" />
          </div>
        </div>
      </section>

      {/* ── Quick Stats ──────────────────────────────────────────────── */}
      <section className="relative py-8 sm:py-12 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-4 p-6 sm:p-8 rounded-2xl bg-slate-900/50 border border-slate-800/80 backdrop-blur-xl shadow-xl shadow-black/20">
            <div className="flex flex-col items-center">
              <StatCard
                value={7}
                unit="formats"
                label="Document Ingest"
                description="PDF, DOCX, CSV, Image OCR, URL…"
              />
            </div>
            <div className="flex flex-col items-center">
              <StatCard
                value={4}
                unit="metrics"
                label="Auto-Evaluation"
                description="Faithfulness, Relevance, Precision, Recall"
              />
            </div>
            <div className="flex flex-col items-center">
              <StatCard
                value={3}
                unit="providers"
                label="LLM Backends"
                description="OpenAI, Anthropic, OpenRouter"
              />
            </div>
            <div className="flex flex-col items-center">
              <StatCard
                value={100}
                suffix="%"
                label="Observability"
                description="Every query tracked & evaluated"
              />
            </div>
          </div>
        </div>
      </section>

      {/* ── The RAG Lifecycle ─────────────────────────────────────────── */}
      <section className="relative py-20 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto space-y-12">
          {/* Section header */}
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-mono text-indigo-400 bg-indigo-950/50 border border-indigo-800/40 px-3 py-1 rounded-full">
              <RefreshCw className="h-3 w-3" />
              The Complete Lifecycle
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              Not a chatbot wrapper. A full pipeline.
            </h2>
            <p className="max-w-xl mx-auto text-sm text-slate-400">
              Most RAG applications glue an LLM to a vector database and stop there. 
              NeuroFlow treats retrieval quality as something you measure, compare, and systematically improve.
            </p>
          </div>

          {/* Flow diagram */}
          <div className="relative overflow-x-auto">
            <div className="flex items-start justify-start lg:justify-center gap-3 min-w-max lg:min-w-0 mx-auto pb-4 px-4">
              {pipelineFlow.map((step, i) => (
                <div key={step.label} className="flex items-start gap-3">
                  <FlowStep {...step} delay={i * 80} />
                  {i < pipelineFlow.length - 1 && (
                    <div className="mt-5 text-slate-700 flex-shrink-0">
                      <ChevronRight className="h-4 w-4" />
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Feedback loop indicator */}
            <div className="mt-6 flex justify-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
                <RefreshCw className="h-3 w-3 text-indigo-400" />
                Every query feeds back into analytics, training data, and pipeline improvement
              </div>
            </div>
          </div>

          {/* Three big ideas */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6">
            {[
              {
                icon: Eye,
                title: "Total observability",
                body: "Every query, every response, every evaluation score is stored in PostgreSQL. Nothing is a black box. You know exactly what was retrieved, how it was ranked, how the answer was generated, and how it scored.",
                color: "text-indigo-400",
                bg: "bg-indigo-950/40 border-indigo-800/30",
              },
              {
                icon: GitCompare,
                title: "Pipeline A/B comparison",
                body: "Run the same query through two different pipeline configurations simultaneously. Compare latency, answer quality, faithfulness scores, and retrieved context side-by-side. Configuration-driven, version-controlled.",
                color: "text-violet-400",
                bg: "bg-violet-950/40 border-violet-800/30",
              },
              {
                icon: TrendingUp,
                title: "Systematic improvement",
                body: "High-quality query responses automatically become training pairs. Review them, export JSONL datasets for SFT or DPO fine-tuning, and close the loop between evaluation quality and model improvement.",
                color: "text-emerald-400",
                bg: "bg-emerald-950/40 border-emerald-800/30",
              },
            ].map((item) => (
              <div
                key={item.title}
                className={`p-6 rounded-2xl border ${item.bg} space-y-3`}
              >
                <div className={`${item.color}`}>
                  <item.icon className="h-6 w-6" />
                </div>
                <h3 className="font-semibold text-white text-sm">{item.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{item.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Live Demo Section ─────────────────────────────────────────── */}
      <section className="relative py-20 px-4 sm:px-6 bg-gradient-to-b from-transparent to-slate-900/20">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left: explanation */}
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/30 px-3 py-1 rounded-full">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Live System
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white">
                Ask a question.
                <br />
                <span className="text-slate-400">Watch everything happen.</span>
              </h2>
              <p className="text-sm text-slate-400 leading-relaxed">
                When you run a query in NeuroFlow, you see the full picture — not just the answer. 
                The sources that were retrieved, the chunks that were ranked, the score the LLM judge assigned, 
                and the cost in tokens and latency.
              </p>
              <div className="space-y-3">
                {[
                  { icon: Zap, text: "Real-time streaming over Server-Sent Events", color: "text-indigo-400" },
                  { icon: BookOpen, text: "Inline citations with source document preview", color: "text-violet-400" },
                  { icon: ScanSearch, text: "Retrieval inspector: see exactly what was fetched", color: "text-cyan-400" },
                  { icon: BarChart3, text: "LLM judge score per query, streamed automatically", color: "text-amber-400" },
                ].map((item) => (
                  <div key={item.text} className="flex items-start gap-3 text-sm">
                    <item.icon className={`h-4 w-4 mt-0.5 flex-shrink-0 ${item.color}`} />
                    <span className="text-slate-300">{item.text}</span>
                  </div>
                ))}
              </div>
              <Link
                href="/playground"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/25"
              >
                <Play className="h-3.5 w-3.5" />
                Open Playground
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>

            {/* Right: terminal mock */}
            <div className="space-y-4">
              <TerminalBlock lines={terminalLines} />

              {/* Mini evaluation scorecard */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                <div className="text-[11px] font-mono text-slate-500 uppercase">Automatic LLM Evaluation</div>
                {[
                  { label: "Faithfulness", value: 0.91, color: "bg-emerald-500" },
                  { label: "Answer Relevance", value: 0.88, color: "bg-indigo-500" },
                  { label: "Context Precision", value: 0.84, color: "bg-violet-500" },
                  { label: "Context Recall", value: 0.79, color: "bg-amber-500" },
                ].map((m) => (
                  <div key={m.label} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300">{m.label}</span>
                      <span className="font-mono text-slate-300">{(m.value * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-slate-800">
                      <div
                        className={`h-full rounded-full ${m.color}`}
                        style={{ width: `${m.value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
                <div className="text-[10px] text-slate-600 font-mono pt-1">
                  Note: Scores shown are illustrative. Your data will reflect actual evaluation results.
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Capabilities Grid ─────────────────────────────────────────── */}
      <section className="relative py-20 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto space-y-10">
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900/60 border border-slate-800 px-3 py-1 rounded-full">
              <Layers className="h-3 w-3 text-indigo-400" />
              Five Integrated Modules
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              Every module connects to the next
            </h2>
            <p className="max-w-lg mx-auto text-sm text-slate-400">
              These aren't isolated tools. Each module feeds data to the others. 
              Documents become chunks. Chunks enable retrieval. Retrieval enables generation. 
              Generation enables evaluation. Evaluation enables improvement.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {capabilities.map((cap) => (
              <CapabilityCard key={cap.href} {...cap} />
            ))}
            {/* Sixth card: Architecture overview */}
            <div className="relative flex flex-col p-6 rounded-2xl bg-gradient-to-br from-indigo-950/40 to-slate-900/60 border border-indigo-800/30 sm:col-span-2 lg:col-span-1">
              <div className="flex items-start justify-between mb-4">
                <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 text-slate-400">
                  <Shield className="h-5 w-5" />
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                  Infrastructure
                </span>
              </div>
              <h3 className="font-semibold text-slate-100 text-sm mb-2">Production-Grade Backend</h3>
              <p className="text-xs text-slate-400 leading-relaxed flex-1">
                FastAPI + PostgreSQL + Redis + ARQ background workers. OpenTelemetry tracing. 
                Prometheus metrics. Circuit breaker resilience. Rate limiting per endpoint. 
                Schema migrations. Backpressure control.
              </p>
              <div className="mt-4 flex flex-wrap gap-1.5">
                {["FastAPI", "PostgreSQL", "pgvector", "Redis", "ARQ", "OTEL"].map((t) => (
                  <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-500">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Pipeline Manager Callout ──────────────────────────────────── */}
      <section className="relative py-20 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/40 p-8 sm:p-12">
            {/* Background accent */}
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-950/20 via-transparent to-violet-950/10 pointer-events-none" />
            <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-600/5 rounded-full blur-3xl pointer-events-none" />

            <div className="relative grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
              <div className="space-y-6">
                <div className="inline-flex items-center gap-2 text-xs font-mono text-violet-400 bg-violet-950/40 border border-violet-800/30 px-3 py-1 rounded-full">
                  <Workflow className="h-3 w-3" />
                  Config-Driven Pipelines
                </div>
                <h2 className="text-2xl sm:text-3xl font-bold text-white">
                  Different retrievals.<br />
                  <span className="text-slate-400">Same question.</span>
                </h2>
                <p className="text-sm text-slate-400 leading-relaxed">
                  Not all retrieval strategies work equally well for all documents. 
                  NeuroFlow lets you define multiple pipeline configurations with different 
                  chunking strategies, retrieval depths, rerankers, and LLM models — 
                  then compare them on real queries to see which actually performs better.
                </p>
                <div className="space-y-2.5">
                  {[
                    "Version-controlled pipeline configurations",
                    "Real-time latency percentiles (P50 / P95 / P99)",
                    "Per-pipeline quality score tracking",
                    "A/B comparison with evaluation scorecard",
                  ].map((item) => (
                    <div key={item} className="flex items-center gap-2.5 text-sm">
                      <CheckCircle className="h-4 w-4 text-indigo-400 flex-shrink-0" />
                      <span className="text-slate-300">{item}</span>
                    </div>
                  ))}
                </div>
                <Link
                  href="/pipelines"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-indigo-500/40 hover:bg-indigo-950/40 text-indigo-300 hover:text-indigo-200 font-semibold text-sm transition-all"
                >
                  <Workflow className="h-3.5 w-3.5" />
                  View Pipeline Manager
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>

              {/* Pipeline comparison mock */}
              <div className="space-y-3">
                {[
                  { name: "Production-Hybrid-RAG", queries: 37, latency: "1.48s", score: "49%", badge: "active" },
                  { name: "Fast-Dense-Search", queries: 24, latency: "1.58s", score: "50%", badge: "active" },
                  { name: "deep-research-rag", queries: 6, latency: "134ms", score: "68%", badge: "active" },
                  { name: "Zero-Runs-Pipeline", queries: 0, latency: "—", score: "N/A", badge: "no data" },
                ].map((p) => (
                  <div
                    key={p.name}
                    className="flex items-center justify-between p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 gap-4"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className="p-1.5 rounded-lg bg-indigo-950/60 border border-indigo-800/40 text-indigo-400 flex-shrink-0">
                        <Workflow className="h-3 w-3" />
                      </div>
                      <span className="text-xs font-mono text-slate-300 truncate">{p.name}</span>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0 text-[11px] font-mono">
                      <span className="text-slate-500">{p.queries}q</span>
                      <span className="text-slate-500">{p.latency}</span>
                      <span
                        className={`px-1.5 py-0.5 rounded-full border font-semibold ${
                          p.score === "N/A"
                            ? "bg-slate-800 text-slate-500 border-slate-700"
                            : "bg-indigo-950/60 text-indigo-300 border-indigo-800/40"
                        }`}
                      >
                        {p.score}
                      </span>
                    </div>
                  </div>
                ))}
                <div className="text-[10px] text-slate-600 font-mono text-center pt-1">
                  * Showing actual pipeline data from local PostgreSQL database
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Evaluation & Fine-Tuning ──────────────────────────────────── */}
      <section className="relative py-20 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Evaluation */}
          <div className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-6">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-800/40 text-emerald-400">
                <Activity className="h-5 w-5" />
              </div>
              <div>
                <h3 className="font-bold text-white">Automatic LLM Evaluation</h3>
                <p className="text-xs text-slate-500 font-mono">Every response. Always scored.</p>
              </div>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">
              After every generation, NeuroFlow automatically runs an LLM-as-judge evaluation 
              using 4 RAGAS-inspired metrics. Results stream in real-time to the evaluation feed 
              via Redis pub/sub and Server-Sent Events.
            </p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { name: "Faithfulness", desc: "Factual consistency with retrieved context" },
                { name: "Answer Relevance", desc: "Direct relevance to the query intent" },
                { name: "Context Precision", desc: "Signal-to-noise of retrieved chunks" },
                { name: "Context Recall", desc: "Completeness of ground-truth retrieval" },
              ].map((m) => (
                <div key={m.name} className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <CheckCircle className="h-3 w-3 text-emerald-400" />
                    <span className="text-xs font-semibold text-slate-200">{m.name}</span>
                  </div>
                  <p className="text-[10px] text-slate-500">{m.desc}</p>
                </div>
              ))}
            </div>
            <Link
              href="/evaluations"
              className="inline-flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 font-medium transition-colors"
            >
              <Activity className="h-3.5 w-3.5" />
              View Live Evaluation Feed
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          {/* Fine-tuning */}
          <div className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-6">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-rose-950/60 border border-rose-800/40 text-rose-400">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <h3 className="font-bold text-white">Training Data Extraction</h3>
                <p className="text-xs text-slate-500 font-mono">Turn good answers into better models.</p>
              </div>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">
              NeuroFlow monitors every evaluation result and automatically identifies 
              high-quality query-response pairs suitable for fine-tuning. No manual labeling required. 
              Export datasets in JSONL format for SFT or DPO training.
            </p>
            <div className="space-y-3">
              {[
                { step: "01", text: "Queries evaluated automatically by LLM judge", done: true },
                { step: "02", text: "High-scoring pairs collected as training candidates", done: true },
                { step: "03", text: "Review and inspect training pairs", done: true },
                { step: "04", text: "Export JSONL dataset for SFT / DPO", done: true },
                { step: "05", text: "Launch fine-tune job (OpenAI API integration)", done: false },
              ].map((item) => (
                <div key={item.step} className="flex items-start gap-3 text-sm">
                  {item.done ? (
                    <CheckCircle className="h-4 w-4 text-rose-400 flex-shrink-0 mt-0.5" />
                  ) : (
                    <Circle className="h-4 w-4 text-slate-700 flex-shrink-0 mt-0.5" />
                  )}
                  <span className={item.done ? "text-slate-300" : "text-slate-600"}>
                    {item.text}
                  </span>
                </div>
              ))}
            </div>
            <Link
              href="/finetuning"
              className="inline-flex items-center gap-2 text-sm text-rose-400 hover:text-rose-300 font-medium transition-colors"
            >
              <Sparkles className="h-3.5 w-3.5" />
              View Dataset Dashboard
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </section>

      {/* ── Architecture Overview ─────────────────────────────────────── */}
      <section className="relative py-20 px-4 sm:px-6 bg-gradient-to-b from-transparent to-slate-900/30">
        <div className="max-w-5xl mx-auto space-y-10">
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900/60 border border-slate-800 px-3 py-1 rounded-full">
              <Code2 className="h-3 w-3 text-indigo-400" />
              Technical Architecture
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              Built for real-world complexity
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                category: "Storage",
                items: ["PostgreSQL + pgvector", "Redis pub/sub + queue", "ARQ background workers", "Document chunk store"],
                icon: Database,
                color: "text-blue-400",
                border: "border-blue-800/30 bg-blue-950/20",
              },
              {
                category: "Retrieval",
                items: ["Dense vector (cosine)", "BM25 sparse (keyword)", "Reciprocal Rank Fusion", "Cross-encoder reranking"],
                icon: Search,
                color: "text-violet-400",
                border: "border-violet-800/30 bg-violet-950/20",
              },
              {
                category: "Generation",
                items: ["Streaming SSE tokens", "Citation extraction", "Multi-provider routing", "System prompt variants"],
                icon: Bot,
                color: "text-emerald-400",
                border: "border-emerald-800/30 bg-emerald-950/20",
              },
              {
                category: "Observability",
                items: ["OpenTelemetry tracing", "Prometheus /metrics", "Rate limiting", "Circuit breakers"],
                icon: BarChart3,
                color: "text-amber-400",
                border: "border-amber-800/30 bg-amber-950/20",
              },
            ].map((col) => (
              <div key={col.category} className={`p-5 rounded-2xl border ${col.border} space-y-4`}>
                <div className="flex items-center gap-2">
                  <col.icon className={`h-4 w-4 ${col.color}`} />
                  <span className="text-xs font-semibold text-slate-300">{col.category}</span>
                </div>
                <ul className="space-y-2">
                  {col.items.map((item) => (
                    <li key={item} className="text-xs text-slate-500 flex items-start gap-1.5">
                      <span className="text-slate-700 mt-1">·</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Final CTA ─────────────────────────────────────────────────── */}
      <section className="relative py-24 px-4 sm:px-6">
        <div className="max-w-3xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Ready to see it work?
            </h2>
            <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
              Upload a document, run a query, and watch NeuroFlow retrieve, generate, cite, and evaluate — 
              all in one continuous loop. No setup required. The system is live.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/documents"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-xl border border-slate-700 hover:border-indigo-500/50 text-slate-300 hover:text-white font-semibold text-sm transition-all hover:bg-slate-800/40"
            >
              <FileText className="h-4 w-4" />
              Add Your Documents
            </Link>
            <Link
              href="/playground"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm transition-all shadow-xl shadow-indigo-600/30 hover:shadow-indigo-500/40 hover:-translate-y-0.5"
            >
              <Play className="h-4 w-4" />
              Open Query Playground
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          {/* App navigation quick links */}
          <div className="pt-4">
            <p className="text-xs text-slate-600 mb-4">Or jump directly to a module:</p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {[
                { label: "Query Playground", href: "/playground", icon: Play },
                { label: "Pipeline Manager", href: "/pipelines", icon: Workflow },
                { label: "Evaluation Feed", href: "/evaluations", icon: Activity },
                { label: "Documents", href: "/documents", icon: FileText },
                { label: "Fine-Tuning", href: "/finetuning", icon: Sparkles },
              ].map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-500 hover:text-slate-300 hover:bg-slate-800/60 border border-transparent hover:border-slate-700 transition-all"
                >
                  <link.icon className="h-3 w-3" />
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800/60 py-10 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600/80 text-white">
              <Cpu className="h-4 w-4" />
            </div>
            <span className="text-sm font-semibold text-slate-400">NeuroFlow</span>
            <span className="text-slate-700">·</span>
            <span className="text-xs text-slate-600 font-mono">End-to-End RAG Lifecycle Platform</span>
          </div>
          <div className="text-xs text-slate-700 font-mono">
            Built with FastAPI · Next.js · PostgreSQL · pgvector · Redis
          </div>
        </div>
      </footer>
    </div>
  );
}
