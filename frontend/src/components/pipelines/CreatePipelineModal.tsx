"use client";

import { useState, useMemo, useEffect } from "react";
import Editor from "@monaco-editor/react";
import {
  X,
  Code,
  CheckCircle,
  AlertTriangle,
  Loader2,
  Workflow,
  Zap,
  Layers,
  Sparkles,
  Sliders,
  Database,
  ArrowDown,
  Bot,
  Gauge,
  HelpCircle,
  FileText,
  FileCode,
  ArrowDownUp,
  Scale,
} from "lucide-react";
import { createPipeline } from "../../lib/api";
import { PipelineConfig } from "../../types";

interface CreatePipelineModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export type PresetKey = "balanced_hybrid" | "fast_dense" | "deep_research";

export interface PresetOption {
  key: PresetKey;
  name: string;
  badge: string;
  badgeColor: string;
  icon: typeof Workflow;
  description: string;
  config: PipelineConfig;
}

export const PRESETS: Record<PresetKey, PresetOption> = {
  balanced_hybrid: {
    key: "balanced_hybrid",
    name: "Balanced Hybrid RAG",
    badge: "Recommended",
    badgeColor: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40",
    icon: Layers,
    description: "Production-grade multi-stage retrieval combining dense vector search, BM25 keyword matching, and cross-encoder reranking.",
    config: {
      name: "enterprise-hybrid-rag",
      description: "Production enterprise retrieval-augmented generation pipeline with hybrid search.",
      ingestion: {
        chunking_strategy: "recursive",
        chunk_size_tokens: 512,
        chunk_overlap_tokens: 64,
        extractors_enabled: ["pdf", "docx", "text"],
      },
      retrieval: {
        dense_k: 20,
        sparse_k: 15,
        reranker: "bge-reranker-large",
        top_k_after_rerank: 5,
        query_expansion: true,
        metadata_filters_enabled: true,
      },
      generation: {
        model_routing: {
          task_type: "rag_generation",
          max_cost_per_call: 0.05,
        },
        max_context_tokens: 4000,
        temperature: 0.7,
        system_prompt_variant: "enterprise_qa",
      },
      evaluation: {
        auto_evaluate: true,
        training_threshold: 0.85,
      },
    },
  },
  fast_dense: {
    key: "fast_dense",
    name: "Fast Dense Search",
    badge: "Low Latency",
    badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    icon: Zap,
    description: "Optimized for ultra-fast response times using pure pgvector similarity search without heavy reranking overhead.",
    config: {
      name: "fast-dense-search",
      description: "High-speed dense vector search pipeline for rapid question-answering.",
      ingestion: {
        chunking_strategy: "recursive",
        chunk_size_tokens: 512,
        chunk_overlap_tokens: 32,
        extractors_enabled: ["pdf", "docx", "text"],
      },
      retrieval: {
        dense_k: 10,
        sparse_k: 5,
        reranker: "none",
        top_k_after_rerank: 5,
        query_expansion: false,
        metadata_filters_enabled: false,
      },
      generation: {
        model_routing: {
          task_type: "rag_generation",
          max_cost_per_call: 0.02,
        },
        max_context_tokens: 2048,
        temperature: 0.3,
        system_prompt_variant: "concise",
      },
      evaluation: {
        auto_evaluate: true,
        training_threshold: 0.80,
      },
    },
  },
  deep_research: {
    key: "deep_research",
    name: "Deep Research & Accuracy",
    badge: "High Precision",
    badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/40",
    icon: Sparkles,
    description: "Maximum accuracy with larger token context windows, deep candidate retrieval, and rigorous quality evaluation thresholds.",
    config: {
      name: "deep-research-rag",
      description: "Comprehensive research pipeline with expanded context window and deep cross-encoder reranking.",
      ingestion: {
        chunking_strategy: "recursive",
        chunk_size_tokens: 1024,
        chunk_overlap_tokens: 128,
        extractors_enabled: ["pdf", "docx", "text", "csv", "pptx"],
      },
      retrieval: {
        dense_k: 30,
        sparse_k: 25,
        reranker: "bge-reranker-large",
        top_k_after_rerank: 10,
        query_expansion: true,
        metadata_filters_enabled: true,
      },
      generation: {
        model_routing: {
          task_type: "rag_generation",
          max_cost_per_call: 0.10,
        },
        max_context_tokens: 8000,
        temperature: 0.5,
        system_prompt_variant: "comprehensive_research",
      },
      evaluation: {
        auto_evaluate: true,
        training_threshold: 0.90,
      },
    },
  },
};

export function CreatePipelineModal({ isOpen, onClose, onSuccess }: CreatePipelineModalProps) {
  // Tabs: "visual" | "json"
  const [activeTab, setActiveTab] = useState<"visual" | "json">("visual");

  // Selected preset key
  const [selectedPreset, setSelectedPreset] = useState<PresetKey>("balanced_hybrid");

  // Form State
  const [pipelineName, setPipelineName] = useState("my-custom-rag-pipeline");
  const [pipelineDescription, setPipelineDescription] = useState(
    "Custom RAG pipeline tailored for domain question-answering."
  );
  const [retrievalStrategy, setRetrievalStrategy] = useState<"hybrid" | "dense">("hybrid");
  const [denseK, setDenseK] = useState(20);
  const [sparseK, setSparseK] = useState(15);
  const [topKAfterRerank, setTopKAfterRerank] = useState(5);
  const [enableReranker, setEnableReranker] = useState(true);
  const [enableQueryExpansion, setEnableQueryExpansion] = useState(true);
  const [contextTokenBudget, setContextTokenBudget] = useState(4000);
  const [temperature, setTemperature] = useState(0.7);
  const [systemPromptVariant, setSystemPromptVariant] = useState("enterprise_qa");
  const [autoEvaluate, setAutoEvaluate] = useState(true);
  const [trainingThreshold, setTrainingThreshold] = useState(0.85);

  // Advanced JSON text
  const [jsonText, setJsonText] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Apply preset to form state
  const applyPreset = (presetKey: PresetKey) => {
    setSelectedPreset(presetKey);
    const p = PRESETS[presetKey].config;
    setPipelineName(p.name);
    setPipelineDescription(p.description || "");
    const isHybrid = (p.retrieval.sparse_k ?? 0) > 0;
    setRetrievalStrategy(isHybrid ? "hybrid" : "dense");
    setDenseK(p.retrieval.dense_k);
    setSparseK(p.retrieval.sparse_k);
    setTopKAfterRerank(p.retrieval.top_k_after_rerank);
    setEnableReranker(p.retrieval.reranker !== "none");
    setEnableQueryExpansion(p.retrieval.query_expansion);
    setContextTokenBudget(p.generation.max_context_tokens);
    setTemperature(p.generation.temperature);
    setSystemPromptVariant(p.generation.system_prompt_variant);
    setAutoEvaluate(p.evaluation.auto_evaluate);
    setTrainingThreshold(p.evaluation.training_threshold);
    setError(null);
  };

  // Build canonical config from current form state
  const currentConfig: PipelineConfig = useMemo(() => {
    const isHybrid = retrievalStrategy === "hybrid";
    return {
      name: pipelineName.trim() || "unnamed-pipeline",
      description: pipelineDescription.trim() || undefined,
      ingestion: {
        chunking_strategy: "recursive",
        chunk_size_tokens: selectedPreset === "deep_research" ? 1024 : 512,
        chunk_overlap_tokens: selectedPreset === "deep_research" ? 128 : 64,
        extractors_enabled:
          selectedPreset === "deep_research"
            ? ["pdf", "docx", "text", "csv", "pptx"]
            : ["pdf", "docx", "text"],
      },
      retrieval: {
        dense_k: Number(denseK) || 20,
        sparse_k: isHybrid ? Number(sparseK) || 15 : 5,
        reranker: enableReranker ? "bge-reranker-large" : "none",
        top_k_after_rerank: Number(topKAfterRerank) || 5,
        query_expansion: Boolean(enableQueryExpansion),
        metadata_filters_enabled: isHybrid,
      },
      generation: {
        model_routing: {
          task_type: "rag_generation",
          max_cost_per_call: selectedPreset === "deep_research" ? 0.1 : selectedPreset === "fast_dense" ? 0.02 : 0.05,
        },
        max_context_tokens: Number(contextTokenBudget) || 4000,
        temperature: Number(temperature) || 0.7,
        system_prompt_variant: systemPromptVariant,
      },
      evaluation: {
        auto_evaluate: Boolean(autoEvaluate),
        training_threshold: Number(trainingThreshold) || 0.85,
      },
    };
  }, [
    pipelineName,
    pipelineDescription,
    retrievalStrategy,
    denseK,
    sparseK,
    topKAfterRerank,
    enableReranker,
    enableQueryExpansion,
    contextTokenBudget,
    temperature,
    systemPromptVariant,
    autoEvaluate,
    trainingThreshold,
    selectedPreset,
  ]);

  // Keep jsonText synced when form state changes (if user is not currently hand-editing JSON)
  useEffect(() => {
    setJsonText(JSON.stringify(currentConfig, null, 2));
  }, [currentConfig]);

  if (!isOpen) return null;

  const handleSubmit = async () => {
    setError(null);
    let payloadToSubmit: PipelineConfig;

    if (activeTab === "json") {
      try {
        payloadToSubmit = JSON.parse(jsonText);
      } catch (e: any) {
        setError(`Invalid JSON Syntax: ${e.message}`);
        return;
      }
    } else {
      payloadToSubmit = currentConfig;
    }

    // Client-side validation
    if (!payloadToSubmit.name || payloadToSubmit.name.trim().length === 0) {
      setError("Validation Error: Pipeline name cannot be empty.");
      return;
    }

    setIsSubmitting(true);
    try {
      await createPipeline(payloadToSubmit);
      onSuccess();
      onClose();
    } catch (err: any) {
      const respData = err.response?.data;
      if (respData?.detail) {
        if (typeof respData.detail === "string" && respData.detail.toLowerCase().includes("already exists")) {
          setError(`A pipeline with the name "${payloadToSubmit.name}" already exists. Please choose a different unique name.`);
        } else if (Array.isArray(respData.detail)) {
          const formatted = respData.detail
            .map((d: any) => `• ${d.loc ? d.loc.join(" -> ") : "Field"}: ${d.msg}`)
            .join("\n");
          setError(`Configuration Error:\n${formatted}`);
        } else if (typeof respData.detail === "string") {
          setError(respData.detail);
        } else {
          setError(JSON.stringify(respData.detail, null, 2));
        }
      } else {
        setError(err.message || "Failed to create pipeline");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div
        className="w-full max-w-4xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl flex flex-col max-h-[92vh] overflow-hidden animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-indigo-950/80 border border-indigo-800/40 text-indigo-400">
              <Workflow className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
                Create RAG Pipeline
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800/50">
                  v1.0
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Choose a ready-to-use preset or customize your retrieval and generation recipe
              </p>
            </div>
          </div>

          {/* Mode Switch Tabs & Close Button */}
          <div className="flex items-center gap-3">
            <div className="flex items-center p-1 rounded-xl bg-slate-900 border border-slate-800 text-xs">
              <button
                type="button"
                onClick={() => setActiveTab("visual")}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === "visual"
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Visual Builder
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("json")}
                className={`px-3 py-1 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                  activeTab === "json"
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Code className="h-3.5 w-3.5" />
                Advanced JSON
              </button>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 max-h-[calc(92vh-130px)]">
          {activeTab === "visual" ? (
            <>
              {/* Step 1: Presets Selection */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="h-4 w-4 text-indigo-400" />
                    1. Choose Starting Preset
                  </label>
                  <span className="text-[11px] text-slate-500">Presets auto-fill tested parameters</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {(Object.values(PRESETS) as PresetOption[]).map((preset) => {
                    const isSelected = selectedPreset === preset.key;
                    const Icon = preset.icon;
                    return (
                      <div
                        key={preset.key}
                        onClick={() => applyPreset(preset.key)}
                        className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                          isSelected
                            ? "bg-indigo-950/40 border-indigo-500 ring-1 ring-indigo-500/50 shadow-lg shadow-indigo-950/30"
                            : "bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60"
                        }`}
                      >
                        <div>
                          <div className="flex items-start justify-between gap-2 mb-2">
                            <div className="flex items-center gap-2">
                              <div
                                className={`p-1.5 rounded-lg border ${
                                  isSelected
                                    ? "bg-indigo-600 text-white border-indigo-400"
                                    : "bg-slate-800 text-slate-400 border-slate-700"
                                }`}
                              >
                                <Icon className="h-4 w-4" />
                              </div>
                              <span className="text-xs font-semibold text-slate-100">{preset.name}</span>
                            </div>
                            <span
                              className={`text-[9px] font-semibold px-2 py-0.5 rounded-full border ${preset.badgeColor}`}
                            >
                              {preset.badge}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 leading-relaxed">{preset.description}</p>
                        </div>

                        <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-400">
                          <span>
                            {preset.key === "fast_dense" ? "Dense Only" : "Dense + BM25"}
                          </span>
                          <span>{preset.config.retrieval.top_k_after_rerank} chunks</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Step 2: Basic Configuration Form */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                {/* Left Column: Pipeline Identity & Retrieval Strategy */}
                <div className="space-y-4">
                  <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Sliders className="h-4 w-4 text-indigo-400" />
                    2. Pipeline Identity
                  </label>

                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1">
                      Pipeline Name <span className="text-rose-400">*</span>
                    </label>
                    <input
                      type="text"
                      value={pipelineName}
                      onChange={(e) => setPipelineName(e.target.value)}
                      placeholder="e.g. enterprise-hybrid-rag"
                      className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-100 text-xs font-mono focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    />
                    <span className="text-[10px] text-slate-500 mt-0.5 block">
                      Unique identifier used to route queries in the Playground.
                    </span>
                  </div>

                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1">Description</label>
                    <textarea
                      value={pipelineDescription}
                      onChange={(e) => setPipelineDescription(e.target.value)}
                      rows={2}
                      placeholder="Briefly describe the purpose of this pipeline..."
                      className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-100 text-xs focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    />
                  </div>

                  {/* Retrieval Strategy Selector */}
                  <div className="pt-2 border-t border-slate-800">
                    <label className="text-xs font-medium text-slate-300 block mb-2">Retrieval Strategy</label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => setRetrievalStrategy("hybrid")}
                        className={`p-3 rounded-xl border text-left transition-all ${
                          retrievalStrategy === "hybrid"
                            ? "bg-indigo-950/60 border-indigo-500 text-indigo-200"
                            : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                        }`}
                      >
                        <div className="text-xs font-semibold flex items-center gap-1.5">
                          <Layers className="h-3.5 w-3.5 text-indigo-400" />
                          Hybrid Search
                        </div>
                        <div className="text-[10px] text-slate-400 mt-1">
                          Dense Vector (halfvec) + Sparse BM25 + RRF
                        </div>
                      </button>

                      <button
                        type="button"
                        onClick={() => setRetrievalStrategy("dense")}
                        className={`p-3 rounded-xl border text-left transition-all ${
                          retrievalStrategy === "dense"
                            ? "bg-indigo-950/60 border-indigo-500 text-indigo-200"
                            : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                        }`}
                      >
                        <div className="text-xs font-semibold flex items-center gap-1.5">
                          <Database className="h-3.5 w-3.5 text-purple-400" />
                          Dense Only
                        </div>
                        <div className="text-[10px] text-slate-400 mt-1">
                          Pure semantic vector similarity search
                        </div>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Right Column: Search Depth, Reranker, Generation */}
                <div className="space-y-4">
                  <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Bot className="h-4 w-4 text-indigo-400" />
                    3. Tuning & Generation Parameters
                  </label>

                  {/* Sliders Grid */}
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-4">
                    {/* Top K After Rerank */}
                    <div>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="font-medium text-slate-300">Final Top-K Context Chunks</span>
                        <span className="font-mono text-indigo-400 font-semibold">{topKAfterRerank} chunks</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={15}
                        step={1}
                        value={topKAfterRerank}
                        onChange={(e) => setTopKAfterRerank(Number(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>

                    {/* LLM Temperature */}
                    <div>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="font-medium text-slate-300">LLM Temperature (Creativity)</span>
                        <span className="font-mono text-amber-400 font-semibold">{temperature.toFixed(1)}</span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={1.0}
                        step={0.1}
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e.target.value))}
                        className="w-full accent-amber-500"
                      />
                      <div className="flex items-center justify-between text-[10px] text-slate-500 mt-0.5">
                        <span>0.0 (Strict / Factual)</span>
                        <span>1.0 (Creative)</span>
                      </div>
                    </div>

                    {/* Context Token Budget */}
                    <div>
                      <label className="text-xs font-medium text-slate-300 block mb-1.5">
                        Max Context Token Window
                      </label>
                      <div className="grid grid-cols-3 gap-2">
                        {[2048, 4000, 8000].map((tokens) => (
                          <button
                            key={tokens}
                            type="button"
                            onClick={() => setContextTokenBudget(tokens)}
                            className={`py-1.5 px-2 rounded-lg text-xs font-mono font-medium border transition-all ${
                              contextTokenBudget === tokens
                                ? "bg-indigo-600 text-white border-indigo-400"
                                : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                            }`}
                          >
                            {tokens} tokens
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Toggles */}
                  <div className="space-y-2">
                    <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer hover:border-slate-700">
                      <div className="flex items-center gap-2">
                        <ArrowDownUp className="h-4 w-4 text-orange-400" />
                        <div>
                          <div className="text-xs font-medium text-slate-200">Cross-Encoder Reranker</div>
                          <div className="text-[10px] text-slate-400">Re-scores chunk candidates with BGE model</div>
                        </div>
                      </div>
                      <input
                        type="checkbox"
                        checked={enableReranker}
                        onChange={(e) => setEnableReranker(e.target.checked)}
                        className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-indigo-500 h-4 w-4"
                      />
                    </label>

                    <label className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800 cursor-pointer hover:border-slate-700">
                      <div className="flex items-center gap-2">
                        <Gauge className="h-4 w-4 text-emerald-400" />
                        <div>
                          <div className="text-xs font-medium text-slate-200">Automated LLM Judge Evaluation</div>
                          <div className="text-[10px] text-slate-400">Scores Faithfulness & Relevance in background</div>
                        </div>
                      </div>
                      <input
                        type="checkbox"
                        checked={autoEvaluate}
                        onChange={(e) => setAutoEvaluate(e.target.checked)}
                        className="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-indigo-500 h-4 w-4"
                      />
                    </label>
                  </div>
                </div>
              </div>

              {/* Step 3: Live Visual Pipeline Preview */}
              <div className="pt-4 border-t border-slate-800">
                <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5 mb-3">
                  <Workflow className="h-4 w-4 text-indigo-400" />
                  Live Pipeline Architecture Preview
                </label>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 overflow-x-auto">
                  <div className="flex items-center justify-start sm:justify-center gap-2 min-w-[650px] text-xs font-mono">
                    {/* Node 1: Query */}
                    <div className="px-3 py-2 rounded-lg bg-blue-950/80 border border-blue-800/60 text-blue-300 flex items-center gap-1.5 shadow-sm shrink-0">
                      <FileText className="h-3.5 w-3.5" />
                      <span>User Query</span>
                    </div>

                    <span className="text-slate-600">&rarr;</span>

                    {/* Node 2: Dense / Hybrid */}
                    <div className="px-3 py-2 rounded-lg bg-purple-950/80 border border-purple-800/60 text-purple-300 flex items-center gap-1.5 shadow-sm shrink-0">
                      <Database className="h-3.5 w-3.5" />
                      <span>Dense ({denseK})</span>
                    </div>

                    {retrievalStrategy === "hybrid" && (
                      <>
                        <span className="text-slate-600">+</span>
                        <div className="px-3 py-2 rounded-lg bg-amber-950/80 border border-amber-800/60 text-amber-300 flex items-center gap-1.5 shadow-sm shrink-0">
                          <FileCode className="h-3.5 w-3.5" />
                          <span>BM25 ({sparseK})</span>
                        </div>
                        <span className="text-slate-600">&rarr;</span>
                        <div className="px-3 py-2 rounded-lg bg-cyan-950/80 border border-cyan-800/60 text-cyan-300 flex items-center gap-1.5 shadow-sm shrink-0">
                          <Layers className="h-3.5 w-3.5" />
                          <span>RRF Fusion</span>
                        </div>
                      </>
                    )}

                    {enableReranker && (
                      <>
                        <span className="text-slate-600">&rarr;</span>
                        <div className="px-3 py-2 rounded-lg bg-orange-950/80 border border-orange-800/60 text-orange-300 flex items-center gap-1.5 shadow-sm shrink-0">
                          <ArrowDownUp className="h-3.5 w-3.5" />
                          <span>Reranker (Top {topKAfterRerank})</span>
                        </div>
                      </>
                    )}

                    <span className="text-slate-600">&rarr;</span>

                    {/* Node 4: Generation */}
                    <div className="px-3 py-2 rounded-lg bg-indigo-950/80 border border-indigo-800/60 text-indigo-300 flex items-center gap-1.5 shadow-sm shrink-0">
                      <Bot className="h-3.5 w-3.5" />
                      <span>LLM ({contextTokenBudget} tok)</span>
                    </div>

                    {autoEvaluate && (
                      <>
                        <span className="text-slate-600">&rarr;</span>
                        <div className="px-3 py-2 rounded-lg bg-emerald-950/80 border border-emerald-800/60 text-emerald-300 flex items-center gap-1.5 shadow-sm shrink-0">
                          <CheckCircle className="h-3.5 w-3.5" />
                          <span>Judge Eval</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </>
          ) : (
            /* Advanced JSON Tab */
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-slate-400">Canonical PipelineConfig.json</span>
                <span className="text-[11px] text-indigo-400 font-medium">Validates against Pydantic schema</span>
              </div>
              <div className="rounded-xl overflow-hidden border border-slate-800 bg-[#1e1e1e]">
                <Editor
                  height="420px"
                  language="json"
                  theme="vs-dark"
                  value={jsonText}
                  onChange={(val) => setJsonText(val || "")}
                  options={{
                    minimap: { enabled: false },
                    fontSize: 13,
                    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                    lineNumbers: "on",
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    tabSize: 2,
                    formatOnPaste: true,
                  }}
                />
              </div>
            </div>
          )}

          {/* Error Banner */}
          {error && (
            <div className="p-3.5 rounded-xl bg-rose-950/70 border border-rose-800/60 flex items-start gap-2.5 text-rose-300 text-xs font-mono">
              <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <div className="overflow-x-auto whitespace-pre-wrap">{error}</div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-950/80">
          <div className="text-xs text-slate-400">
            {activeTab === "visual" ? (
              <span className="flex items-center gap-1">
                <CheckCircle className="h-3.5 w-3.5 text-emerald-400" />
                Valid recipe configured
              </span>
            ) : (
              <span className="font-mono text-[11px] text-slate-500">Direct JSON submission mode</span>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="flex items-center gap-2 px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Registering Pipeline...
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4" />
                  Create Pipeline
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
