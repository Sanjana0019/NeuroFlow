"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Play,
  Columns,
  Layers,
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  Clock,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { fetchPipelines, startQueryStream, executeCompare, fetchRunEvaluation, submitHumanRating } from "../../lib/api";
import { Pipeline, Citation, Source, Evaluation } from "../../types";
import { useSSEStream } from "../../hooks/useSSEStream";
import { MetricGauge } from "../../components/common/MetricGauge";
import { ScoreBadge } from "../../components/common/Badge";
import { CitationDrawer } from "../../components/playground/CitationDrawer";
import { RetrievalInspector } from "../../components/playground/RetrievalInspector";
import { AnswerDiff } from "../../components/playground/AnswerDiff";
import { CompareScorecard } from "../../components/playground/CompareScorecard";

export default function PlaygroundPage() {
  const [queryText, setQueryText] = useState("");
  const [selectedPipelineId, setSelectedPipelineId] = useState<string>("");
  const [compareMode, setCompareMode] = useState(false);
  const [selectedPipelineBId, setSelectedPipelineBId] = useState<string>("");
  const [showInspector, setShowInspector] = useState(false);

  // Citation Drawer State
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [activeSource, setActiveSource] = useState<Source | null>(null);
  const [isCitationDrawerOpen, setIsCitationDrawerOpen] = useState(false);

  // Single Generation Run State
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Async Evaluation & Rating State
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [userRating, setUserRating] = useState<number | null>(null);
  const [isRatingSubmitting, setIsRatingSubmitting] = useState(false);

  // Compare Mode State
  const [isCompareSubmitting, setIsCompareSubmitting] = useState(false);
  const [compareResult, setCompareResult] = useState<any | null>(null);
  const [compareEvalA, setCompareEvalA] = useState<Evaluation | null>(null);
  const [compareEvalB, setCompareEvalB] = useState<Evaluation | null>(null);

  // Single SSE stream hook
  const streamA = useSSEStream(null);

  // Fetch Pipelines
  const { data: pipelines = [], isLoading: isPipelinesLoading } = useQuery<Pipeline[]>({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  // Default selection
  useEffect(() => {
    if (pipelines.length > 0 && !selectedPipelineId) {
      setSelectedPipelineId(pipelines[0].id);
      if (pipelines.length > 1) {
        setSelectedPipelineBId(pipelines[1].id);
      }
    }
  }, [pipelines, selectedPipelineId]);

  // Poll evaluation when generation completes in Single Mode
  useEffect(() => {
    let timer: NodeJS.Timeout;
    let attempts = 0;
    const targetId = streamA.runId || activeRunId;
    if (streamA.isComplete && targetId && !evaluation) {
      const pollEval = async () => {
        attempts++;
        try {
          const res = await fetchRunEvaluation(targetId);
          if (res && res.overall_score !== undefined && res.overall_score !== null) {
            setEvaluation(res);
            if (res.user_rating) setUserRating(res.user_rating);
            return;
          }
        } catch {
          // retry
        }
        if (attempts < 20) {
          timer = setTimeout(pollEval, 1000);
        }
      };
      timer = setTimeout(pollEval, 500);
    }
    return () => clearTimeout(timer);
  }, [streamA.isComplete, streamA.runId, activeRunId, evaluation]);

  // Submit Single Query
  const handleSingleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryText.trim()) return;

    setSubmitError(null);
    setIsSubmitting(true);
    setEvaluation(null);
    setUserRating(null);

    try {
      const { run_id } = await startQueryStream({
        query: queryText,
        pipeline_id: selectedPipelineId || undefined,
      });
      setActiveRunId(run_id);
      streamA.startStream(run_id);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || "Failed to start query execution";
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Submit Compare Query
  const handleCompareSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryText.trim() || !selectedPipelineId || !selectedPipelineBId) return;

    setSubmitError(null);
    setIsCompareSubmitting(true);
    setCompareResult(null);
    setCompareEvalA(null);
    setCompareEvalB(null);

    try {
      const result = await executeCompare({
        query: queryText,
        pipeline_a_id: selectedPipelineId,
        pipeline_b_id: selectedPipelineBId,
      });
      setCompareResult(result);

      const evalA = result.pipeline_a.evaluation;
      const evalB = result.pipeline_b.evaluation;
      if (evalA) setCompareEvalA(evalA);
      if (evalB) setCompareEvalB(evalB);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || "Compare execution failed";
      setSubmitError(msg);
    } finally {
      setIsCompareSubmitting(false);
    }
  };

  // Handle Human Feedback
  const handleRating = async (rating: number) => {
    if (!streamA.runId) return;
    setIsRatingSubmitting(true);
    try {
      const res = await submitHumanRating(streamA.runId, rating);
      setUserRating(res.user_rating || rating);
    } catch (err: any) {
      console.error("Failed to submit rating:", err);
    } finally {
      setIsRatingSubmitting(false);
    }
  };

  // Open Citation in Side Drawer
  const handleCitationClick = (reference: string) => {
    const citation = streamA.citations.find((c) => c.reference === reference) || {
      reference,
      chunk_id: null,
      document_name: "Retrieved Source",
      page_number: null,
      content_preview: "",
    };

    // Find matching source index
    const match = reference.match(/\d+/);
    const sourceIndex = match ? parseInt(match[0], 10) : 1;
    const source = streamA.sources.find((s) => s.source_index === sourceIndex) || null;

    setActiveCitation(citation);
    setActiveSource(source);
    setIsCitationDrawerOpen(true);
  };

  const selectedPipelineObj = pipelines.find((p) => p.id === selectedPipelineId);
  const selectedPipelineBObj = pipelines.find((p) => p.id === selectedPipelineBId);

  return (
    <div className="space-y-6">
      {/* Top Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Play className="h-5 w-5 text-indigo-500 fill-indigo-500/20" />
            Query Playground
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Test progressive SSE generation, interactive citations, live evaluation, and A/B comparison
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Compare Mode Toggle */}
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
              compareMode
                ? "bg-indigo-950/80 border-indigo-500/50 text-indigo-300 shadow-md shadow-indigo-950/50"
                : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            <Columns className="h-3.5 w-3.5" />
            Compare Mode {compareMode ? "(ON)" : "(OFF)"}
          </button>

          {/* Retrieval Inspector Toggle */}
          <button
            onClick={() => setShowInspector(!showInspector)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
              showInspector
                ? "bg-purple-950/80 border-purple-500/50 text-purple-300 shadow-md shadow-purple-950/50"
                : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            Inspect Retrieval {showInspector ? "(Open)" : ""}
          </button>
        </div>
      </div>

      {/* Query Form & Pipeline Selectors */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Pipeline A Selector */}
          <div>
            <label className="text-xs font-medium text-slate-300 mb-1.5 block flex items-center justify-between">
              <span>{compareMode ? "Pipeline A" : "Active Pipeline"}</span>
              {selectedPipelineObj?.average_score !== undefined && (
                <ScoreBadge score={selectedPipelineObj.average_score} label="Avg Score" />
              )}
            </label>
            <select
              value={selectedPipelineId}
              onChange={(e) => setSelectedPipelineId(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {pipelines.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} (v{p.version})
                </option>
              ))}
            </select>
          </div>

          {/* Pipeline B Selector (Compare Mode) */}
          {compareMode && (
            <div>
              <label className="text-xs font-medium text-purple-300 mb-1.5 block flex items-center justify-between">
                <span>Pipeline B</span>
                {selectedPipelineBObj?.average_score !== undefined && (
                  <ScoreBadge score={selectedPipelineBObj.average_score} label="Avg Score" />
                )}
              </label>
              <select
                value={selectedPipelineBId}
                onChange={(e) => setSelectedPipelineBId(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-purple-800/60 text-xs font-mono text-slate-200 focus:outline-none focus:border-purple-500"
              >
                {pipelines
                  .filter((p) => p.id !== selectedPipelineId)
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} (v{p.version})
                    </option>
                  ))}
              </select>
            </div>
          )}
        </div>

        {/* Query Input */}
        <form onSubmit={compareMode ? handleCompareSubmit : handleSingleSubmit} className="space-y-3">
          <div className="relative">
            <textarea
              rows={3}
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="Ask a question across indexed documents (e.g. What are the key enterprise risk factors in Q3?)"
              className="w-full p-4 rounded-xl bg-slate-950 border border-slate-700 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 font-sans leading-relaxed resize-none selection:bg-indigo-600/40"
            />
            <div className="absolute right-3 bottom-3 text-[10px] font-mono text-slate-500">
              {queryText.length} chars
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-500 font-mono">
                {compareMode ? "Concurrent A/B Execution" : "Real-time SSE Streaming"}
              </span>
            </div>

            <button
              type="submit"
              disabled={isSubmitting || isCompareSubmitting || !queryText.trim()}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-50"
            >
              {isSubmitting || isCompareSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {compareMode ? "Executing Dual A/B..." : "Streaming Generation..."}
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  {compareMode ? "Compare Pipelines" : "Run Query"}
                </>
              )}
            </button>
          </div>
        </form>

        {submitError && (
          <div className="p-3 rounded-lg bg-rose-950/60 border border-rose-800/50 flex items-center gap-2 text-rose-300 text-xs font-mono">
            <AlertCircle className="h-4 w-4 text-rose-400 shrink-0" />
            <span>{submitError}</span>
          </div>
        )}
      </div>

      {/* Retrieval Inspector View */}
      {showInspector && (
        <div className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Layers className="h-4 w-4 text-purple-400" />
              Retrieval Inspector (Live Architecture Breakdown)
            </h3>
            <button
              onClick={() => setShowInspector(false)}
              className="text-xs text-slate-500 hover:text-slate-300"
            >
              Close
            </button>
          </div>
          <RetrievalInspector
            query={queryText || "Active Query"}
            stageCounts={streamA.stageCounts || {}}
            pipelineName={selectedPipelineObj?.name || "Production-Hybrid-RAG"}
          />
        </div>
      )}

      {/* Single Mode Output Panel */}
      {!compareMode && (streamA.text || streamA.isLoading || streamA.isStreaming) && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Answer Column */}
          <div className="lg:col-span-2 space-y-4">
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
              {/* Header Info */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
                  <Cpu className="h-4 w-4 text-indigo-400" />
                  <span>{selectedPipelineObj?.name || "Pipeline Output"}</span>
                  {streamA.modelUsed && (
                    <span className="text-[10px] font-mono text-slate-500">({streamA.modelUsed})</span>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  {streamA.latencyMs && (
                    <span className="flex items-center gap-1 text-[11px] font-mono text-slate-400">
                      <Clock className="h-3.5 w-3.5 text-slate-500" />
                      {streamA.latencyMs}ms
                    </span>
                  )}
                  {streamA.isStreaming && (
                    <span className="flex items-center gap-1.5 text-xs text-indigo-400 font-mono">
                      <span className="h-2 w-2 rounded-full bg-indigo-400 animate-ping" />
                      Streaming...
                    </span>
                  )}
                </div>
              </div>

              {/* Streaming Answer Text */}
              <div className="font-sans text-sm text-slate-200 leading-relaxed whitespace-pre-wrap selection:bg-indigo-600/30 min-h-[120px]">
                {streamA.text || (
                  <div className="flex items-center gap-2 text-slate-500 text-xs py-8">
                    <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
                    Retrieving context and preparing generation...
                  </div>
                )}
                {streamA.isStreaming && (
                  <span className="inline-block w-1.5 h-4 ml-1 bg-indigo-400 animate-pulse" />
                )}
              </div>

              {/* Interactive Citation Chips */}
              {streamA.citations.length > 0 && (
                <div className="pt-4 border-t border-slate-800 space-y-2">
                  <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                    Citations (Click to Inspect Source Chunk):
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {streamA.citations.map((c, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleCitationClick(c.reference)}
                        className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium border transition-all flex items-center gap-1.5 ${
                          c.invalid_citation
                            ? "bg-rose-950/60 text-rose-300 border-rose-800 hover:bg-rose-900/60"
                            : "bg-indigo-950/60 text-indigo-300 border-indigo-800 hover:bg-indigo-900/60 hover:border-indigo-600"
                        }`}
                      >
                        <span>{c.reference}</span>
                        <span className="text-[10px] text-slate-400 truncate max-w-[120px]">
                          ({c.document_name})
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Human Feedback Thumbs Up / Down */}
              {streamA.isComplete && (
                <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                  <span className="text-xs text-slate-400">Rate answer quality:</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleRating(5)}
                      disabled={isRatingSubmitting}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        userRating === 5
                          ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                          : "bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700"
                      }`}
                    >
                      <ThumbsUp className="h-3.5 w-3.5" />
                      Helpful
                    </button>
                    <button
                      onClick={() => handleRating(1)}
                      disabled={isRatingSubmitting}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        userRating === 1
                          ? "bg-rose-950 text-rose-300 border border-rose-700"
                          : "bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700"
                      }`}
                    >
                      <ThumbsDown className="h-3.5 w-3.5" />
                      Poor
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Evaluation Gauges Column */}
          <div className="space-y-4">
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wide flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-emerald-400" />
                  Live Evaluation
                </h3>
                {evaluation?.overall_score !== undefined && (
                  <ScoreBadge score={evaluation.overall_score} label="Overall" />
                )}
              </div>

              {!evaluation ? (
                <div className="py-8 flex flex-col items-center justify-center text-center gap-2 text-slate-500">
                  <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
                  <span className="text-xs font-mono">Running asynchronous LLM Judge evaluation...</span>
                </div>
              ) : (
                <div className="space-y-2.5">
                  <MetricGauge label="Faithfulness" score={evaluation.faithfulness} weight="35%" />
                  <MetricGauge label="Answer Relevance" score={evaluation.answer_relevance} weight="30%" />
                  <MetricGauge label="Context Precision" score={evaluation.context_precision} weight="20%" />
                  <MetricGauge label="Context Recall" score={evaluation.context_recall} weight="15%" />

                  <div className="pt-2 text-[10px] font-mono text-slate-500 text-center">
                    Judge Model: {evaluation.judge_model || "gpt-4o-mini"}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Compare Mode Output Panels */}
      {compareMode && compareResult && (
        <div className="space-y-6 animate-in fade-in duration-300">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pipeline A Panel */}
            <div className="p-6 rounded-2xl bg-slate-900 border border-indigo-900/40 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <span className="font-semibold text-indigo-400 text-sm">
                  {compareResult.pipeline_a.name} (v{compareResult.pipeline_a.version})
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  Latency: {compareResult.pipeline_a.latency_ms}ms | Chunks: {compareResult.pipeline_a.chunks_used}
                </span>
              </div>
              <div className="font-sans text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">
                {compareResult.pipeline_a.generation}
              </div>
            </div>

            {/* Pipeline B Panel */}
            <div className="p-6 rounded-2xl bg-slate-900 border border-purple-900/40 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <span className="font-semibold text-purple-400 text-sm">
                  {compareResult.pipeline_b.name} (v{compareResult.pipeline_b.version})
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  Latency: {compareResult.pipeline_b.latency_ms}ms | Chunks: {compareResult.pipeline_b.chunks_used}
                </span>
              </div>
              <div className="font-sans text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">
                {compareResult.pipeline_b.generation}
              </div>
            </div>
          </div>

          {/* Side-by-Side Word Diff */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider px-1">
              Side-by-Side Answer Diff
            </h3>
            <AnswerDiff
              textA={compareResult.pipeline_a.generation}
              textB={compareResult.pipeline_b.generation}
              nameA={compareResult.pipeline_a.name}
              nameB={compareResult.pipeline_b.name}
            />
          </div>

          {/* Comparative Evaluation Scorecard */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider px-1">
              Evaluation Metrics Comparison
            </h3>
            <CompareScorecard
              nameA={compareResult.pipeline_a.name}
              nameB={compareResult.pipeline_b.name}
              evalA={compareEvalA}
              evalB={compareEvalB}
            />
          </div>
        </div>
      )}

      {/* Citation Drawer */}
      <CitationDrawer
        isOpen={isCitationDrawerOpen}
        onClose={() => setIsCitationDrawerOpen(false)}
        citation={activeCitation}
        source={activeSource}
      />
    </div>
  );
}
