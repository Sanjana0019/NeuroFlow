"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, Filter, ChevronDown, ChevronUp, Clock, Hash, Sparkles, Loader2, ThumbsUp, ThumbsDown, AlertTriangle } from "lucide-react";
import { fetchEvaluations, fetchPipelines, submitHumanRating } from "../../lib/api";
import { useEvaluationStream } from "../../hooks/useEvaluationStream";
import { MetricGauge } from "../../components/common/MetricGauge";
import { ScoreBadge } from "../../components/common/Badge";
import { Pipeline } from "../../types";

export default function EvaluationsPage() {
  const [selectedPipelineFilter, setSelectedPipelineFilter] = useState<string>("");
  const [metricThresholdFilter, setMetricThresholdFilter] = useState<string>("all");
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [userRatings, setUserRatings] = useState<Record<string, number>>({});
  const [ratingSubmitting, setRatingSubmitting] = useState<Record<string, boolean>>({});

  const handleCardRating = async (runId: string, rating: number) => {
    if (!runId) return;
    setRatingSubmitting((prev) => ({ ...prev, [runId]: true }));
    try {
      const res = await submitHumanRating(runId, rating);
      setUserRatings((prev) => ({ ...prev, [runId]: res.user_rating || rating }));
    } catch (err: any) {
      console.error("Failed to submit rating:", err);
    } finally {
      setRatingSubmitting((prev) => ({ ...prev, [runId]: false }));
    }
  };

  // Fetch initial historical evaluations
  const { data: initialEvals = [], isLoading } = useQuery({
    queryKey: ["evaluations", selectedPipelineFilter],
    queryFn: () =>
      fetchEvaluations({
        pipeline_id: selectedPipelineFilter || undefined,
        limit: 50,
      }),
  });

  // Fetch pipelines for filter dropdown
  const { data: pipelines = [] } = useQuery<Pipeline[]>({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  // Real-time live SSE stream
  const { evaluations, isConnected } = useEvaluationStream(initialEvals);

  // Apply Client Filters
  const filteredEvaluations = evaluations.filter((item) => {
    if (selectedPipelineFilter && item.pipeline_name) {
      const p = pipelines.find((pipe) => pipe.id === selectedPipelineFilter);
      if (p && !item.pipeline_name.toLowerCase().includes(p.name.toLowerCase())) {
        return false;
      }
    }

    if (metricThresholdFilter === "low_overall" && (item.overall_score ?? 1) >= 0.7) {
      return false;
    }
    if (metricThresholdFilter === "low_faithfulness" && (item.faithfulness ?? 1) >= 0.7) {
      return false;
    }
    if (metricThresholdFilter === "low_relevance" && (item.answer_relevance ?? 1) >= 0.7) {
      return false;
    }

    return true;
  });

  const toggleExpand = (runId: string) => {
    setExpandedRunId(expandedRunId === runId ? null : runId);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Activity className="h-5 w-5 text-emerald-400" />
            Live Evaluation Feed
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time SSE event stream of asynchronous LLM Judge evaluation metrics
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
            <span
              className={`h-2 w-2 rounded-full ${
                isConnected ? "bg-emerald-400 animate-pulse" : "bg-rose-400"
              }`}
            />
            {isConnected ? "SSE Stream Connected" : "Connecting..."}
          </div>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-xl flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-slate-400 font-medium">
            <Filter className="h-3.5 w-3.5" />
            Filters:
          </div>

          {/* Pipeline Dropdown */}
          <select
            value={selectedPipelineFilter}
            onChange={(e) => setSelectedPipelineFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
          >
            <option value="">All Pipelines</option>
            {pipelines.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          {/* Metric Threshold Dropdown */}
          <select
            value={metricThresholdFilter}
            onChange={(e) => setMetricThresholdFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
          >
            <option value="all">All Thresholds</option>
            <option value="low_overall">Overall Score &lt; 0.7</option>
            <option value="low_faithfulness">Faithfulness &lt; 0.7</option>
            <option value="low_relevance">Answer Relevance &lt; 0.7</option>
          </select>
        </div>

        <div className="text-[11px] font-mono text-slate-500">
          Showing {filteredEvaluations.length} evaluation events
        </div>
      </div>

      {/* Evaluation Stream Cards */}
      {isLoading ? (
        <div className="py-20 flex flex-col items-center justify-center gap-3 text-slate-400">
          <Loader2 className="h-6 w-6 animate-spin text-emerald-500" />
          <span className="text-xs font-mono">Loading live evaluation feed...</span>
        </div>
      ) : filteredEvaluations.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-slate-900 border border-slate-800 text-xs text-slate-400 font-mono">
          No evaluations matching current filters. Run queries in the playground to trigger evaluation.
        </div>
      ) : (
        <div className="space-y-4">
          {filteredEvaluations.map((ev, idx) => {
            const isExpanded = expandedRunId === ev.run_id;

            return (
              <div
                key={ev.run_id || idx}
                className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl hover:border-slate-700 transition-all"
              >
                {/* Top Summary Row */}
                <div
                  className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 cursor-pointer"
                  onClick={() => toggleExpand(ev.run_id)}
                >
                  <div className="space-y-1 max-w-2xl">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-indigo-950 border border-indigo-800 text-indigo-300 text-[10px] font-mono font-medium">
                        {ev.pipeline_name || "Production Pipeline"}
                      </span>
                      {ev.evaluated_at && (
                        <span className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
                          <Clock className="h-3 w-3" />
                          {new Date(ev.evaluated_at).toLocaleTimeString()}
                        </span>
                      )}
                    </div>
                    <h3 className="text-xs font-medium text-slate-200 line-clamp-1">
                      {ev.query || "Query prompt"}
                    </h3>
                  </div>

                  <div className="flex items-center gap-3">
                    <ScoreBadge score={ev.overall_score} label="Overall" />
                    <button className="p-1 rounded-lg text-slate-400 hover:text-slate-200">
                      {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                {/* Metric Gauges Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                  <MetricGauge label="Faithfulness" score={ev.faithfulness} weight="35%" />
                  <MetricGauge label="Answer Relevance" score={ev.answer_relevance} weight="30%" />
                  <MetricGauge label="Context Precision" score={ev.context_precision} weight="20%" />
                  <MetricGauge label="Context Recall" score={ev.context_recall} weight="15%" />
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t border-slate-800 space-y-4 animate-in fade-in duration-200">
                    <div>
                      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                        Full User Query:
                      </h4>
                      <p className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-sans text-slate-200">
                        {ev.query || "No query recorded"}
                      </p>
                    </div>

                    <div>
                      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                        Generated Answer:
                      </h4>
                      <p className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-sans text-slate-200 whitespace-pre-wrap leading-relaxed">
                        {ev.generation || "No generation recorded"}
                      </p>
                    </div>

                    {ev.chunks && ev.chunks.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                          <Hash className="h-3.5 w-3.5 text-indigo-400" />
                          Retrieved Context Chunks ({ev.chunks.length}):
                        </h4>
                        <div className="space-y-2">
                          {ev.chunks.map((c, i) => (
                            <div
                              key={c.id || i}
                              className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300"
                            >
                              <span className="text-[10px] text-indigo-400 block mb-1">Chunk #{c.chunk_index}</span>
                              {c.content}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Human Calibration & Training Pair Status */}
                    <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/80">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] text-slate-400 font-medium">Rate Answer Quality:</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCardRating(ev.run_id, 5);
                          }}
                          disabled={ratingSubmitting[ev.run_id]}
                          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                            (userRatings[ev.run_id] ?? ev.user_rating) === 5
                              ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                              : "bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700"
                          }`}
                        >
                          <ThumbsUp className="h-3 w-3" />
                          Helpful
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCardRating(ev.run_id, 1);
                          }}
                          disabled={ratingSubmitting[ev.run_id]}
                          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                            (userRatings[ev.run_id] ?? ev.user_rating) === 1
                              ? "bg-rose-950 text-rose-300 border border-rose-700"
                              : "bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700"
                          }`}
                        >
                          <ThumbsDown className="h-3 w-3" />
                          Poor
                        </button>
                      </div>

                      <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
                        <span>Judge: {ev.judge_model || "nvidia/nemotron-3-nano-30b-a3b:free"}</span>
                        <span>•</span>
                        <span className="text-emerald-400 font-medium">Training Pair: Extracted</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
