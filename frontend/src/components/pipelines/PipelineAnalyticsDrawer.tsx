"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import {
  X,
  Activity,
  DollarSign,
  Clock,
  AlertOctagon,
  CheckCircle2,
  BarChart3,
  Loader2,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Cpu,
  Layers,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";
import { fetchPipelineAnalytics } from "../../lib/api";
import { PipelineAnalytics } from "../../types";

interface PipelineAnalyticsDrawerProps {
  pipelineId: string | null;
  pipelineName?: string;
  onClose: () => void;
}

export function PipelineAnalyticsDrawer({
  pipelineId,
  pipelineName = "Pipeline",
  onClose,
}: PipelineAnalyticsDrawerProps) {
  const { data: analytics, isLoading, isError } = useQuery<PipelineAnalytics | null>({
    queryKey: ["pipeline-analytics", pipelineId],
    queryFn: () => (pipelineId ? fetchPipelineAnalytics(pipelineId) : null),
    enabled: !!pipelineId,
  });

  if (!pipelineId) return null;

  const totalRuns = analytics?.total_runs ?? 0;
  const hasHistory = totalRuns > 0;

  const latencyChartData = hasHistory
    ? [
        { name: "P50", latency: analytics?.latency.p50_ms ?? 0 },
        { name: "P95", latency: analytics?.latency.p95_ms ?? 0 },
        { name: "P99", latency: analytics?.latency.p99_ms ?? 0 },
        { name: "Mean", latency: analytics?.latency.avg_ms ?? 0 },
      ]
    : [];

  const evalScores = analytics?.evaluation_scores;
  const hasEvalData =
    evalScores &&
    (evalScores.faithfulness !== null ||
      evalScores.answer_relevance !== null ||
      evalScores.context_precision !== null ||
      evalScores.context_recall !== null ||
      evalScores.overall_score !== null);

  const formatPercent = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "N/A";
    return `${(val * 100).toFixed(1)}%`;
  };

  const getScoreColor = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "bg-slate-700 text-slate-300";
    if (val >= 0.85) return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    if (val >= 0.7) return "bg-amber-500/20 text-amber-300 border-amber-500/40";
    return "bg-rose-500/20 text-rose-300 border-rose-500/40";
  };

  const getProgressColor = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "bg-slate-700";
    if (val >= 0.85) return "bg-emerald-500";
    if (val >= 0.7) return "bg-amber-500";
    return "bg-rose-500";
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/70 backdrop-blur-sm flex justify-end animate-in fade-in duration-200">
      <div
        className="w-full max-w-2xl bg-slate-900 border-l border-slate-800 h-full flex flex-col shadow-2xl overflow-hidden animate-in slide-in-from-right duration-300"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-950/80 border border-indigo-800/40 text-indigo-400">
              <BarChart3 className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-slate-100 text-sm">
                  {analytics?.pipeline_name || pipelineName}
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                  {hasHistory ? `${totalRuns} ${totalRuns === 1 ? "run" : "runs"}` : "0 runs"}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5 truncate max-w-sm">
                ID: {pipelineId}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoading ? (
            <div className="h-full flex flex-col items-center justify-center gap-3 text-slate-400 py-24">
              <Loader2 className="h-7 w-7 animate-spin text-indigo-500" />
              <span className="text-xs font-medium">Computing pipeline analytics...</span>
            </div>
          ) : isError ? (
            <div className="p-6 rounded-2xl bg-rose-950/20 border border-rose-800/40 text-center space-y-3">
              <AlertOctagon className="h-8 w-8 text-rose-400 mx-auto" />
              <h4 className="text-sm font-semibold text-slate-200">Failed to Load Analytics</h4>
              <p className="text-xs text-slate-400">Could not retrieve analytics for this pipeline.</p>
            </div>
          ) : !hasHistory ? (
            /* Empty State for Zero-Run Pipelines */
            <div className="py-12 px-6 flex flex-col items-center justify-center text-center space-y-5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
              <div className="p-4 rounded-2xl bg-indigo-950/40 border border-indigo-800/30 text-indigo-400">
                <BarChart3 className="h-8 w-8" />
              </div>
              <div className="max-w-md space-y-2">
                <h4 className="text-base font-semibold text-slate-100">
                  No Query Runs Recorded Yet
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  This pipeline has not processed any queries yet. Run queries in the Query Playground
                  to start collecting real-time latency percentiles, LLM judge evaluation metrics, and token cost telemetry.
                </p>
              </div>

              <div className="pt-2">
                <Link
                  href="/playground"
                  onClick={onClose}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 transition-all group"
                >
                  <Sparkles className="h-4 w-4" />
                  Run in Query Playground
                  <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-0.5 transition-transform" />
                </Link>
              </div>
            </div>
          ) : (
            /* Populated Analytics Dashboard */
            <>
              {/* Section 1: Performance Latency */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-indigo-400" />
                    <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                      Performance & Latency (ms)
                    </h4>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">
                    Based on {totalRuns} {totalRuns === 1 ? "run" : "runs"}
                  </span>
                </div>

                {/* Metric Cards Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">P50 (Median)</span>
                    <div className="text-lg font-bold text-slate-100 mt-0.5">
                      {Math.round(analytics?.latency?.p50_ms ?? 0)} <span className="text-xs font-normal text-slate-400">ms</span>
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">P95 (95th %)</span>
                    <div className="text-lg font-bold text-indigo-300 mt-0.5">
                      {Math.round(analytics?.latency?.p95_ms ?? 0)} <span className="text-xs font-normal text-slate-400">ms</span>
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">P99 (Tail)</span>
                    <div className="text-lg font-bold text-purple-300 mt-0.5">
                      {Math.round(analytics?.latency?.p99_ms ?? 0)} <span className="text-xs font-normal text-slate-400">ms</span>
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Mean Average</span>
                    <div className="text-lg font-bold text-slate-200 mt-0.5">
                      {Math.round(analytics?.latency?.avg_ms ?? 0)} <span className="text-xs font-normal text-slate-400">ms</span>
                    </div>
                  </div>
                </div>

                {/* Latency Bar Chart */}
                <div className="h-40 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={latencyChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                      <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "#334155",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                        formatter={(val: any) => [`${val} ms`, "Latency"]}
                      />
                      <Bar dataKey="latency" fill="#6366f1" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Section 2: Quality Evaluation Scorecard */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-emerald-400" />
                    <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                      Quality & Evaluation Scorecard
                    </h4>
                  </div>
                  {evalScores?.overall_score !== null && (
                    <span
                      className={`text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full border ${getScoreColor(
                        evalScores?.overall_score
                      )}`}
                    >
                      Overall: {formatPercent(evalScores?.overall_score)}
                    </span>
                  )}
                </div>

                {hasEvalData ? (
                  <div className="space-y-3">
                    {[
                      {
                        label: "Faithfulness",
                        desc: "Factual consistency with retrieved context chunks",
                        val: evalScores?.faithfulness,
                      },
                      {
                        label: "Answer Relevance",
                        desc: "Direct relevance to the user query intent",
                        val: evalScores?.answer_relevance,
                      },
                      {
                        label: "Context Precision",
                        desc: "Signal-to-noise ratio of retrieved knowledge chunks",
                        val: evalScores?.context_precision,
                      },
                      {
                        label: "Context Recall",
                        desc: "Completeness of ground-truth knowledge retrieval",
                        val: evalScores?.context_recall,
                      },
                    ].map((item) => (
                      <div key={item.label} className="p-3 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-1.5">
                        <div className="flex items-center justify-between text-xs">
                          <div>
                            <span className="font-semibold text-slate-200">{item.label}</span>
                            <span className="text-[11px] text-slate-500 ml-2 hidden sm:inline">{item.desc}</span>
                          </div>
                          <span className="font-mono font-semibold text-slate-100">
                            {formatPercent(item.val)}
                          </span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${getProgressColor(item.val)}`}
                            style={{ width: `${item.val !== null && item.val !== undefined ? Math.min(100, Math.max(0, item.val * 100)) : 0}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center text-xs text-slate-400">
                    No automated judge evaluations recorded for these runs yet.
                  </div>
                )}
              </div>

              {/* Section 3: Usage & Cost Accounting */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <DollarSign className="h-4 w-4 text-amber-400" />
                    <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                      Usage & Token Cost Accounting
                    </h4>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">30-Day Window</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Total Queries</span>
                    <div className="text-lg font-bold text-slate-100 mt-0.5">{totalRuns}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Cost Per Query</span>
                    <div className="text-lg font-bold text-amber-300 mt-0.5">
                      ${(analytics?.cost?.cost_per_query_usd ?? 0).toFixed(5)}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800/80">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Total Token Cost</span>
                    <div className="text-lg font-bold text-emerald-300 mt-0.5">
                      ${(analytics?.cost?.total_cost_usd ?? 0).toFixed(4)}
                    </div>
                  </div>
                </div>

                {/* Daily Activity Chart */}
                {analytics?.queries_per_day && analytics.queries_per_day.length > 0 && (
                  <div className="pt-2">
                    <span className="text-[11px] font-medium text-slate-400 mb-2 block">Daily Query Volume</span>
                    <div className="h-36 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={analytics.queries_per_day} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <XAxis dataKey="date" stroke="#64748b" fontSize={10} tickLine={false} />
                          <YAxis stroke="#64748b" fontSize={10} tickLine={false} allowDecimals={false} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "8px",
                              fontSize: "12px",
                            }}
                            formatter={(val: any) => [`${val} queries`, "Volume"]}
                          />
                          <Line type="monotone" dataKey="count" stroke="#f59e0b" strokeWidth={2} dot={{ fill: "#f59e0b", r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>

              {/* Section 4: Reliability & Failures */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <AlertOctagon className="h-4 w-4 text-rose-400" />
                    <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                      Reliability & Failure Events
                    </h4>
                  </div>
                </div>

                {analytics?.recent_failures && analytics.recent_failures.length > 0 ? (
                  <div className="space-y-2">
                    {analytics.recent_failures.map((f) => (
                      <div key={f.run_id} className="p-3 rounded-xl bg-rose-950/20 border border-rose-900/40 space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-semibold text-slate-200 truncate max-w-sm">{f.query}</span>
                          <span className="text-[10px] font-mono text-slate-500">{f.timestamp}</span>
                        </div>
                        <p className="text-xs text-rose-400 font-mono">{f.error_message}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex items-center gap-3 p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-800/40 text-emerald-300 text-xs">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                    <span>100% Success Rate — 0 failure events recorded for this pipeline.</span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <span className="text-xs text-slate-500">NeuroFlow Telemetry v1.0</span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
          >
            Close Drawer
          </button>
        </div>
      </div>
    </div>
  );
}
