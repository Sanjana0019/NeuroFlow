"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Workflow,
  Plus,
  BarChart3,
  Clock,
  CheckCircle,
  Cpu,
  Loader2,
  Trash2,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  HelpCircle,
} from "lucide-react";
import clsx from "clsx";
import { fetchPipelines, deletePipeline } from "../../lib/api";
import { Pipeline } from "../../types";
import { ScoreBadge } from "../../components/common/Badge";
import { CreatePipelineModal } from "../../components/pipelines/CreatePipelineModal";
import { PipelineAnalyticsDrawer } from "../../components/pipelines/PipelineAnalyticsDrawer";

function PipelineTrendSparkline({
  trend,
  id,
}: {
  trend: { date: string; query_count: number }[];
  id: string;
}) {
  const counts = trend.map((t) => t.query_count);
  const maxVal = Math.max(...counts, 1);
  const width = 280;
  const height = 44;
  const paddingX = 8;
  const paddingY = 6;
  const step = (width - paddingX * 2) / (counts.length - 1 || 1);

  const points = counts.map((c, i) => {
    const x = paddingX + i * step;
    const y = height - paddingY - (c / maxVal) * (height - paddingY * 2);
    return { x, y, count: c, date: trend[i]?.date };
  });

  const pathD = points.reduce((acc, p, i) => {
    if (i === 0) return `M ${p.x},${p.y}`;
    const prev = points[i - 1];
    const cx1 = prev.x + (p.x - prev.x) / 2;
    const cy1 = prev.y;
    const cx2 = prev.x + (p.x - prev.x) / 2;
    const cy2 = p.y;
    return `${acc} C ${cx1},${cy1} ${cx2},${cy2} ${p.x},${p.y}`;
  }, "");

  const areaD = `${pathD} L ${points[points.length - 1].x},${height} L ${points[0].x},${height} Z`;
  const gradId = `spark-grad-${id.replace(/[^a-zA-Z0-9]/g, "")}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#818cf8" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <path d={areaD} fill={`url(#${gradId})`} />
      <path
        d={pathD}
        fill="none"
        stroke="#818cf8"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {points.map((p, idx) => (
        <circle
          key={idx}
          cx={p.x}
          cy={p.y}
          r={p.count > 0 ? "3" : "1.5"}
          className={
            p.count > 0
              ? "fill-indigo-300 stroke-indigo-950 stroke-2"
              : "fill-slate-700"
          }
        >
          <title>{`${p.date}: ${p.count} queries`}</title>
        </circle>
      ))}
    </svg>
  );
}

export default function PipelinesPage() {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedPipelineForAnalytics, setSelectedPipelineForAnalytics] = useState<Pipeline | null>(null);
  const [pipelineToDelete, setPipelineToDelete] = useState<Pipeline | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const {
    data: pipelines = [],
    isLoading,
    refetch,
  } = useQuery<Pipeline[]>({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  const isDefaultPipeline = (name: string) => {
    const normalized = name.toLowerCase().replace(/[-_]/g, "");
    return (
      normalized.includes("productionhybridrag") ||
      normalized.includes("fastdensesearch") ||
      normalized.includes("defaultragpipeline")
    );
  };

  const handleDeleteConfirm = async () => {
    if (!pipelineToDelete) return;
    setDeletingId(pipelineToDelete.id);
    try {
      await deletePipeline(pipelineToDelete.id);
      setPipelineToDelete(null);
      await refetch();
    } catch (err: any) {
      console.error("Failed to delete pipeline:", err);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Workflow className="h-5 w-5 text-indigo-500" />
            Pipeline Manager
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Config-driven RAG pipelines, schema validation, versioning, and analytics
          </p>
        </div>

        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-lg shadow-indigo-600/30"
        >
          <Plus className="h-4 w-4" />
          Create Pipeline
        </button>
      </div>

      {/* Pipeline Cards Grid */}
      {isLoading ? (
        <div className="py-20 flex flex-col items-center justify-center gap-3 text-slate-400">
          <Loader2 className="h-6 w-6 animate-spin text-indigo-500" />
          <span className="text-xs font-mono">Loading pipelines...</span>
        </div>
      ) : pipelines.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="p-3 w-12 h-12 mx-auto rounded-xl bg-indigo-950/80 border border-indigo-800/40 text-indigo-400 flex items-center justify-center">
            <Workflow className="h-6 w-6" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">No pipelines registered</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Get started by creating your first versioned pipeline configuration.
          </p>
          <button
            onClick={() => setIsCreateModalOpen(true)}
            className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-xs font-medium"
          >
            Create Pipeline
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {pipelines.map((pipeline) => {
            const summary = pipeline.metrics_summary;
            const qualityScore = summary?.quality_score;
            const hasData = summary?.has_data ?? false;
            const queries7d = summary?.queries_7d ?? 0;
            const queriesChange = summary?.queries_change_pct;
            const p50Latency = summary?.latency_p50_ms;
            const latChange = summary?.latency_change_pct;
            const faithfulness = summary?.faithfulness;
            const faithChange = summary?.faithfulness_change_pct;
            const trend7d = summary?.trend_7d || [];
            const hasActiveQueries = trend7d.some((d) => d.query_count > 0);

            return (
              <div
                key={pipeline.id}
                onClick={() => setSelectedPipelineForAnalytics(pipeline)}
                className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl hover:border-indigo-500/50 hover:shadow-indigo-950/20 transition-all cursor-pointer flex flex-col justify-between group"
              >
                <div>
                  {/* Card Header */}
                  <div className="flex items-start justify-between gap-2 mb-3">
                    <div className="flex items-center gap-2.5">
                      <div className="p-2 rounded-xl bg-indigo-950/80 border border-indigo-800/40 text-indigo-400 group-hover:text-indigo-300">
                        <Cpu className="h-4 w-4" />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-slate-100 group-hover:text-indigo-300 transition-colors">
                          {pipeline.name}
                        </h3>
                        <span className="text-[10px] font-mono text-slate-500">v{pipeline.version}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Quality Score Badge or N/A */}
                      {qualityScore !== null && qualityScore !== undefined ? (
                        <span
                          className={clsx(
                            "px-2 py-0.5 rounded-full text-xs font-mono font-semibold border",
                            qualityScore >= 0.8
                              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                              : qualityScore >= 0.6
                              ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                              : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                          )}
                          title={`Overall Quality: ${(qualityScore * 100).toFixed(1)}%`}
                        >
                          {`${Math.round(qualityScore * 100)}%`}
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-xs font-mono font-semibold bg-slate-800 text-slate-400 border border-slate-700">
                          N/A
                        </span>
                      )}

                      {!isDefaultPipeline(pipeline.name) && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setPipelineToDelete(pipeline);
                          }}
                          disabled={deletingId === pipeline.id}
                          title={`Delete ${pipeline.name}`}
                          className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-950/40 border border-transparent hover:border-rose-800/40 transition-colors"
                        >
                          {deletingId === pipeline.id ? (
                            <Loader2 className="h-4 w-4 animate-spin text-rose-400" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>

                  <p className="text-xs text-slate-400 line-clamp-2 mb-4 font-sans leading-relaxed">
                    {pipeline.description || pipeline.config?.description || "Configurable RAG pipeline."}
                  </p>
                </div>

                <div className="space-y-3.5">
                  {/* 3 Metric Columns Row */}
                  <div className="grid grid-cols-3 gap-2 py-3 border-t border-slate-800/80 text-left">
                    {/* Column 1: 7d Queries */}
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-mono text-slate-400 block">7d Queries</span>
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-base font-bold font-mono text-slate-100">
                          {queries7d}
                        </span>
                        {queriesChange !== null && queriesChange !== undefined ? (
                          <span
                            className={clsx(
                              "text-[10px] font-mono font-semibold",
                              queriesChange >= 0 ? "text-emerald-400" : "text-rose-400"
                            )}
                          >
                            {queriesChange >= 0 ? "↑" : "↓"} {Math.abs(queriesChange)}%
                          </span>
                        ) : summary && summary.queries_previous_7d === 0 && summary.queries_7d > 0 ? (
                          <span className="text-[10px] font-mono text-slate-400">New</span>
                        ) : (
                          <span className="text-[10px] font-mono text-slate-600">—</span>
                        )}
                      </div>
                      <span className="text-[9px] text-slate-500 block">vs last 7d</span>
                    </div>

                    {/* Column 2: Avg Latency */}
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-mono text-slate-400 block">Avg Latency</span>
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-base font-bold font-mono text-slate-100">
                          {p50Latency !== null && p50Latency !== undefined
                            ? p50Latency >= 1000
                              ? `${(p50Latency / 1000).toFixed(2)}s`
                              : `${p50Latency.toFixed(0)}ms`
                            : "—"}
                        </span>
                        {latChange !== null && latChange !== undefined && (
                          <span
                            className={clsx(
                              "text-[10px] font-mono font-semibold",
                              latChange <= 0 ? "text-emerald-400" : "text-amber-400"
                            )}
                          >
                            {latChange <= 0 ? "↓" : "↑"} {Math.abs(latChange)}%
                          </span>
                        )}
                      </div>
                      <span className="text-[9px] text-slate-500 block">p50</span>
                    </div>

                    {/* Column 3: Avg Score (Faithfulness) */}
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-mono text-slate-400 block">Avg Score</span>
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-base font-bold font-mono text-slate-100">
                          {faithfulness !== null && faithfulness !== undefined
                            ? faithfulness.toFixed(2)
                            : "—"}
                        </span>
                        {faithChange !== null && faithChange !== undefined && (
                          <span
                            className={clsx(
                              "text-[10px] font-mono font-semibold",
                              faithChange >= 0 ? "text-emerald-400" : "text-rose-400"
                            )}
                          >
                            {faithChange >= 0 ? "↑" : "↓"} {Math.abs(faithChange)}%
                          </span>
                        )}
                      </div>
                      <span className="text-[9px] text-slate-500 block">Faithfulness</span>
                    </div>
                  </div>

                  {/* 7d Trend Area */}
                  <div className="space-y-1.5 pt-1 border-t border-slate-800/40">
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                      <span>7d Trend</span>
                      {hasData && (
                        <span className="text-[9px] text-slate-500">
                          {trend7d.reduce((sum, d) => sum + d.query_count, 0)} runs total
                        </span>
                      )}
                    </div>

                    {hasData && hasActiveQueries ? (
                      <div className="h-14 w-full pt-1">
                        <PipelineTrendSparkline trend={trend7d} id={pipeline.id} />
                      </div>
                    ) : (
                      <div className="h-14 flex flex-col items-center justify-center text-center p-2 rounded-xl bg-slate-950/40 border border-slate-800/60 space-y-0.5">
                        <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1.5">
                          <BarChart3 className="h-3.5 w-3.5 text-slate-500" />
                          No data yet
                        </span>
                        <span className="text-[10px] text-slate-500">
                          Run queries in the Playground to collect analytics.
                        </span>
                      </div>
                    )}
                  </div>

                  {/* View Analytics CTA */}
                  <div className="pt-2 flex items-center justify-between text-[11px] font-medium text-indigo-400 group-hover:text-indigo-300 border-t border-slate-800/40">
                    <span className="flex items-center gap-1.5">
                      <BarChart3 className="h-3.5 w-3.5" />
                      View Analytics Drawer
                    </span>
                    <span className="group-hover:translate-x-0.5 transition-transform">&rarr;</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {pipelineToDelete && (
        <div className="fixed inset-0 z-50 overflow-hidden bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div
            className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6 space-y-4 animate-in fade-in zoom-in-95 duration-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 text-rose-400">
              <div className="p-2 rounded-xl bg-rose-950/80 border border-rose-800/40">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-white">Delete Pipeline</h3>
                <p className="text-xs text-slate-400">This action cannot be undone.</p>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-xs text-slate-300 space-y-1">
              <span className="font-semibold text-slate-100">{pipelineToDelete.name}</span>
              <p className="text-slate-400 text-[11px]">
                Are you sure you want to permanently delete this pipeline configuration?
              </p>
            </div>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setPipelineToDelete(null)}
                disabled={deletingId === pipelineToDelete.id}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteConfirm}
                disabled={deletingId === pipelineToDelete.id}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white flex items-center gap-2 transition-all shadow-lg shadow-rose-600/30 disabled:opacity-50"
              >
                {deletingId === pipelineToDelete.id ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete Permanently
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Visual Create Pipeline Modal */}
      <CreatePipelineModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onSuccess={() => refetch()}
      />

      {/* Analytics Drawer */}
      <PipelineAnalyticsDrawer
        pipelineId={selectedPipelineForAnalytics?.id ?? null}
        pipelineName={selectedPipelineForAnalytics?.name}
        onClose={() => setSelectedPipelineForAnalytics(null)}
      />
    </div>
  );
}
