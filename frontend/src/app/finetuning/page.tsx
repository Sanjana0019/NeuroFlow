"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Sparkles,
  Download,
  CheckCircle2,
  AlertCircle,
  Clock,
  ArrowRight,
  Database,
  Cpu,
  Layers,
  FileText,
  ShieldCheck,
  Zap,
  Info,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Loader2,
  XCircle,
  HelpCircle,
} from "lucide-react";
import clsx from "clsx";
import {
  fetchDatasetReadiness,
  fetchTrainingPairs,
  fetchDPOPreview,
  fetchFinetuneJobs,
  createFinetuneJob,
  getDatasetExportUrl,
} from "../../lib/api";
import { TrainingPairDetail } from "../../types";

export default function FineTuningPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<"sft" | "dpo" | "jobs">("sft");
  const [selectedBaseModel, setSelectedBaseModel] = useState("gpt-4o-mini-2024-07-18");
  const [expandedPairId, setExpandedPairId] = useState<string | null>(null);
  const [isSubmittingJob, setIsSubmittingJob] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [jobSuccess, setJobSuccess] = useState<string | null>(null);

  // Queries
  const { data: readiness, isLoading: isLoadingReadiness } = useQuery({
    queryKey: ["dataset-readiness"],
    queryFn: fetchDatasetReadiness,
  });

  const { data: trainingPairs = [], isLoading: isLoadingPairs } = useQuery({
    queryKey: ["training-pairs"],
    queryFn: () => fetchTrainingPairs(100),
  });

  const { data: dpoPairs = [], isLoading: isLoadingDpo } = useQuery({
    queryKey: ["dpo-pairs"],
    queryFn: fetchDPOPreview,
  });

  const { data: jobs = [], isLoading: isLoadingJobs } = useQuery({
    queryKey: ["finetune-jobs"],
    queryFn: fetchFinetuneJobs,
  });

  // Mutation
  const createJobMutation = useMutation({
    mutationFn: createFinetuneJob,
    onSuccess: (data) => {
      setIsSubmittingJob(false);
      setJobSuccess(`Fine-tuning job submitted successfully! Job ID: ${data.job_id}`);
      setJobError(null);
      queryClient.invalidateQueries({ queryKey: ["finetune-jobs"] });
      queryClient.invalidateQueries({ queryKey: ["dataset-readiness"] });
      setActiveTab("jobs");
    },
    onError: (err: any) => {
      setIsSubmittingJob(false);
      const msg = err.response?.data?.detail || err.message || "Failed to submit fine-tuning job";
      setJobError(msg);
      setJobSuccess(null);
    },
  });

  const handleLaunchJob = () => {
    setJobError(null);
    setJobSuccess(null);
    setIsSubmittingJob(true);
    createJobMutation.mutate({
      base_model: selectedBaseModel,
      min_quality_score: 0.82,
    });
  };

  const eligibleCount = readiness?.eligible_sft_count ?? trainingPairs.filter((p) => p.is_valid).length;
  const minRequired = readiness?.min_required_for_finetuning ?? 10;
  const progressPercent = Math.min(100, Math.round((eligibleCount / minRequired) * 100));
  const remainingCount = Math.max(0, minRequired - eligibleCount);
  const canExport = eligibleCount > 0;
  const isOpenAIConfigured = readiness?.openai_configured ?? false;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 pb-16">
      {/* Top Banner */}
      <div className="border-b border-slate-800 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-800/40 text-indigo-400 text-xs font-semibold">
                <Sparkles className="h-3.5 w-3.5" />
                Data Flywheel & Model Customization
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
                Fine-Tuning & Datasets
              </h1>
              <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">
                Turn your best NeuroFlow answers into standardized training datasets. Review harvested
                examples, export clean JSONL files, and optionally fine-tune supported language models.
              </p>
            </div>

            {/* Quick Export Actions */}
            <div className="flex flex-wrap items-center gap-2">
              <a
                href={canExport ? getDatasetExportUrl("sft", "jsonl") : undefined}
                download
                className={clsx(
                  "inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold shadow-sm transition-all",
                  canExport
                    ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/20 cursor-pointer"
                    : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-800"
                )}
                title={canExport ? "Download SFT dataset in OpenAI JSONL format" : "No valid examples available to export"}
              >
                <Download className="h-4 w-4" />
                Export SFT JSONL
              </a>
              <a
                href={canExport ? getDatasetExportUrl("sft", "json") : undefined}
                download
                className={clsx(
                  "inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium border transition-colors",
                  canExport
                    ? "bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800 cursor-pointer"
                    : "bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed"
                )}
              >
                <FileText className="h-3.5 w-3.5" />
                JSON
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 space-y-8">
        {/* Step-by-Step Educational Flow */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 space-y-4">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-indigo-400" />
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              How Data Harvesting & Fine-Tuning Works in NeuroFlow
            </h3>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {[
              {
                step: "1",
                title: "Ask in Playground",
                desc: "You run queries on uploaded enterprise documents.",
              },
              {
                step: "2",
                title: "AI Judge Evaluates",
                desc: "Every answer is scored for Faithfulness & Relevance.",
              },
              {
                step: "3",
                title: "Auto-Harvest Pairs",
                desc: "High-scoring answers (>= 80%) become training candidates.",
              },
              {
                step: "4",
                title: "Validate & Curate",
                desc: "Checked for citations, 50-2000 tokens, and no PII.",
              },
              {
                step: "5",
                title: "Export or Fine-Tune",
                desc: "Download JSONL or optionally train a custom model.",
              },
            ].map((item, i) => (
              <div
                key={item.step}
                className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/60 space-y-1 relative"
              >
                <div className="flex items-center justify-between text-xs font-mono text-indigo-400">
                  <span>Step {item.step}</span>
                  {i < 4 && <ArrowRight className="h-3 w-3 text-slate-600 hidden lg:block" />}
                </div>
                <h4 className="text-xs font-semibold text-slate-200">{item.title}</h4>
                <p className="text-[11px] text-slate-400 leading-snug">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Dataset Readiness Overview Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Readiness Gauge Card */}
          <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-indigo-950/80 border border-indigo-800/40 text-indigo-400">
                  <Database className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Dataset Readiness for Fine-Tuning</h3>
                  <p className="text-xs text-slate-400">
                    Real-time counts from PostgreSQL <code className="font-mono text-indigo-300">training_pairs</code>
                  </p>
                </div>
              </div>
              <span
                className={clsx(
                  "text-xs font-mono px-2.5 py-1 rounded-full border",
                  eligibleCount >= minRequired
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                    : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                )}
              >
                {eligibleCount >= minRequired ? "Ready for Training" : "Building Dataset"}
              </span>
            </div>

            {/* Progress bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">
                  {eligibleCount} / {minRequired} training examples ready
                </span>
                <span className="font-mono text-slate-400">{progressPercent}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-indigo-500 h-full rounded-full transition-all duration-500"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <p className="text-xs text-slate-400">
                {eligibleCount >= minRequired
                  ? "✓ You have met the minimum threshold of 10 valid examples required to submit a fine-tuning job."
                  : `Need ${remainingCount} more valid training examples before fine-tuning can be launched.`}
              </p>
            </div>

            {/* Validation Checklist */}
            <div className="pt-2 border-t border-slate-800/80">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
                Automated Quality & Safety Requirements
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {[
                  { label: "Quality Threshold", rule: "Overall score >= 80% from AI Judge" },
                  { label: "Citation Presence", rule: "Must contain verifiable [Source N] tags" },
                  { label: "Token Length", rule: "Response between 50 and 2,000 tokens" },
                  { label: "Privacy / PII Filter", rule: "No email or phone numbers in prompt" },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="flex items-start gap-2.5 p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60 text-xs"
                  >
                    <CheckCircle2 className="h-4 w-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold text-slate-200 block">{item.label}</span>
                      <span className="text-[11px] text-slate-400">{item.rule}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SFT vs DPO Summary Card */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-indigo-400" />
                <h3 className="text-sm font-semibold text-white">Dataset Summary</h3>
              </div>

              <div className="space-y-3">
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold text-slate-200 block">SFT Training Pairs</span>
                    <span className="text-[11px] text-slate-400">Supervised QA examples</span>
                  </div>
                  <span className="text-lg font-bold font-mono text-indigo-300">
                    {eligibleCount}
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold text-slate-200 block">DPO Preference Pairs</span>
                    <span className="text-[11px] text-slate-400">Chosen vs. Rejected pairs</span>
                  </div>
                  <span className="text-lg font-bold font-mono text-purple-300">
                    {dpoPairs.length}
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold text-slate-200 block">Active Fine-Tuning Jobs</span>
                    <span className="text-[11px] text-slate-400">OpenAI / MLflow runs</span>
                  </div>
                  <span className="text-lg font-bold font-mono text-emerald-300">
                    {jobs.length}
                  </span>
                </div>
              </div>
            </div>

            <div className="text-[11px] text-slate-400 bg-slate-950/40 p-3 rounded-xl border border-slate-800/60 leading-relaxed">
              <span className="font-semibold text-slate-300">Note:</span> Exporting datasets is available anytime. You do not need to wait for 10 pairs to export your SFT or DPO data.
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
          <button
            onClick={() => setActiveTab("sft")}
            className={clsx(
              "px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-2",
              activeTab === "sft"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            )}
          >
            <FileText className="h-4 w-4" />
            SFT Training Examples ({trainingPairs.length})
          </button>
          <button
            onClick={() => setActiveTab("dpo")}
            className={clsx(
              "px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-2",
              activeTab === "dpo"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            )}
          >
            <Layers className="h-4 w-4" />
            DPO Preference Pairs ({dpoPairs.length})
          </button>
          <button
            onClick={() => setActiveTab("jobs")}
            className={clsx(
              "px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-2",
              activeTab === "jobs"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            )}
          >
            <Cpu className="h-4 w-4" />
            Fine-Tuning Jobs ({jobs.length})
          </button>
        </div>

        {/* TAB 1: SFT Training Examples */}
        {activeTab === "sft" && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-white">Harvested Supervised Examples</h3>
                <p className="text-xs text-slate-400">
                  Each pair represents a high-quality user query and verified assistant response.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={canExport ? getDatasetExportUrl("sft", "jsonl") : undefined}
                  download
                  className={clsx(
                    "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors",
                    canExport
                      ? "bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800"
                      : "bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed"
                  )}
                >
                  <Download className="h-3.5 w-3.5" />
                  Download JSONL
                </a>
                <a
                  href={canExport ? getDatasetExportUrl("sft", "json") : undefined}
                  download
                  className={clsx(
                    "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors",
                    canExport
                      ? "bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800"
                      : "bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed"
                  )}
                >
                  <Download className="h-3.5 w-3.5" />
                  Download JSON
                </a>
              </div>
            </div>

            {isLoadingPairs ? (
              <div className="py-16 text-center text-slate-400 space-y-3">
                <Loader2 className="h-6 w-6 animate-spin text-indigo-500 mx-auto" />
                <p className="text-xs font-medium">Loading training examples from PostgreSQL...</p>
              </div>
            ) : trainingPairs.length === 0 ? (
              <div className="p-12 rounded-2xl bg-slate-900 border border-slate-800 text-center space-y-4">
                <div className="p-3.5 rounded-2xl bg-indigo-950/40 border border-indigo-800/30 text-indigo-400 inline-block">
                  <Database className="h-6 w-6" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-slate-200">No Training Examples Harvested Yet</h4>
                  <p className="text-xs text-slate-400 max-w-md mx-auto">
                    Ask questions in the Query Playground. When an answer achieves an automated quality score of 80% or higher, NeuroFlow will automatically record it here as a candidate.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {trainingPairs.map((pair) => {
                  const isExpanded = expandedPairId === pair.id;
                  return (
                    <div
                      key={pair.id}
                      className="rounded-2xl bg-slate-900 border border-slate-800/90 overflow-hidden transition-all"
                    >
                      <div
                        onClick={() => setExpandedPairId(isExpanded ? null : pair.id)}
                        className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer hover:bg-slate-850 transition-colors"
                      >
                        <div className="space-y-1.5 flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-slate-100 truncate">
                              {pair.user_message}
                            </span>
                          </div>
                          <p className="text-xs text-slate-400 line-clamp-1">
                            {pair.assistant_message}
                          </p>
                        </div>

                        <div className="flex items-center gap-3 flex-shrink-0">
                          <span
                            className={clsx(
                              "text-[11px] font-mono px-2 py-0.5 rounded-full border font-semibold",
                              pair.quality_score && pair.quality_score >= 0.85
                                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                : "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                            )}
                          >
                            Score: {pair.quality_score ? `${(pair.quality_score * 100).toFixed(1)}%` : "N/A"}
                          </span>

                          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                            {pair.token_count} tokens
                          </span>

                          {pair.has_citation && (
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800/40">
                              [Source Cited]
                            </span>
                          )}

                          <span
                            className={clsx(
                              "text-[11px] font-medium px-2 py-0.5 rounded-full border",
                              pair.is_valid
                                ? "bg-emerald-950/60 text-emerald-400 border-emerald-800/50"
                                : "bg-rose-950/60 text-rose-400 border-rose-800/50"
                            )}
                          >
                            {pair.is_valid ? "Valid" : "Invalid"}
                          </span>

                          {isExpanded ? (
                            <ChevronDown className="h-4 w-4 text-slate-400" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-slate-400" />
                          )}
                        </div>
                      </div>

                      {/* Expanded View */}
                      {isExpanded && (
                        <div className="p-4 border-t border-slate-800 bg-slate-950/80 space-y-4">
                          <div className="space-y-1">
                            <span className="text-[10px] font-mono text-slate-500 uppercase">User Query</span>
                            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-200">
                              {pair.user_message}
                            </div>
                          </div>

                          <div className="space-y-1">
                            <span className="text-[10px] font-mono text-slate-500 uppercase">Assistant Response</span>
                            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                              {pair.assistant_message}
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400 font-mono pt-1">
                            <span>Pair ID: {pair.id}</span>
                            <span>Run ID: {pair.run_id}</span>
                            <span>Created: {pair.created_at || "N/A"}</span>
                            {pair.rejection_reason && (
                              <span className="text-rose-400">Rejection: {pair.rejection_reason}</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* TAB 2: DPO Preference Pairs */}
        {activeTab === "dpo" && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-white">Direct Preference Optimization (DPO) Pairs</h3>
                <p className="text-xs text-slate-400">
                  Pairs where the same question received contrasting human ratings (chosen vs. rejected).
                </p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={dpoPairs.length > 0 ? getDatasetExportUrl("dpo", "jsonl") : undefined}
                  download
                  className={clsx(
                    "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors",
                    dpoPairs.length > 0
                      ? "bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800"
                      : "bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed"
                  )}
                >
                  <Download className="h-3.5 w-3.5" />
                  Download DPO JSONL
                </a>
              </div>
            </div>

            {dpoPairs.length === 0 ? (
              <div className="p-10 rounded-2xl bg-slate-900 border border-slate-800 text-center space-y-4">
                <div className="p-3.5 rounded-2xl bg-purple-950/40 border border-purple-800/30 text-purple-400 inline-block">
                  <Layers className="h-6 w-6" />
                </div>
                <div className="space-y-1.5 max-w-md mx-auto">
                  <h4 className="text-sm font-semibold text-slate-200">No Preference Pairs Yet</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    DPO examples require the same question to have both a strongly preferred response (&gt;= 4 stars) and a poorly rated response (&lt;= 2 stars).
                  </p>
                  <p className="text-xs text-indigo-400 pt-1">
                    Tip: Ask questions in the Query Playground, test different pipelines, and use the thumbs up/down rating buttons in the Live Evaluation Feed to create contrasting pairs.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {dpoPairs.map((pair, idx) => (
                  <div key={idx} className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
                    <div className="space-y-1">
                      <span className="text-[10px] font-mono text-slate-400 uppercase">Prompt</span>
                      <p className="text-xs font-semibold text-slate-100">{pair.prompt}</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-800/40 space-y-1">
                        <span className="text-[10px] font-mono text-emerald-400 font-semibold uppercase">
                          ✓ Chosen Response (&gt;= 4 Stars)
                        </span>
                        <p className="text-xs text-slate-300 whitespace-pre-wrap">{pair.chosen}</p>
                      </div>
                      <div className="p-3.5 rounded-xl bg-rose-950/20 border border-rose-800/40 space-y-1">
                        <span className="text-[10px] font-mono text-rose-400 font-semibold uppercase">
                          ✗ Rejected Response (&lt;= 2 Stars)
                        </span>
                        <p className="text-xs text-slate-300 whitespace-pre-wrap">{pair.rejected}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Fine-Tuning Jobs & Launcher */}
        {activeTab === "jobs" && (
          <div className="space-y-6">
            {/* Optional / Advanced Fine-Tuning Launcher Card */}
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-400">
                    <Zap className="h-3.5 w-3.5" />
                    Advanced Capability (Optional)
                  </div>
                  <h3 className="text-base font-semibold text-white">Fine-Tune a Custom Language Model</h3>
                  <p className="text-xs text-slate-400 max-w-xl">
                    Fine-tuning adapts a base model on your validated dataset. Unlike RAG (which retrieves facts at query time), fine-tuning specializes the model&apos;s tone, formatting, and reasoning style.
                  </p>
                </div>

                {!isOpenAIConfigured && (
                  <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/40 text-amber-300 text-xs space-y-1">
                    <div className="flex items-center gap-1.5 font-semibold">
                      <AlertCircle className="h-4 w-4 text-amber-400" />
                      Fine-Tuning Not Configured
                    </div>
                    <p className="text-[11px] text-amber-400/80">
                      Set <code className="font-mono">OPENAI_API_KEY</code> in <code className="font-mono">.env</code> to enable job submission.
                    </p>
                  </div>
                )}
              </div>

              {/* Form Controls */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Base Model</label>
                  <select
                    value={selectedBaseModel}
                    onChange={(e) => setSelectedBaseModel(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="gpt-4o-mini-2024-07-18">gpt-4o-mini-2024-07-18 (Recommended)</option>
                    <option value="gpt-4o-2024-08-06">gpt-4o-2024-08-06</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Minimum Quality Filter</label>
                  <input
                    type="text"
                    disabled
                    value=">= 82% Quality Score"
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 cursor-not-allowed"
                  />
                </div>

                <div className="flex items-end">
                  <button
                    onClick={handleLaunchJob}
                    disabled={eligibleCount < minRequired || !isOpenAIConfigured || isSubmittingJob}
                    className={clsx(
                      "w-full py-2.5 px-4 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all",
                      eligibleCount >= minRequired && isOpenAIConfigured && !isSubmittingJob
                        ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 cursor-pointer"
                        : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-800"
                    )}
                  >
                    {isSubmittingJob ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Submitting Job...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        Launch Fine-Tuning Job
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Status & Error feedback */}
              {jobError && (
                <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/40 text-rose-300 text-xs flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-rose-400 flex-shrink-0" />
                  <span>{jobError}</span>
                </div>
              )}
              {jobSuccess && (
                <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-800/40 text-emerald-300 text-xs flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                  <span>{jobSuccess}</span>
                </div>
              )}
            </div>

            {/* Historical Fine-Tuning Jobs Table */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-white">Historical & Active Jobs</h3>

              {isLoadingJobs ? (
                <div className="py-12 text-center text-slate-400 space-y-2">
                  <Loader2 className="h-5 w-5 animate-spin text-indigo-500 mx-auto" />
                  <p className="text-xs">Loading historical jobs...</p>
                </div>
              ) : jobs.length === 0 ? (
                <div className="p-10 rounded-2xl bg-slate-900 border border-slate-800 text-center space-y-2">
                  <p className="text-xs font-semibold text-slate-300">No Fine-Tuning Jobs Submitted Yet</p>
                  <p className="text-xs text-slate-500 max-w-sm mx-auto">
                    Once a fine-tuning job is launched, its live OpenAI status, MLflow experiment tracking ID, and registered model name will appear here.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto rounded-2xl bg-slate-900 border border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                      <tr>
                        <th className="py-3 px-4">Base Model</th>
                        <th className="py-3 px-4">Dataset Size</th>
                        <th className="py-3 px-4">Status</th>
                        <th className="py-3 px-4">Fine-Tuned Model</th>
                        <th className="py-3 px-4">Trained Tokens</th>
                        <th className="py-3 px-4">Created</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-slate-200">
                      {jobs.map((job) => (
                        <tr key={job.job_id} className="hover:bg-slate-850">
                          <td className="py-3 px-4 font-mono">{job.base_model}</td>
                          <td className="py-3 px-4 font-mono">{job.training_pair_count} pairs</td>
                          <td className="py-3 px-4">
                            <span
                              className={clsx(
                                "px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold border uppercase",
                                job.status === "succeeded"
                                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                                  : job.status === "running"
                                  ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40 animate-pulse"
                                  : job.status === "failed"
                                  ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                                  : "bg-slate-800 text-slate-300 border-slate-700"
                              )}
                            >
                              {job.status}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-mono text-indigo-300">
                            {job.fine_tuned_model || "Pending completion"}
                          </td>
                          <td className="py-3 px-4 font-mono">
                            {job.metrics?.trained_tokens ? `${job.metrics.trained_tokens.toLocaleString()} tokens` : "Not reported"}
                          </td>
                          <td className="py-3 px-4 text-slate-400 font-mono text-[11px]">
                            {job.created_at || "N/A"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
