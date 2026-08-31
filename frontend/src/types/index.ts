export interface PipelineSummaryMetrics {
  has_data: boolean;
  quality_score: number | null;
  quality_label: string;
  faithfulness: number | null;
  faithfulness_change_pct: number | null;
  queries_7d: number;
  queries_previous_7d: number;
  queries_change_pct: number | null;
  latency_p50_ms: number | null;
  latency_change_pct: number | null;
  trend_7d: { date: string; query_count: number }[];
}

export interface Pipeline {
  id: string;
  name: string;
  version: number;
  description?: string;
  status?: string;
  config: PipelineConfig;
  created_at: string;
  updated_at?: string;
  is_active?: boolean;
  last_run_metrics?: {
    total_runs?: number;
    last_run_at?: string | null;
    last_run_status?: string | null;
    last_run_latency_ms?: number | null;
  };
  metrics_summary?: PipelineSummaryMetrics;
  average_score?: number;
  query_count_7d?: number;
  sparkline_scores?: number[];
}

export interface IngestionConfig {
  chunking_strategy: string;
  chunk_size_tokens: number;
  chunk_overlap_tokens: number;
  extractors_enabled: string[];
}

export interface RetrievalConfig {
  dense_k: number;
  sparse_k: number;
  reranker: string;
  top_k_after_rerank: number;
  query_expansion: boolean;
  metadata_filters_enabled: boolean;
}

export interface ModelRoutingConfig {
  task_type: string;
  max_cost_per_call: number;
}

export interface GenerationConfig {
  model_routing: ModelRoutingConfig;
  max_context_tokens: number;
  temperature: number;
  system_prompt_variant: string;
}

export interface EvaluationConfig {
  auto_evaluate: boolean;
  training_threshold: number;
}

export interface PipelineConfig {
  name: string;
  description?: string;
  ingestion: IngestionConfig;
  retrieval: RetrievalConfig;
  generation: GenerationConfig;
  evaluation: EvaluationConfig;
}

export interface Citation {
  reference: string;
  chunk_id: string | null;
  document_name: string;
  page_number: number | null;
  content_preview: string;
  invalid_citation?: boolean;
}

export interface Source {
  source_index: number;
  document_id: string;
  filename: string;
  page_number: number | null;
  chunk_id?: string;
  content?: string;
  score?: number;
  metadata?: Record<string, any>;
}

export interface RetrievalStageCounts {
  dense?: number;
  sparse?: number;
  metadata?: number;
  rrf?: number;
  reranker?: number;
  final_context?: number;
}

export interface QueryResponse {
  run_id: string;
  query: string;
  generation: string;
  citations: Citation[];
  sources: Source[];
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  model_used: string;
  status: string;
  stage_counts?: RetrievalStageCounts;
}

export interface CompareResponse {
  run_id_a: string;
  run_id_b: string;
  pipeline_a: {
    id: string;
    name: string;
    version: number;
    run_id: string;
    query: string;
    generation: string;
    citations: Citation[];
    sources: Source[];
    latency_ms: number;
    chunks_used: number;
    evaluation?: Evaluation;
  };
  pipeline_b: {
    id: string;
    name: string;
    version: number;
    run_id: string;
    query: string;
    generation: string;
    citations: Citation[];
    sources: Source[];
    latency_ms: number;
    chunks_used: number;
    evaluation?: Evaluation;
  };
}

export interface Evaluation {
  id?: string;
  run_id: string;
  query?: string;
  generation?: string;
  pipeline_name?: string;
  faithfulness: number | null;
  answer_relevance: number | null;
  context_precision: number | null;
  context_recall: number | null;
  overall_score: number | null;
  judge_model?: string | null;
  user_rating?: number | null;
  calibration_needed?: boolean;
  evaluated_at?: string;
  chunks?: { id: string; content: string; chunk_index: number }[];
}

export interface PipelineAnalytics {
  pipeline_id: string;
  pipeline_name: string;
  total_runs: number;
  latency: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    avg_ms: number;
  };
  evaluation_scores: {
    faithfulness: number | null;
    answer_relevance: number | null;
    context_precision: number | null;
    context_recall: number | null;
    overall_score: number | null;
  };
  cost: {
    cost_per_query_usd: number;
    total_cost_usd: number;
  };
  queries_per_day: {
    date: string;
    count: number;
  }[];
  recent_failures: {
    run_id: string;
    query: string;
    timestamp: string;
    error_message: string;
  }[];
}

export interface DocumentItem {
  id: string;
  filename: string;
  source_type: string;
  status: "pending" | "processing" | "completed" | "failed";
  chunk_count: number;
  metadata: Record<string, any>;
  created_at: string;
}

export interface ChunkItem {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  token_count: number;
  metadata: Record<string, any>;
  similarity_score?: number;
}

export interface SimilarChunk {
  id: string;
  chunk_index: number;
  content: string;
  similarity_score: number;
}

export interface ValidationRule {
  name: string;
  requirement: string;
  passed: boolean;
}

export interface DatasetReadiness {
  total_candidates: number;
  eligible_sft_count: number;
  min_required_for_finetuning: number;
  remaining_for_finetuning: number;
  can_export: boolean;
  can_finetune: boolean;
  openai_configured: boolean;
  dpo_pair_count: number;
  validation_rules: ValidationRule[];
  rejected_count: number;
  rejection_summary: Record<string, number>;
}

export interface TrainingPairDetail {
  id: string;
  run_id: string;
  user_message: string;
  assistant_message: string;
  system_prompt: string;
  quality_score: number | null;
  faithfulness: number | null;
  answer_relevance: number | null;
  token_count: number;
  has_citation: boolean;
  has_pii: boolean;
  user_rating: number | null;
  is_valid: boolean;
  rejection_reason: string | null;
  included_in_job: string | null;
  created_at: string | null;
}

export interface DPOPair {
  prompt: string;
  chosen: string;
  rejected: string;
}

export interface FinetuneJob {
  job_id: string;
  provider_job_id: string | null;
  base_model: string;
  fine_tuned_model: string | null;
  status: "pending" | "running" | "succeeded" | "failed" | "cancelled";
  training_pair_count: number;
  mlflow_run_id: string | null;
  metrics: {
    trained_tokens?: number;
    training_loss?: number;
    validation_loss?: number;
    fine_tuned_model?: string;
  } | null;
  created_at: string | null;
  completed_at: string | null;
}

