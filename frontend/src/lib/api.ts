import axios from "axios";
import {
  Pipeline,
  PipelineConfig,
  QueryResponse,
  CompareResponse,
  Evaluation,
  PipelineAnalytics,
  DocumentItem,
  ChunkItem,
  SimilarChunk,
  DatasetReadiness,
  TrainingPairDetail,
  DPOPair,
  FinetuneJob,
} from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

// 1. Pipelines
export const fetchPipelines = async (): Promise<Pipeline[]> => {
  const res = await apiClient.get("/pipelines");
  return res.data;
};

export const createPipeline = async (config: PipelineConfig): Promise<Pipeline> => {
  const res = await apiClient.post("/pipelines", config);
  return res.data;
};

export const deletePipeline = async (pipelineId: string): Promise<any> => {
  const res = await apiClient.delete(`/pipelines/${pipelineId}`);
  return res.data;
};

export const fetchPipelineAnalytics = async (pipelineId: string): Promise<PipelineAnalytics> => {
  const res = await apiClient.get(`/pipelines/${pipelineId}/analytics`);
  const d = res.data;
  return {
    pipeline_id: d.pipeline_id || pipelineId,
    pipeline_name: d.pipeline_name || "Pipeline",
    total_runs: d.total_runs ?? 0,
    latency: {
      p50_ms: d.latency_p50_ms ?? d.retrieval_latency_p50 ?? 0,
      p95_ms: d.latency_p95_ms ?? d.retrieval_latency_p95 ?? 0,
      p99_ms: d.latency_p99_ms ?? d.retrieval_latency_p99 ?? 0,
      avg_ms: d.avg_latency_ms ?? d.avg_generation_latency_ms ?? 0,
    },
    evaluation_scores: {
      faithfulness: d.avg_evaluation_scores?.faithfulness ?? null,
      answer_relevance: d.avg_evaluation_scores?.answer_relevance ?? null,
      context_precision: d.avg_evaluation_scores?.context_precision ?? null,
      context_recall: d.avg_evaluation_scores?.context_recall ?? null,
      overall_score: d.avg_evaluation_scores?.overall_score ?? null,
    },
    cost: {
      cost_per_query_usd: d.cost_per_query_usd ?? d.cost_per_query ?? 0,
      total_cost_usd: d.total_cost_usd ?? 0,
    },
    queries_per_day: d.queries_per_day || [],
    recent_failures: d.recent_failures || [],
  };
};

// 2. Query & Generation
export const startQueryStream = async (params: {
  query: string;
  pipeline_id?: string;
}): Promise<{ run_id: string; status: string }> => {
  const res = await apiClient.post("/query", {
    query: params.query,
    pipeline_id: params.pipeline_id,
    stream: true,
  });
  return res.data;
};

export const executeCompare = async (params: {
  query: string;
  pipeline_a_id: string;
  pipeline_b_id: string;
}): Promise<CompareResponse> => {
  const res = await apiClient.post("/pipelines/compare", {
    query: params.query,
    pipeline_a_id: params.pipeline_a_id,
    pipeline_b_id: params.pipeline_b_id,
  });
  return res.data;
};

// 3. Evaluations & Ratings
export const fetchEvaluations = async (filters?: {
  pipeline_id?: string;
  min_overall?: number;
  min_faithfulness?: number;
  min_relevance?: number;
  min_precision?: number;
  min_recall?: number;
  limit?: number;
}): Promise<Evaluation[]> => {
  const res = await apiClient.get("/evaluations", { params: filters });
  return res.data;
};

export const fetchRunEvaluation = async (runId: string): Promise<Evaluation> => {
  const res = await apiClient.get(`/runs/${runId}/evaluation`);
  return res.data;
};

export const submitHumanRating = async (runId: string, rating: number): Promise<Evaluation> => {
  const res = await apiClient.patch(`/runs/${runId}/rating`, { rating });
  return res.data;
};

// 4. Documents & Ingestion
export const fetchDocuments = async (): Promise<DocumentItem[]> => {
  const res = await apiClient.get("/documents");
  return res.data;
};

export const fetchDocumentChunks = async (documentId: string): Promise<ChunkItem[]> => {
  const res = await apiClient.get(`/documents/${documentId}/chunks`);
  return res.data;
};

export const findSimilarDocumentChunks = async (
  documentId: string,
  query: string,
  limit = 5
): Promise<SimilarChunk[]> => {
  const res = await apiClient.post(`/documents/${documentId}/similar`, { query, limit });
  return res.data;
};

export const uploadDocument = async (
  file: File,
  pipelineId?: string,
  onProgress?: (percent: number) => void
): Promise<any> => {
  const formData = new FormData();
  formData.append("file", file);
  if (pipelineId) {
    formData.append("pipeline_id", pipelineId);
  }

  const res = await apiClient.post("/ingest", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percent);
      }
    },
  });
  return res.data;
};

// 5. Fine-Tuning & Datasets
export const fetchDatasetReadiness = async (): Promise<DatasetReadiness> => {
  const res = await apiClient.get("/finetune/readiness");
  return res.data;
};

export const fetchTrainingPairs = async (limit = 100): Promise<TrainingPairDetail[]> => {
  const res = await apiClient.get("/finetune/training-data", { params: { limit } });
  return res.data;
};

export const fetchDPOPreview = async (): Promise<DPOPair[]> => {
  const res = await apiClient.get("/finetune/dpo/preview");
  return res.data;
};

export const fetchFinetuneJobs = async (): Promise<FinetuneJob[]> => {
  const res = await apiClient.get("/finetune/jobs");
  return res.data;
};

export const createFinetuneJob = async (params: {
  base_model: string;
  min_quality_score?: number;
}): Promise<FinetuneJob> => {
  const res = await apiClient.post("/finetune/jobs", params);
  return res.data;
};

export const getDatasetExportUrl = (type: "sft" | "dpo" = "sft", format: "jsonl" | "json" = "jsonl"): string => {
  return `${API_BASE}/finetune/datasets/export?dataset_type=${type}&format=${format}`;
};

