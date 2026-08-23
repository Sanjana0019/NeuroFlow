from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class IngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunking_strategy: str = Field(default="recursive", min_length=1)
    chunk_size_tokens: int = Field(..., gt=0, description="Target chunk size in tokens")
    chunk_overlap_tokens: int = Field(..., ge=0, description="Chunk overlap in tokens")
    extractors_enabled: list[str] = Field(default_factory=list, description="List of enabled extractors")


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_k: int = Field(..., gt=0, description="Dense retrieval top-k count")
    sparse_k: int = Field(..., gt=0, description="Sparse BM25 retrieval top-k count")
    reranker: str = Field(default="bge-reranker-large", min_length=1)
    top_k_after_rerank: int = Field(..., gt=0, description="Final top-k chunks after reranking")
    query_expansion: bool = Field(default=True, description="Whether query expansion is enabled")
    metadata_filters_enabled: bool = Field(default=True, description="Whether metadata filters are active")


class ModelRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str = Field(default="rag_generation", min_length=1)
    max_cost_per_call: float = Field(..., ge=0.0, description="Max cost budget per LLM invocation in USD")


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_routing: ModelRoutingConfig
    max_context_tokens: int = Field(..., gt=0, description="Maximum token budget for context assembly")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    system_prompt_variant: str = Field(default="default", min_length=1)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_evaluate: bool = Field(default=True, description="Whether to enqueue automated evaluation")
    training_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Quality score threshold for training pairs")


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique pipeline identifier name")
    description: str | None = Field(default=None, description="Human-readable pipeline description")
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
