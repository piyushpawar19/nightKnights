from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, confloat


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

class RetrievalConfig(BaseModel):
    """Configuration settings for the candidate retrieval module."""

    embedding_model: str = Field(..., description="Name or path of the embedding model.")
    embedding_dimension: PositiveInt = Field(..., description="Dimension of the embeddings.")
    faiss_index_type: str = Field(
        "IVF", description="Type of FAISS index to use (e.g., IVF, HNSW)."
    )
    dense_top_k: PositiveInt = Field(
        50, description="Number of top candidates for dense retrieval."
    )
    bm25_top_k: PositiveInt = Field(
        50, description="Number of top candidates for BM25 retrieval."
    )
    hybrid_top_k: PositiveInt = Field(
        100, description="Number of top candidates after hybrid retrieval."
    )
    cache_embeddings: bool = Field(
        True, description="Whether to cache candidate embeddings."
    )
    batch_size: PositiveInt = Field(
        32, description="Batch size for embedding generation/retrieval."
    )

    model_config = {"frozen": True}


class RankingConfig(BaseModel):
    """Configuration settings for the candidate ranking module."""

    dense_weight: confloat(ge=0.0, le=1.0) = Field(
        0.5, description="Weight for dense retrieval scores in hybrid ranking."
    )
    bm25_weight: confloat(ge=0.0, le=1.0) = Field(
        0.5, description="Weight for BM25 scores in hybrid ranking."
    )
    feature_weight: confloat(ge=0.0, le=1.0) = Field(
        0.5, description="Weight for engineered features in hybrid ranking."
    )
    rerank_top_k: PositiveInt = Field(
        20, description="Number of top candidates to send for re-ranking."
    )
    feature_thresholds: dict[str, PositiveFloat] = Field(
        default_factory=dict,
        description='Thresholds for specific features (e.g., {"skill_match": 0.7}).',
    )
    normalization_method: str = Field(
        "min_max", description="Method for score normalization (e.g., min_max, z_score)."
    )

    model_config = {"frozen": True}


class LLMConfig(BaseModel):
    """Configuration settings for Large Language Models (LLMs)."""

    provider: str = Field(..., description="LLM provider (e.g., OpenAI, Google, Anthropic).")
    model_name: str = Field(..., description="Specific model name (e.g., gpt-4, gemini-pro).")
    temperature: confloat(ge=0.0, le=2.0) = Field(
        0.7, description="Sampling temperature for text generation."
    )
    max_tokens: PositiveInt = Field(1024, description="Maximum tokens in LLM response.")
    timeout: PositiveFloat = Field(60.0, description="Request timeout in seconds.")
    retry_attempts: PositiveInt = Field(3, description="Number of retry attempts for LLM calls.")
    prompt_directory: Path = Field(
        Path("src/prompts"), description="Filesystem path to the prompt templates."
    )

    model_config = {"frozen": True}


class EvaluationConfig(BaseModel):
    """Configuration settings for the evaluation module."""

    output_dir: Path = Field(
        Path("outputs/reports"), description="Directory for evaluation reports."
    )
    enabled_metrics: list[str] = Field(
        default_factory=lambda: [
            "recall_at_k",
            "precision_at_k",
            "hit_rate",
            "mrr",
            "ndcg_at_k",
            "map",
            "explanation_success_rate",
            "total_pipeline_runtime_ms",
        ],
        description="List of metrics to enable for evaluation.",
    )
    k_values: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"recall_k": 10, "precision_k": 10, "ndcg_k": 10, "top_k_accuracy_k": 5},
        description="Default K values for metrics that require them.",
    )
    report_formats: list[str] = Field(
        default_factory=lambda: ["json", "md"],
        description="List of formats for evaluation reports (json, csv, md).",
    )
    pipeline_version: str = Field("1.0.0", description="Version of the pipeline being evaluated.")

    model_config = {"frozen": True}


class ExportConfig(BaseModel):
    """Configuration settings for the export service."""

    output_dir: Path = Field(
        Path("outputs/submissions"), description="Directory for export files."
    )
    filename: str = Field("submission.csv", description="Default export filename.")
    delimiter: str = Field(",", description="CSV delimiter.")
    quotechar: str = Field("\"", description="CSV quote character.")
    quoting: int = Field(0, description="CSV quoting style (0=QUOTE_MINIMAL, 1=QUOTE_ALL, 2=QUOTE_NONNUMERIC, 3=QUOTE_NONE).")
    encoding: str = Field("utf-8", description="File encoding.")
    overwrite: bool = Field(False, description="Whether to overwrite existing files.")
    export_schema: list[str] = Field(
        default_factory=lambda: [
            "candidate_id",
            "rank",
            "hybrid_score",
            "recruiter_technical_score",
            "recruiter_career_score",
            "recruiter_behavior_score",
            "recruiter_risk_score",
            "recruiter_culture_fit",
            "recruiter_hiring_confidence",
            "recruiter_final_score",
            "explanation_summary",
            "explanation_strengths",
            "explanation_weaknesses",
            "explanation_reasoning",
            "explanation_recommendation",
            "explanation_confidence",
        ],
        description="List of fields to include in the export, in order.",
    )

    model_config = {"frozen": True}


class BenchmarkConfig(BaseModel):
    """Configuration settings for the benchmarking module."""
    base_output_dir: Path = Field(
        Path("outputs/benchmarks"), description="Base directory for benchmark results."
    )
    benchmark_runs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Definitions for individual benchmark runs, with config overrides.",
    )
    default_k_values: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"recall_k": 10, "precision_k": 10, "ndcg_k": 10, "top_k_accuracy_k": 5},
        description="Default K values for metrics within benchmarks.",
    )
    report_format: str = Field("json", description="Default report format for benchmarks.")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Top-level application configuration
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Top-level application configuration, composed of sub-configs."""

    retrieval: RetrievalConfig = Field(..., description="Retrieval module configuration.")
    ranking: RankingConfig = Field(..., description="Ranking module configuration.")
    llm: LLMConfig = Field(..., description="LLM interaction configuration.")
    evaluation: EvaluationConfig = Field(..., description="Evaluation module configuration.")
    export: ExportConfig = Field(..., description="Export service configuration.")
    benchmark: BenchmarkConfig = Field(..., description="Benchmarking module configuration.")

    model_config = {"frozen": True}
