"""Pydantic schemas for validated data structures used throughout the pipeline.

This module defines the strongly-typed data models that flow through the
LangGraph orchestration pipeline. Each model enforces runtime validation
so that downstream nodes can trust the shape and constraints of their inputs.

Design Principles:
    - Models are frozen (immutable) where practical to prevent accidental mutation.
    - Field descriptions mirror the upstream candidate_schema.json where applicable.
    - All models use strict validation to catch type coercion bugs early.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    """Status of an individual pipeline node execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class SkillProficiency(str, Enum):
    """Proficiency levels matching candidate_schema.json."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# ---------------------------------------------------------------------------
# JD Schemas
# ---------------------------------------------------------------------------

class SkillRequirement(BaseModel):
    """A single skill requirement extracted from a job description."""

    name: str = Field(..., description="Skill name (e.g. 'Python', 'System Design').")
    importance: str = Field(
        default="required",
        description="How critical the skill is: 'required', 'preferred', or 'nice_to_have'.",
    )
    min_years: float | None = Field(
        default=None,
        description="Minimum years of experience with this skill, if specified.",
    )

    model_config = {"frozen": True}


# (StructuredJD removed as it is replaced by ParsedJD from jd_schema.py)


# ---------------------------------------------------------------------------
# Candidate Schemas
# ---------------------------------------------------------------------------

class CandidateRecord(BaseModel):
    """A candidate record retrieved from the candidate pool.

    This is a simplified view; the full schema is in candidate_schema.json.
    Nodes that need the full record should work with the raw dict.
    """

    candidate_id: str = Field(..., pattern=r"^CAND_\d{7}$")
    headline: str = Field(default="")
    summary: str = Field(default="")
    years_of_experience: float = Field(default=0.0, ge=0, le=50)
    current_title: str = Field(default="")
    current_company: str = Field(default="")
    skills: list[str] = Field(default_factory=list, description="Flat list of skill names.")
    location: str = Field(default="")
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete raw candidate dict for full-fidelity access.",
    )

    model_config = {"frozen": True}


class CandidateScore(BaseModel):
    """A scored and ranked candidate produced by ranking / re-ranking nodes."""

    candidate_id: str = Field(..., pattern=r"^CAND_\d{7}$")
    rank: int = Field(..., ge=1, description="1-indexed rank position.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Normalised 0-1 score.")
    component_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown by scoring dimension (e.g. 'skill_match': 0.85).",
    )

    model_config = {"frozen": True}


class FeatureVector(BaseModel):
    """Engineered feature vector for a single candidate."""

    candidate_id: str = Field(..., pattern=r"^CAND_\d{7}$")
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Named features and their numeric values.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Explanation Schemas
# ---------------------------------------------------------------------------

class ExplanationRecord(BaseModel):
    """Human-readable explanation for why a candidate was ranked at a position."""

    candidate_id: str = Field(..., pattern=r"^CAND_\d{7}$")
    rank: int = Field(..., ge=1)
    reasoning: str = Field(
        ...,
        min_length=10,
        description="1-2 sentence reasoning for the ranking decision.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Evaluation Schemas
# ---------------------------------------------------------------------------

class EvaluationMetrics(BaseModel):
    """Evaluation metrics computed after ranking to assess quality."""

    honeypot_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of top-100 that are honeypot candidates.",
    )
    skill_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of required skills covered by top candidates.",
    )
    diversity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Diversity metric across top candidates.",
    )
    avg_experience_fit: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well candidate experience matches JD requirements.",
    )
    custom_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Extensible slot for team-specific metrics.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Execution Metadata Schemas
# ---------------------------------------------------------------------------

class NodeTimestamp(BaseModel):
    """Timing record for a single node execution."""

    node_name: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0.0)
    status: NodeStatus = NodeStatus.SUCCESS

    model_config = {"frozen": True}


class NodeError(BaseModel):
    """Error record captured when a node fails."""

    node_name: str
    error_type: str = Field(description="Exception class name.")
    error_message: str
    timestamp: datetime
    traceback: str | None = Field(default=None, description="Full traceback string.")

    model_config = {"frozen": True}


class ExecutionMetadata(BaseModel):
    """Top-level metadata about a pipeline run."""

    run_id: str = Field(..., description="Unique identifier for this pipeline invocation.")
    pipeline_version: str = Field(default="0.1.0")
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    total_duration_ms: float | None = Field(default=None, ge=0.0)
    node_count: int = Field(default=8, ge=0)
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Frozen copy of configuration used for this run.",
    )
