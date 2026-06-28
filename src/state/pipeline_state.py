"""Pipeline state definition for the LangGraph orchestration layer.

This module defines ``PipelineState`` — the central TypedDict that flows
through every node in the candidate-ranking graph.  Using ``TypedDict``
keeps the state lightweight and natively compatible with LangGraph's
``StateGraph``, while the heavy validation lives in the Pydantic schemas
imported from ``src.schemas.graph_schema``.

Key design choices
------------------
* ``errors`` and ``timestamps`` use ``Annotated[list, operator.add]`` so that
  each node can *append* to these lists without overwriting entries from
  prior nodes (LangGraph reducer semantics).
* All other fields use simple overwrite semantics — the last node to write
  a field wins.
* ``create_initial_state()`` is the single factory for constructing a
  correctly-initialised state dict; avoid building one by hand.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from src.schemas.graph_schema import (
    CandidateRecord,
    CandidateScore,
    EvaluationMetrics,
    ExecutionMetadata,
    ExplanationRecord,
    FeatureVector,
    NodeError,
    NodeTimestamp,
    StructuredJD,
)


class PipelineState(TypedDict, total=False):
    """Global state passed through every node in the LangGraph pipeline.

    Fields
    ------
    raw_jd : str
        The verbatim job-description text supplied by the user.
    structured_jd : dict | None
        Serialised ``StructuredJD`` produced by the JD-parsing node.
    extracted_skills : list[str]
        Flat list of skill names extracted from the JD.
    retrieved_candidates : list[dict]
        Serialised ``CandidateRecord`` dicts from the retrieval node.
    feature_vectors : list[dict]
        Serialised ``FeatureVector`` dicts from feature engineering.
    ranked_candidates : list[dict]
        Serialised ``CandidateScore`` dicts from hybrid ranking.
    reranked_candidates : list[dict]
        Serialised ``CandidateScore`` dicts after recruiter re-ranking.
    explanations : list[dict]
        Serialised ``ExplanationRecord`` dicts from the explanation node.
    evaluation_metrics : dict | None
        Serialised ``EvaluationMetrics`` from the evaluation node.
    submission_path : str | None
        Filesystem path to the generated CSV submission file.
    execution_metadata : dict | None
        Serialised ``ExecutionMetadata`` for the current run.
    errors : Annotated[list[dict], operator.add]
        Accumulated ``NodeError`` dicts — uses a *reducer* so every node
        can append without overwriting previous errors.
    timestamps : Annotated[list[dict], operator.add]
        Accumulated ``NodeTimestamp`` dicts — same reducer semantics.
    """

    raw_jd: str
    structured_jd: dict[str, Any] | None
    extracted_skills: list[str]
    retrieved_candidates: list[dict[str, Any]]
    feature_vectors: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    reranked_candidates: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    evaluation_metrics: dict[str, Any] | None
    submission_path: str | None
    execution_metadata: dict[str, Any] | None
    errors: Annotated[list[dict[str, Any]], operator.add]
    timestamps: Annotated[list[dict[str, Any]], operator.add]
    # Integration bridge fields for external module adapters (Divyansh / Piyush).
    candidate_profiles: dict[str, Any]
    candidates_with_features: list[Any]
    user_preferences: dict[str, Any]
    raw_candidate_data: list[Any]


def create_initial_state(raw_jd: str) -> PipelineState:
    """Create a correctly-initialised ``PipelineState`` for a new pipeline run.

    Parameters
    ----------
    raw_jd : str
        The raw job-description text to process.

    Returns
    -------
    PipelineState
        A state dict with all fields set to safe defaults and a fresh
        ``execution_metadata`` containing a unique ``run_id``.
    """

    run_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc)

    metadata = ExecutionMetadata(
        run_id=run_id,
        started_at=now,
    )

    return PipelineState(
        raw_jd=raw_jd,
        structured_jd=None,
        extracted_skills=[],
        retrieved_candidates=[],
        feature_vectors=[],
        ranked_candidates=[],
        reranked_candidates=[],
        explanations=[],
        evaluation_metrics=None,
        submission_path=None,
        execution_metadata=metadata.model_dump(mode="json"),
        errors=[],
        timestamps=[],
    )
