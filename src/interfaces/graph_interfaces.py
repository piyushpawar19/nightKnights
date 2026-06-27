"""Interface contracts (Protocols) for pipeline module implementations.

These ``Protocol`` classes define the *structural* contracts that future
team implementations must satisfy.  Because they are protocols, implementors
do **not** need to inherit from them — they just need to expose methods with
matching signatures.  This is *structural subtyping* (duck typing with
static-analysis support).

Usage by teams
--------------
Retrieval Team::

    class MyRetriever:  # no base class needed
        def retrieve(self, structured_jd, *, top_k=200):
            ...

The orchestration layer will accept any object whose public API matches
the corresponding protocol.

Extension Points
----------------
To add a new interface:

1. Define a new ``Protocol`` class below.
2. Reference it in the relevant node (``src/graph/nodes.py``).
3. Future implementors satisfy the contract by implementing the method(s).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.schemas.graph_schema import (
    CandidateRecord,
    CandidateScore,
    EvaluationMetrics,
    ExplanationRecord,
    FeatureVector,
    StructuredJD,
)


@runtime_checkable
class JDParserInterface(Protocol):
    """Contract for job-description parsing implementations.

    The parser receives raw JD text and produces a structured
    representation suitable for downstream retrieval and ranking.
    """

    def parse(self_,
              raw_jd: str,
              config: dict[str, Any] | None = None) -> StructuredJD:
        """Parse raw job-description text into a structured model.

        Parameters
        ----------
        raw_jd : str
            The verbatim job-description text.
        config : dict[str, Any], optional
            Configuration parameters for the parser, by default None.

        Returns
        -------
        StructuredJD
            Parsed and validated JD structure.
        """
        ...


@runtime_checkable
class RetrieverInterface(Protocol):
    """Contract for candidate retrieval implementations.

    Given a structured JD, the retriever should return a broad set of
    potentially-relevant candidates from the pool.
    """

    def retrieve(
        self,
        structured_jd: StructuredJD,
        *,
        top_k: int = 200,
        config: dict[str, Any] | None = None,
    ) -> list[CandidateRecord]:
        """Retrieve candidates matching the structured JD.

        Parameters
        ----------
        structured_jd : StructuredJD
            The parsed job description.
        top_k : int
            Maximum number of candidates to retrieve.
        config : dict[str, Any], optional
            Configuration parameters for the retriever, by default None.

        Returns
        -------
        list[CandidateRecord]
            Retrieved candidate records.
        """
        ...


@runtime_checkable
class FeatureEngineerInterface(Protocol):
    """Contract for feature-engineering implementations."""

    def engineer_features(
        self,
        candidates: list[CandidateRecord],
        structured_jd: StructuredJD,
        config: dict[str, Any] | None = None,
    ) -> list[FeatureVector]:
        """Compute feature vectors for each candidate relative to the JD.

        Parameters
        ----------
        candidates : list[CandidateRecord]
            Retrieved candidates.
        structured_jd : StructuredJD
            Parsed job description.
        config : dict[str, Any], optional
            Configuration parameters for feature engineering, by default None.

        Returns
        -------
        list[FeatureVector]
            Feature vectors aligned with the candidate list.
        """
        ...


@runtime_checkable
class RankerInterface(Protocol):
    """Contract for hybrid-ranking implementations."""

    def rank(
        self,
        candidates: list[CandidateRecord],
        feature_vectors: list[FeatureVector],
        structured_jd: StructuredJD,
        config: dict[str, Any] | None = None,
    ) -> list[CandidateScore]:
        """Produce a ranked list of candidates using hybrid scoring.

        Parameters
        ----------
        candidates : list[CandidateRecord]
            Retrieved candidates.
        feature_vectors : list[FeatureVector]
            Pre-computed feature vectors.
        structured_jd : StructuredJD
            Parsed job description.
        config : dict[str, Any], optional
            Configuration parameters for the ranker, by default None.

        Returns
        -------
        list[CandidateScore]
            Candidates with scores and rank positions.
        """
        ...


@runtime_checkable
class RerankerInterface(Protocol):
    """Contract for recruiter-style re-ranking implementations."""

    def rerank(
        self,
        ranked_candidates: list[CandidateScore],
        structured_jd: StructuredJD,
        config: dict[str, Any] | None = None,
    ) -> list[CandidateScore]:
        """Re-rank candidates using recruiter-quality signals.

        Parameters
        ----------
        ranked_candidates : list[CandidateScore]
            Initially ranked candidates.
        structured_jd : StructuredJD
            Parsed job description.
        config : dict[str, Any], optional
            Configuration parameters for the reranker, by default None.

        Returns
        -------
        list[CandidateScore]
            Re-ranked candidates with updated scores/positions.
        """
        ...


@runtime_checkable
class ExplainerInterface(Protocol):
    """Contract for explanation-generation implementations."""

    def explain(
        self,
        reranked_candidates: list[CandidateScore],
        structured_jd: StructuredJD,
        config: dict[str, Any] | None = None,
    ) -> list[ExplanationRecord]:
        """Generate human-readable explanations for ranking decisions.

        Parameters
        ----------
        reranked_candidates : list[CandidateScore]
            Final ranked candidates.
        structured_jd : StructuredJD
            Parsed job description.
        config : dict[str, Any], optional
            Configuration parameters for the explainer, by default None.

        Returns
        -------
        list[ExplanationRecord]
            One explanation per candidate.
        """
        ...


@runtime_checkable
class EvaluatorInterface(Protocol):
    """Contract for evaluation implementations."""

    def evaluate(
        self,
        reranked_candidates: list[CandidateScore],
        explanations: list[ExplanationRecord],
        structured_jd: StructuredJD,
        config: dict[str, Any] | None = None,
    ) -> EvaluationMetrics:
        """Evaluate the quality of the ranking output.

        Parameters
        ----------
        reranked_candidates : list[CandidateScore]
            Final ranked candidates.
        explanations : list[ExplanationRecord]
            Generated explanations.
        structured_jd : StructuredJD
            Parsed job description.
        config : dict[str, Any], optional
            Configuration parameters for the evaluator, by default None.

        Returns
        -------
        EvaluationMetrics
            Quality metrics for the current run.
        """
        ...
