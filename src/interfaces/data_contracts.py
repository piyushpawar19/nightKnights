from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.schemas.jd_schema import StructuredJD, SkillTaxonomy
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.retrieval_schema import RetrievalResult
from src.schemas.ranking_schema import RankedCandidate, FeatureVector
from src.schemas.explanation_schema import Explanation, RecruiterAssessment
from src.schemas.evaluation_schema import EvaluationReport
from src.schemas.submission_schema import SubmissionRecord


class JDParserOutput(ABC):
    """Abstract contract for the output of a Job Description parsing module."""

    @abstractmethod
    def get_structured_jd(self) -> StructuredJD:
        """Returns the structured job description."""
        pass

    @abstractmethod
    def get_skill_taxonomy(self) -> SkillTaxonomy:
        """Returns the skill taxonomy extracted from the JD."""
        pass


class RetrievalOutput(ABC):
    """Abstract contract for the output of a candidate retrieval module."""

    @abstractmethod
    def get_retrieval_results(self) -> List[RetrievalResult]:
        """Returns a list of retrieval results."""
        pass


class FeatureEngineeringOutput(ABC):
    """Abstract contract for the output of a feature engineering module."""

    @abstractmethod
    def get_feature_vectors(self) -> List[FeatureVector]:
        """Returns a list of engineered feature vectors."""
        pass


class RankingOutput(ABC):
    """Abstract contract for the output of a candidate ranking module."""

    @abstractmethod
    def get_ranked_candidates(self) -> List[RankedCandidate]:
        """Returns a list of ranked candidates."""
        pass


class RecruiterAssessmentOutput(ABC):
    """Abstract contract for the output of a recruiter assessment module."""

    @abstractmethod
    def get_recruiter_assessments(self) -> List[RecruiterAssessment]:
        """Returns a list of recruiter assessments."""
        pass


class ExplanationOutput(ABC):
    """Abstract contract for the output of an explanation generation module."""

    @abstractmethod
    def get_explanations(self) -> List[Explanation]:
        """Returns a list of explanations for ranked candidates."""
        pass


class EvaluationOutput(ABC):
    """Abstract contract for the output of an evaluation module."""

    @abstractmethod
    def get_evaluation_report(self) -> EvaluationReport:
        """Returns the comprehensive evaluation report."""
        pass


class SubmissionOutput(ABC):
    """Abstract contract for the final submission output."""

    @abstractmethod
    def get_submission_records(self) -> List[SubmissionRecord]:
        """Returns a list of submission records."""
        pass


class DataIngestionInput(ABC):
    """Abstract contract for the input to a data ingestion module."""

    @abstractmethod
    def get_raw_jds(self) -> List[str]:
        """Returns a list of raw job description texts."""
        pass

    @abstractmethod
    def get_raw_candidate_data(self) -> List[Dict[str, Any]]:
        """Returns a list of raw candidate data, e.g., from JSON or databases."""
        pass


class CandidateProfileOutput(ABC):
    """Abstract contract for the output of a candidate profile processing module."""

    @abstractmethod
    def get_candidate_profiles(self) -> List[CandidateProfile]:
        """Returns a list of processed candidate profiles."""
        pass