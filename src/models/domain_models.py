from src.schemas.common_schema import (
    Skill,
    Education,
    Experience,
    Project,
    Certification,
    Location,
    Company,
    Timestamp,
    Score,
    Metadata,
    UUIDMixin
)
from src.schemas.jd_schema import (
    StructuredJD,
    SkillTaxonomy
)
from src.schemas.candidate_schema import (
    CandidateProfile
)
from src.schemas.retrieval_schema import (
    RetrievalResult
)
from src.schemas.ranking_schema import (
    FeatureVector,
    RankedCandidate
)
from src.schemas.explanation_schema import (
    Explanation,
    RecruiterAssessment
)
from src.schemas.evaluation_schema import (
    EvaluationMetrics,
    EvaluationReport
)
from src.schemas.submission_schema import (
    SubmissionRecord
)

# Integration aliases — canonical schema types used by external-module adapters.
from src.schemas.graph_schema import StructuredJD as JobDescription
from src.schemas.candidate_schema import CandidateProfile as Candidate
from src.schemas.retrieval_schema import RetrievalResult as SearchResult

__all__ = [
    "Skill",
    "Education",
    "Experience",
    "Project",
    "Certification",
    "Location",
    "Company",
    "Timestamp",
    "Score",
    "Metadata",
    "UUIDMixin",
    "StructuredJD",
    "SkillTaxonomy",
    "CandidateProfile",
    "RetrievalResult",
    "FeatureVector",
    "RankedCandidate",
    "Explanation",
    "RecruiterAssessment",
    "EvaluationMetrics",
    "EvaluationReport",
    "SubmissionRecord",
    "JobDescription",
    "Candidate",
    "SearchResult",
]