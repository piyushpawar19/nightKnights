from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat, conint
from src.schemas.common_schema import Metadata, UUIDMixin
from src.schemas.retrieval_schema import RetrievalResult


class FeatureVector(UUIDMixin):
    """Represents engineered recruiter features for a candidate."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    years_experience: Optional[conint(ge=0)] = Field(None, description="Years of professional experience.")
    leadership_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score indicating leadership potential or experience.")
    retrieval_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score derived from retrieval relevance.")
    llm_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score from LLM-based assessment.")
    startup_experience: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score indicating experience in startup environments.")
    education_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score based on educational background.")
    project_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score based on projects and portfolio.")
    activity_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score based on candidate activity or engagement.")
    risk_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score indicating potential hiring risks.")
    normalized_vector: Optional[List[confloat(ge=0.0, le=1.0)]] = Field(None, description="A normalized vector representation of features.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the feature vector.")


class RankedCandidate(UUIDMixin):
    """Represents the output of a hybrid ranking step for a single candidate."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    hybrid_score: confloat(ge=0.0, le=1.0) = Field(..., description="Overall hybrid ranking score.")
    ranking_breakdown: Optional[Dict[str, confloat(ge=0.0, le=1.0)]] = Field(None, description="Detailed breakdown of scores from different ranking components.")
    rank: conint(ge=1) = Field(..., description="Final rank of the candidate.")
    retrieval_result: RetrievalResult = Field(..., description="The retrieval result associated with this ranked candidate.")
    feature_vector: Optional[FeatureVector] = Field(None, description="The feature vector used for ranking.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the ranked candidate.")