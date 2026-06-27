from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat
from src.schemas.common_schema import Metadata, UUIDMixin


class RecruiterAssessment(BaseModel):
    """Represents a recruiter's assessment based on reranking output."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    technical_score: confloat(ge=0.0, le=1.0) = Field(..., description="Score for technical skills.")
    career_score: confloat(ge=0.0, le=1.0) = Field(..., description="Score for career trajectory and growth.")
    behavior_score: confloat(ge=0.0, le=1.0) = Field(..., description="Score for behavioral traits and cultural fit.")
    risk_score: confloat(ge=0.0, le=1.0) = Field(..., description="Score indicating potential risks (e.g., flight risk, red flags).")
    culture_fit: confloat(ge=0.0, le=1.0) = Field(..., description="Score for cultural alignment with the company.")
    hiring_confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Recruiter\"s confidence level in hiring the candidate.")
    final_score: confloat(ge=0.0, le=1.0) = Field(..., description="Overall final score from the recruiter\"s assessment.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the recruiter assessment.")


class Explanation(UUIDMixin):
    """Represents a recruiter's explanation for a candidate's ranking."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    summary: constr(min_length=1, max_length=1000) = Field(..., description="Brief summary of the explanation.")
    strengths: List[constr(min_length=1, max_length=500)] = Field(default_factory=list, description="Key strengths of the candidate relative to the job.")
    weaknesses: List[constr(min_length=1, max_length=500)] = Field(default_factory=list, description="Key weaknesses or areas for concern.")
    reasoning: constr(min_length=1, max_length=2000) = Field(..., description="Detailed reasoning behind the assessment and recommendation.")
    recommendation: constr(min_length=1, max_length=500) = Field(..., description="Hiring recommendation (e.g., \"Strong Hire\", \"Consider\", \"No Hire\").")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence level of the explanation.")
    recruiter_assessment: Optional[RecruiterAssessment] = Field(None, description="The detailed recruiter assessment if available.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the explanation.")