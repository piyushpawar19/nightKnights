
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal, Optional

class RecruiterAssessmentMetadata(BaseModel):
    assessment_timestamp: datetime = Field(default_factory=datetime.now)
    schema_version: str = "1.0"

class RecruiterAssessment(BaseModel):
    strengths: List[str] = Field(..., description="List of strengths of the candidate based on the job description.")
    concerns: List[str] = Field(..., description="List of concerns about the candidate based on the job description.")
    hiring_recommendation: Literal["Strong Yes", "Yes", "No", "Strong No"] = Field(..., description="Hiring recommendation for the candidate.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the assessment (0.0 to 1.0).")
    recruiter_score: float = Field(..., ge=0.0, le=1.0, description="Recruiter-assigned score for the candidate (0.0 to 1.0).")
    reasoning: str = Field(..., description="Detailed reasoning behind the assessment and score.")
    metadata: RecruiterAssessmentMetadata = Field(default_factory=RecruiterAssessmentMetadata)

class RerankedCandidateMetadata(BaseModel):
    assessment_timestamp: datetime = Field(default_factory=datetime.now)
    schema_version: str = "1.0"

class RerankedCandidate(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate.")
    previous_rank: int = Field(..., description="Candidate's rank before recruiter reranking.")
    new_rank: int = Field(..., description="Candidate's new rank after recruiter reranking.")
    previous_score: float = Field(..., description="Candidate's score from the hybrid ranker.")
    recruiter_score: float = Field(..., ge=0.0, le=1.0, description="Normalized recruiter score for the candidate.")
    final_score: float = Field(..., description="Combined final score after reranking.")
    assessment: RecruiterAssessment = Field(..., description="Detailed recruiter assessment for the candidate.")
    metadata: RerankedCandidateMetadata = Field(default_factory=RerankedCandidateMetadata)

class RerankedCandidates(BaseModel):
    candidates: List[RerankedCandidate] = Field(..., description="List of reranked candidates.")
