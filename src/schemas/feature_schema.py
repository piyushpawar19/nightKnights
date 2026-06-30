
import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class RawFeatureMetrics(BaseModel):
    matched_skills: List[str] = Field(default_factory=list, description="List of skills matched between candidate and job description.")
    missing_skills: List[str] = Field(default_factory=list, description="List of required skills from JD not found in candidate profile.")
    required_skill_matches: int = Field(..., description="Number of required skills from JD matched in candidate profile.")
    preferred_skill_matches: int = Field(..., description="Number of preferred skills from JD matched in candidate profile.")
    candidate_experience_years: float = Field(..., description="Candidate\'s total years of experience.")
    required_experience_years: float = Field(..., description="Required years of experience from job description.")
    experience_gap_years: float = Field(..., description="Difference between candidate\'s and required experience.")
    matched_certifications: List[str] = Field(default_factory=list, description="List of certifications matched.")
    matched_languages: List[str] = Field(default_factory=list, description="List of programming languages matched.")
    matched_frameworks: List[str] = Field(default_factory=list, description="List of frameworks matched.")
    matched_databases: List[str] = Field(default_factory=list, description="List of databases matched.")
    matched_cloud_platforms: List[str] = Field(default_factory=list, description="List of cloud platforms matched.")
    matched_devops_tools: List[str] = Field(default_factory=list, description="List of DevOps tools matched.")
    matched_ai_ml_skills: List[str] = Field(default_factory=list, description="List of AI/ML skills matched.")
    matched_soft_skills: List[str] = Field(default_factory=list, description="List of soft skills matched.")
    keyword_matches: List[str] = Field(default_factory=list, description="List of keywords matched between JD and candidate.")

class NormalizedFeatureMetrics(BaseModel):
    skill_overlap: float = Field(..., ge=0, le=1, description="Normalized skill overlap score [0,1].")
    required_skill_coverage: float = Field(..., ge=0, le=1, description="Normalized required skill coverage [0,1].")
    preferred_skill_coverage: float = Field(..., ge=0, le=1, description="Normalized preferred skill coverage [0,1].")
    experience_match: float = Field(..., ge=0, le=1, description="Normalized experience match score [0,1].")
    education_match: float = Field(..., ge=0, le=1, description="Normalized education match score [0,1].")
    certification_match: float = Field(..., ge=0, le=1, description="Normalized certification match score [0,1].")
    title_similarity: float = Field(..., ge=0, le=1, description="Normalized title similarity score [0,1].")
    seniority_match: float = Field(..., ge=0, le=1, description="Normalized seniority match score [0,1].")
    domain_match: float = Field(..., ge=0, le=1, description="Normalized domain match score [0,1].")
    industry_match: float = Field(..., ge=0, le=1, description="Normalized industry match score [0,1].")
    location_match: float = Field(..., ge=0, le=1, description="Normalized location match score [0,1].")
    employment_type_match: float = Field(..., ge=0, le=1, description="Normalized employment type match score [0,1].")
    language_match: float = Field(..., ge=0, le=1, description="Normalized language match score [0,1].")
    framework_match: float = Field(..., ge=0, le=1, description="Normalized framework match score [0,1].")
    database_match: float = Field(..., ge=0, le=1, description="Normalized database match score [0,1].")
    cloud_match: float = Field(..., ge=0, le=1, description="Normalized cloud platform match score [0,1].")
    devops_match: float = Field(..., ge=0, le=1, description="Normalized DevOps tools match score [0,1].")
    ai_ml_match: float = Field(..., ge=0, le=1, description="Normalized AI/ML skills match score [0,1].")
    soft_skill_match: float = Field(..., ge=0, le=1, description="Normalized soft skills match score [0,1].")
    keyword_similarity: float = Field(..., ge=0, le=1, description="Normalized keyword similarity score [0,1].")
    technology_stack_match: float = Field(..., ge=0, le=1, description="Normalized technology stack match score [0,1].")

class FeatureMetadata(BaseModel):
    feature_count: int = Field(..., description="Total number of features generated.")
    generation_timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, description="Timestamp of feature generation.")
    schema_version: str = Field("1.0.0", description="Version of the feature schema.")

class CandidateFeatures(BaseModel):
    raw: RawFeatureMetrics = Field(..., description="Raw, unnormalized feature metrics.")
    normalized: NormalizedFeatureMetrics = Field(..., description="Normalized feature metrics [0,1].")
    metadata: FeatureMetadata = Field(..., description="Metadata about the feature generation process.")

class FeatureEngineeringRequest(BaseModel):
    parsed_jd: dict = Field(..., description="Parsed Job Description.")
    extracted_skills: dict = Field(..., description="Extracted skills from JD and candidate.")
    candidate_profile: dict = Field(..., description="Candidate\'s professional profile.")

class FeatureEngineeringResponse(BaseModel):
    candidate_features: CandidateFeatures = Field(..., description="Engineered features for the candidate.")
