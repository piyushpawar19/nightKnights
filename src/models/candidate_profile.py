from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class CandidateProfile(BaseModel):
    """
    Normalized and retrieval-optimized representation of a candidate profile.
    Designed for BM25 search, dense embedding generation, and feature engineering.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        str_strip_whitespace=True
    )
    
    candidate_id: str = Field(..., description="Unique identifier for the candidate (CAND_XXXXXXX)")
    full_name: str = Field(default="", description="Normalized full name of the candidate")
    headline: str = Field(default="", description="One-line professional headline")
    summary: str = Field(default="", description="Multi-sentence professional summary")
    skills: List[str] = Field(default_factory=list, description="Normalized list of skill names")
    experience: List[str] = Field(default_factory=list, description="Formated list of career history events")
    education: List[str] = Field(default_factory=list, description="Formatted list of educational details")
    projects: List[str] = Field(default_factory=list, description="List of personal or work projects")
    certifications: List[str] = Field(default_factory=list, description="Formatted list of certifications")
    location: str = Field(default="", description="City, country/region")
    current_company: str = Field(default="", description="Name of current employer")
    current_role: str = Field(default="", description="Current job title")
    years_experience: float = Field(default=0.0, description="Total years of work experience")
    github: Optional[str] = Field(default=None, description="GitHub username or link if connected")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile link if connected")
    portfolio: Optional[str] = Field(default=None, description="Personal portfolio website URL")
    activity: str = Field(default="", description="Summary of platform engagement activity")
    search_text: str = Field(default="", description="Consolidated document text optimized for dense and sparse retrieval")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata metrics, profile completeness details, etc.")
