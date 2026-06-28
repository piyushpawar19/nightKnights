from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat
from src.schemas.common_schema import Skill, Education, Experience, Project, Certification, Location, Metadata, UUIDMixin


class CandidateProfile(UUIDMixin):
    """Canonical representation of one candidate."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    name: constr(min_length=1, max_length=150) = Field(..., description="Full name of the candidate.")
    summary: Optional[constr(max_length=2000)] = Field(None, description="A brief summary or personal statement.")
    experience: List[Experience] = Field(default_factory=list, description="List of work experiences.")
    education: List[Education] = Field(default_factory=list, description="List of educational backgrounds.")
    projects: List[Project] = Field(default_factory=list, description="List of personal or professional projects.")
    skills: List[Skill] = Field(default_factory=list, description="List of skills possessed by the candidate.")
    certifications: List[Certification] = Field(default_factory=list, description="List of professional certifications.")
    location: Optional[Location] = Field(None, description="Candidate\'s geographical location.")
    activity_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="An arbitrary score representing candidate activity or engagement.")
    resume_text: Optional[constr(max_length=10000)] = Field(None, description="Raw text extracted from the candidate\'s resume.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the candidate profile.")