from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr, conint, confloat
from src.schemas.common_schema import Skill, Metadata


class SkillTaxonomy(BaseModel):
    """Represents normalized skills with aliases, categories, and confidence scores."""
    normalized_skills: List[constr(min_length=1, max_length=100)] = Field(..., description="List of normalized skill names.")
    aliases: Dict[constr(min_length=1), List[constr(min_length=1)]] = Field(default_factory=dict, description="Mapping of normalized skills to their known aliases.")
    categories: Dict[constr(min_length=1), List[constr(min_length=1)]] = Field(default_factory=dict, description="Mapping of normalized skills to their categories.")
    confidence_scores: Dict[constr(min_length=1), confloat(ge=0.0, le=1.0)] = Field(default_factory=dict, description="Confidence scores for each normalized skill.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the skill taxonomy.")


class StructuredJD(BaseModel):
    """Represents information extracted from a Job Description (JD)."""
    job_title: constr(min_length=1, max_length=200) = Field(..., description="Title of the job.")
    company: constr(min_length=1, max_length=150) = Field(..., description="Hiring company name.")
    industry: Optional[constr(max_length=100)] = Field(None, description="Industry of the job.")
    seniority: Optional[constr(max_length=50)] = Field(None, description="Seniority level (e.g., Junior, Senior, Lead).")
    experience_required: Optional[conint(ge=0)] = Field(None, description="Years of experience required.")
    education: List[constr(min_length=1, max_length=200)] = Field(default_factory=list, description="Required educational qualifications.")
    employment_type: Optional[constr(max_length=50)] = Field(None, description="Employment type (e.g., Full-time, Contract).")
    location: Optional[constr(max_length=100)] = Field(None, description="Job location.")
    must_have_skills: List[Skill] = Field(default_factory=list, description="List of essential skills.")
    nice_to_have_skills: List[Skill] = Field(default_factory=list, description="List of desirable skills.")
    behavioral_traits: List[constr(min_length=1, max_length=100)] = Field(default_factory=list, description="Required behavioral traits.")
    responsibilities: List[constr(min_length=1, max_length=500)] = Field(default_factory=list, description="Key job responsibilities.")
    technologies: List[Skill] = Field(default_factory=list, description="Specific technologies associated with the job.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the structured JD.")