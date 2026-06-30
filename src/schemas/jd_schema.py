from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class JobInfo(BaseModel):
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Hiring company")
    location: Optional[str] = Field(None, description="Job location")
    employment_type: Optional[str] = Field(None, description="Full-time, part-time, contract, etc.")
    remote_type: Optional[str] = Field(None, description="On-site, remote, hybrid")
    seniority: Optional[str] = Field(None, description="Junior, mid, senior, lead, principal, architect")
    industry: Optional[str] = Field(None, description="Industry of the company")
    domain: Optional[str] = Field(None, description="Specific domain within the industry")
    salary: Optional[Dict[str, Any]] = Field(None, description="Parsed salary information")
    minimum_experience: Optional[int] = Field(None, description="Minimum years of experience required")
    maximum_experience: Optional[int] = Field(None, description="Maximum years of experience required")

class Education(BaseModel):
    degree: Optional[str] = Field(None, description="Degree required (e.g., BE, MTech, PhD)")
    field: Optional[str] = Field(None, description="Field of study")
    required: bool = Field(False, description="Is this education mandatory?")

class Certification(BaseModel):
    name: str = Field(..., description="Name of the certification")
    required: bool = Field(False, description="Is this certification mandatory?")

class Requirements(BaseModel):
    mandatory_requirements: List[str] = Field([], description="List of mandatory requirements")
    preferred_requirements: List[str] = Field([], description="List of preferred requirements")
    certifications: List[Certification] = Field([], description="List of required/preferred certifications")
    education: List[Education] = Field([], description="List of required education")

class Skills(BaseModel):
    technical_skills: List[str] = Field([], description="General technical skills")
    programming_languages: List[str] = Field([], description="Programming languages required")
    frameworks: List[str] = Field([], description="Frameworks required")
    libraries: List[str] = Field([], description="Libraries required")
    databases: List[str] = Field([], description="Databases required")
    cloud: List[str] = Field([], description="Cloud platforms/services")
    devops: List[str] = Field([], description="DevOps tools/practices")
    ai_ml: List[str] = Field([], description="AI/ML specific skills")
    soft_skills: List[str] = Field([], description="Soft skills required")

class Responsibilities(BaseModel):
    responsibilities_list: List[str] = Field([], description="List of job responsibilities")

class Preferences(BaseModel):
    # This can be expanded based on common JD preferences
    pass

class ParsingMetadata(BaseModel):
    parse_timestamp: str = Field(..., description="Timestamp of when the JD was parsed")
    parser_version: str = Field("1.0", description="Version of the parser used")
    # Add other metadata as needed, e.g., confidence scores, detected sections

class ParsedJD(BaseModel):
    job_info: JobInfo = Field(..., description="General job information")
    requirements: Requirements = Field(..., description="Job requirements")
    skills: Skills = Field(..., description="Skills required for the job")
    responsibilities: Responsibilities = Field(..., description="Key job responsibilities")
    preferences: Preferences = Field(..., description="Job preferences or nice-to-haves")
    metadata: ParsingMetadata = Field(..., description="Metadata about the parsing process")
