from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field, conint, confloat, constr, model_validator


class Metadata(BaseModel):
    """Represents generic metadata as a key-value store."""
    data: Dict[str, Any] = Field(default_factory=dict, description="Dictionary for arbitrary metadata.")


class Skill(BaseModel):
    """Represents a skill with its name and an optional proficiency level."""
    name: constr(min_length=1, max_length=100) = Field(..., description="Name of the skill.")
    proficiency: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Proficiency level, if applicable (0.0 to 1.0).")


class Education(BaseModel):
    """Represents an educational background."""
    institution: constr(min_length=1, max_length=200) = Field(..., description="Name of the educational institution.")
    degree: constr(min_length=1, max_length=150) = Field(..., description="Degree obtained (e.g., 'B.S. in Computer Science').")
    field_of_study: Optional[constr(max_length=150)] = Field(None, description="Field of study.")
    start_date: Optional[datetime] = Field(None, description="Start date of education.")
    end_date: Optional[datetime] = Field(None, description="End date of education.")
    description: Optional[constr(max_length=1000)] = Field(None, description="Description of academic achievements or coursework.")


class Experience(BaseModel):
    """Represents a work experience entry."""
    title: constr(min_length=1, max_length=150) = Field(..., description="Job title.")
    company: constr(min_length=1, max_length=150) = Field(..., description="Company name.")
    start_date: datetime = Field(..., description="Start date of employment.")
    end_date: Optional[datetime] = Field(None, description="End date of employment. None if currently employed.")
    description: Optional[constr(max_length=2000)] = Field(None, description="Description of responsibilities and achievements.")
    technologies: List[Skill] = Field(default_factory=list, description="List of technologies used.")

    @model_validator(mode='after')
    def validate_dates(self):
        if self.end_date and self.start_date > self.end_date:
            raise ValueError("End date cannot be before start date.")
        return self


class Project(BaseModel):
    """Represents a project undertaken by a candidate."""
    name: constr(min_length=1, max_length=200) = Field(..., description="Name of the project.")
    description: Optional[constr(max_length=2000)] = Field(None, description="Description of the project.")
    technologies: List[Skill] = Field(default_factory=list, description="Technologies used in the project.")
    url: Optional[str] = Field(None, description="URL to the project (e.g., GitHub, portfolio).")


class Certification(BaseModel):
    """Represents a professional certification."""
    name: constr(min_length=1, max_length=200) = Field(..., description="Name of the certification.")
    issuing_organization: constr(min_length=1, max_length=150) = Field(..., description="Organization that issued the certification.")
    issue_date: Optional[datetime] = Field(None, description="Date the certification was issued.")
    expiration_date: Optional[datetime] = Field(None, description="Date the certification expires.")

    @model_validator(mode='after')
    def validate_dates(self):
        if self.issue_date and self.expiration_date and self.issue_date > self.expiration_date:
            raise ValueError("Issue date cannot be after expiration date.")
        return self


class Location(BaseModel):
    """Represents a geographical location."""
    city: Optional[constr(max_length=100)] = Field(None, description="City.")
    state: Optional[constr(max_length=100)] = Field(None, description="State or province.")
    country: constr(min_length=1, max_length=100) = Field(..., description="Country.")
    zip_code: Optional[constr(max_length=20)] = Field(None, description="Zip code.")


class Company(BaseModel):
    """Represents a company with its name and an optional industry."""
    name: constr(min_length=1, max_length=150) = Field(..., description="Name of the company.")
    industry: Optional[constr(max_length=100)] = Field(None, description="Industry of the company.")


class Timestamp(BaseModel):
    """Represents a timestamp with creation and last updated dates."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of creation.")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of last update.")

    @model_validator(mode='after')
    def set_update_timestamp(self):
        self.updated_at = datetime.now(timezone.utc)
        return self


class Score(BaseModel):
    """Represents a generic scoring mechanism."""
    value: confloat(ge=0.0, le=1.0) = Field(..., description="The score value, between 0.0 and 1.0.")
    metric: constr(min_length=1, max_length=100) = Field(..., description="The metric used for scoring (e.g., 'relevance', 'confidence').")
    breakdown: Optional[Dict[str, float]] = Field(None, description="Detailed breakdown of the score components.")


class UUIDMixin(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the entity.")

    @model_validator(mode='before')
    @classmethod
    def validate_id(cls, values):
        if not isinstance(values, dict):
            return values
        if 'id' in values and values['id'] is not None:
            try:
                uuid.UUID(str(values['id']))
            except ValueError:
                raise ValueError("Invalid UUID format.")
        return values