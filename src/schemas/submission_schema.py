from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat, conint
from src.schemas.common_schema import Metadata, UUIDMixin


class SubmissionRecord(UUIDMixin):
    """Represents one row of the final submission, detailing a ranked candidate."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    rank: conint(ge=1) = Field(..., description="The final rank of the candidate in the submission.")
    score: confloat(ge=0.0, le=1.0) = Field(..., description="The final score of the candidate.")
    explanation_reference: Optional[constr(max_length=200)] = Field(None, description="Reference to a detailed explanation for the candidate, e.g., a file path or URL.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the submission record.")