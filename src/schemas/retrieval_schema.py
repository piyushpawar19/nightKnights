from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat, conint
from src.schemas.common_schema import Metadata, UUIDMixin


class RetrievalResult(UUIDMixin):
    """Represents the output of a retrieval step for a single candidate."""
    candidate_id: constr(min_length=1) = Field(..., description="Unique identifier for the candidate.")
    dense_score: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Score from dense retrieval models.")
    bm25_score: Optional[confloat(ge=0.0)] = Field(None, description="Score from BM25 retrieval.")
    retrieval_source: constr(min_length=1, max_length=100) = Field(..., description="Source of the retrieval (e.g., \'vector_db\', \'keyword_search\').")
    rank: conint(ge=1) = Field(..., description="Rank of the candidate in the retrieval results.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the retrieval result.")