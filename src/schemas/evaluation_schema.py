from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, constr, confloat
from src.schemas.common_schema import Metadata, UUIDMixin


class EvaluationMetrics(UUIDMixin):
    """Represents a collection of evaluation metrics for the AI hiring platform."""
    metric_name: constr(min_length=1, max_length=100) = Field(..., description="Name of the evaluation metric (e.g., \"Recall@K\", \"Precision@K\").")
    value: confloat(ge=0.0) = Field(..., description="The value of the metric.")
    threshold: Optional[confloat(ge=0.0)] = Field(None, description="Optional threshold for the metric.")
    unit: Optional[constr(max_length=50)] = Field(None, description="Unit of the metric (e.g., \"percentage\", \"seconds\").")
    description: Optional[constr(max_length=500)] = Field(None, description="Description of the metric and its calculation.")
    group: Optional[constr(max_length=100)] = Field(None, description="Grouping for the metric (e.g., \"ranking\", \"retrieval\").")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional data related to the metric.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the evaluation metrics.")


class EvaluationReport(UUIDMixin):
    """Represents a comprehensive evaluation report containing various metrics."""
    report_name: constr(min_length=1, max_length=200) = Field(..., description="Name of the evaluation report.")
    start_time: str = Field(..., description="Start time of the evaluation.")
    end_time: str = Field(..., description="End time of the evaluation.")
    metrics: List[EvaluationMetrics] = Field(default_factory=list, description="List of individual evaluation metrics.")
    configuration_used: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters used for the evaluation.")
    summary: Optional[constr(max_length=2000)] = Field(None, description="Overall summary of the evaluation report.")
    metadata: Metadata = Field(default_factory=Metadata, description="Additional metadata for the evaluation report.")