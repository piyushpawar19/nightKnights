from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Protocol

from src.models.domain_models import Explanation, RankedCandidate, RecruiterAssessment


class Exportable(Protocol):
    """Protocol for objects that can be exported."""

    def to_export_dict(self) -> Dict[str, Any]:
        """Converts the object to a dictionary format suitable for export."""
        ...


class DataValidator(ABC):
    """Abstract base class for data validation."""

    @abstractmethod
    def validate(self, data: List[Dict[str, Any]], schema: List[str]) -> None:
        """Validates the given data against a specified schema.

        Args:
            data (List[Dict[str, Any]]): The list of dictionaries to validate.
            schema (List[str]): The expected schema (list of field names).

        Raises:
            ValueError: If validation fails.
        """
        pass


class Serializer(ABC):
    """Abstract base class for data serialization."""

    @abstractmethod
    def serialize(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        export_schema: List[str],
    ) -> List[Dict[str, Any]]:
        """Serializes domain models into a list of dictionaries for export."""
        pass


class CSVWriter(ABC):
    """Abstract base class for writing data to CSV format."""

    @abstractmethod
    def write_csv(
        self,
        filepath: Path,
        data: List[Dict[str, Any]],
        schema: List[str],
        delimiter: str = ",",
        quotechar: str = ",",
        quoting: int = 0, # csv.QUOTE_MINIMAL
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> None:
        """Writes a list of dictionaries to a CSV file."""
        pass


class ExportService(ABC):
    """Abstract base class for the export service."""

    @abstractmethod
    def export_csv(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        output_dir: Path,
        filename: str,
        export_schema: List[str],
        delimiter: str = ",",
        quotechar: str = ",",
        quoting: int = 0, # csv.QUOTE_MINIMAL
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> Path:
        """Exports validated ranking results to a CSV file."""
        pass

    @abstractmethod
    def export_json(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        output_dir: Path,
        filename: str,
        overwrite: bool = False,
    ) -> Path:
        """Exports validated ranking results to a JSON file."""
        pass

    @abstractmethod
    def export_parquet(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        output_dir: Path,
        filename: str,
        overwrite: bool = False,
    ) -> Path:
        """Exports validated ranking results to a Parquet file."""
        pass