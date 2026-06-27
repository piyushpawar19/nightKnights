from typing import Any, Dict, List

from src.interfaces.export_interface import DataValidator
from src.models.domain_models import Explanation, RankedCandidate, RecruiterAssessment


class ExportValidationError(ValueError):
    """Custom exception for export validation errors."""

    pass


class SubmissionValidator(DataValidator):
    """Validates export data for the hackathon submission format."""

    def validate(self, data: List[Dict[str, Any]], schema: List[str]) -> None:
        """Validates the given data against a specified schema and business rules.

        Args:
            data (List[Dict[str, Any]]): The list of dictionaries to validate.
            schema (List[str]): The expected schema (list of field names).

        Raises:
            ExportValidationError: If validation fails.
        """
        if not data:
            raise ExportValidationError("Export data cannot be empty.")

        self._validate_schema(data, schema)
        self._validate_duplicate_candidate_ids(data)
        self._validate_ranks_and_scores(data)
        self._validate_null_values(data)
        self._validate_empty_explanations(data)

    def _validate_schema(self, data: List[Dict[str, Any]], schema: List[str]) -> None:
        """Ensures all data rows conform to the export schema."""
        for i, row in enumerate(data):
            if set(row.keys()) != set(schema):
                missing_fields = set(schema) - set(row.keys())
                extra_fields = set(row.keys()) - set(schema)
                error_msg = f"Schema mismatch in row {i+1}."
                if missing_fields:
                    error_msg += f" Missing fields: {', '.join(missing_fields)}."
                if extra_fields:
                    error_msg += f" Extra fields: {', '.join(extra_fields)}."
                raise ExportValidationError(error_msg)

    def _validate_duplicate_candidate_ids(self, data: List[Dict[str, Any]]) -> None:
        """Checks for duplicate candidate IDs in the export data."""
        candidate_ids = [row.get("candidate_id") for row in data]
        if len(candidate_ids) != len(set(candidate_ids)):
            duplicates = set([x for x in candidate_ids if candidate_ids.count(x) > 1])
            raise ExportValidationError(f"Duplicate candidate IDs found: {', '.join(map(str, duplicates))}")

    def _validate_ranks_and_scores(self, data: List[Dict[str, Any]]) -> None:
        """Validates that ranks are sequential and scores are within a valid range (0.0 to 1.0)."""
        ranks = sorted([row.get("rank") for row in data])
        expected_ranks = list(range(1, len(data) + 1))
        if ranks != expected_ranks:
            raise ExportValidationError("Invalid or non-sequential ranks found.")

        for i, row in enumerate(data):
            score = row.get("hybrid_score")
            if not (0.0 <= score <= 1.0):
                raise ExportValidationError(f"Invalid hybrid_score in row {i+1}: {score}. Must be between 0.0 and 1.0.")

    def _validate_null_values(self, data: List[Dict[str, Any]]) -> None:
        """Checks for any null values in required fields."""
        for i, row in enumerate(data):
            for key, value in row.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    raise ExportValidationError(f"Null or empty value found in row {i+1}, field \"{key}\".")

    def _validate_empty_explanations(self, data: List[Dict[str, Any]]) -> None:
        """Checks if explanation fields are not empty."""
        for i, row in enumerate(data):
            if "explanation_summary" in row and not row["explanation_summary"].strip():
                raise ExportValidationError(f"Empty explanation summary found in row {i+1}.")
            if "explanation_reasoning" in row and not row["explanation_reasoning"].strip():
                raise ExportValidationError(f"Empty explanation reasoning found in row {i+1}.")