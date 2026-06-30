
import json
import logging
from typing import Dict, Any
from pydantic import ValidationError
from ..interfaces.reranking_interface import ResponseParserInterface
from ..schemas.reranking_schema import RecruiterAssessment

logger = logging.getLogger(__name__)

class RecruiterParser(ResponseParserInterface):
    """Parses and validates the LLM's recruiter assessment response."""

    def parse_recruiter_assessment(self, response: str) -> RecruiterAssessment:
        """Parses JSON, validates schema, and handles malformed responses.

        Args:
            response (str): The raw JSON response from the LLM.

        Returns:
            RecruiterAssessment: A validated Pydantic model of the assessment.

        Raises:
            ValueError: If the response is not valid JSON or fails schema validation.
        """
        try:
            # Attempt to parse the JSON response
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Malformed JSON response from LLM: {e}\nResponse: {response}")
            # Attempt basic recovery for common issues like extra text before/after JSON
            try:
                start_idx = response.find("{")
                end_idx = response.rfind("}")
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    cleaned_response = response[start_idx : end_idx + 1]
                    data = json.loads(cleaned_response)
                else:
                    raise ValueError("Could not find valid JSON object in response.")
            except (json.JSONDecodeError, ValueError) as recovery_e:
                raise ValueError(f"Failed to parse and recover JSON response: {recovery_e}") from e

        try:
            # Validate the parsed data against the Pydantic schema
            assessment = RecruiterAssessment(**data)
            self._validate_assessment_fields(assessment)
            return assessment
        except ValidationError as e:
            logger.error(f"Schema validation failed for recruiter assessment: {e}\nData: {data}")
            raise ValueError(f"Invalid recruiter assessment schema: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during assessment parsing: {e}\nData: {data}")
            raise ValueError(f"Unexpected error during assessment parsing: {e}") from e

    def _validate_assessment_fields(self, assessment: RecruiterAssessment):
        """Validates confidence and recruiter score ranges."""
        if not (0.0 <= assessment.confidence <= 1.0):
            raise ValueError(f"Confidence score out of range (0.0-1.0): {assessment.confidence}")
        if not (0.0 <= assessment.recruiter_score <= 1.0):
            raise ValueError(f"Recruiter score out of range (0.0-1.0): {assessment.recruiter_score}")
