from typing import Dict, Any, List
from string import Template
from datetime import datetime, timezone
import json

from src.schemas.explanation_schema import Explanation, RecruiterAssessment
from src.interfaces.explanation_interface import FallbackGeneratorInterface
import logging

from src.utils.logger import get_logger


class FallbackGenerator(FallbackGeneratorInterface):
    """Generates deterministic fallback explanations using templates."""

    def __init__(self, fallback_template: str, logger: logging.Logger):
        self.fallback_template = fallback_template # Raw string content of the fallback template
        self.logger = logger

    def generate_fallback_explanation(
        self,
        candidate_id: str,
        reason: str = "LLM explanation failed or timed out."
    ) -> Explanation:
        """Generates a deterministic fallback explanation.

        Args:
            candidate_id: The ID of the candidate for whom the explanation is generated.
            reason: The reason for generating a fallback explanation.

        Returns:
            An Explanation object with fallback details.
        """
        self.logger.warning(f"Generating fallback explanation for candidate {candidate_id} due to: {reason}")

        try:
            # Use string.Template to substitute basic variables
            template = Template(self.fallback_template)
            substituted_content = template.substitute(candidate_id=candidate_id, reason=reason)

            # Attempt to parse as YAML/JSON to extract structured data
            # For simplicity, assuming a structure that can be easily parsed or directly mapped
            # A more robust solution might use a YAML parser here if the template is complex YAML
            # For now, let's assume a simple key: value structure that can be coerced to JSON
            
            # Convert simple key-value pairs to a dict. This is a naive approach.
            # A better way would be to use a proper YAML parser like PyYAML if the template is complex YAML.
            data = {}
            lines = substituted_content.split("\n")
            current_key = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if ":" in line and not line.startswith("-"):
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ["strengths", "weaknesses"]:
                        data[key] = [] # Initialize as list
                        current_key = key
                    elif key == "recruiter_assessment":
                        data[key] = {}
                        current_key = key
                    elif current_key == "recruiter_assessment": # Nested fields
                        data[current_key][key] = self._try_convert_to_type(value)
                    else:
                        data[key] = self._try_convert_to_type(value)
                        current_key = None # Reset for non-list/dict types
                elif line.startswith("-") and current_key and isinstance(data.get(current_key), list):
                    data[current_key].append(line[1:].strip())

            # Ensure recruiter_assessment is a valid RecruiterAssessment instance
            recruiter_assessment_data = data.pop("recruiter_assessment", {})
            recruiter_assessment_data.setdefault("candidate_id", candidate_id)
            recruiter_assessment = RecruiterAssessment(**recruiter_assessment_data)

            # Construct the Explanation object
            explanation = Explanation(
                candidate_id=data.get("candidate_id", candidate_id),
                summary=data.get("summary", reason),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                reasoning=data.get("reasoning", reason),
                recommendation=data.get("recommendation", "No Hire"),
                confidence=data.get("confidence", 0.2),
                recruiter_assessment=recruiter_assessment
            )
            return explanation
        except Exception as e:
            self.logger.error(f"Failed to generate structured fallback explanation for {candidate_id}: {e}")
            # As a last resort, return a minimal Explanation object to prevent pipeline failure
            return Explanation(
                candidate_id=candidate_id,
                summary="Critical error during fallback explanation generation.",
                strengths=[],
                weaknesses=[f"Failed to generate a proper explanation: {e}"],
                reasoning=f"An unrecoverable error occurred during the fallback process. Original reason: {reason}.",
                recommendation="Error",
                confidence=0.0,
                recruiter_assessment=RecruiterAssessment(
                    candidate_id=candidate_id,
                    technical_score=0.0, career_score=0.0, behavior_score=0.0,
                    risk_score=1.0, culture_fit=0.0, hiring_confidence=0.0, final_score=0.0
                )
            )

    def _try_convert_to_type(self, value: str) -> Any:
        """Tries to convert a string value to a more specific type (int, float, bool)."""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                if value.lower() == "true":
                    return True
                if value.lower() == "false":
                    return False
                return value
