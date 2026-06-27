from typing import Any, Dict, List

from src.interfaces.export_interface import Serializer
from src.models.domain_models import Explanation, RankedCandidate, RecruiterAssessment


class SubmissionSerializer(Serializer):
    """Serializes domain models into a flat dictionary for hackathon submission CSV."""

    def serialize(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        export_schema: List[str],
    ) -> List[Dict[str, Any]]:
        """Serializes the provided domain models into a list of dictionaries.

        Args:
            ranked_candidates (List[RankedCandidate]): List of ranked candidate models.
            recruiter_assessments (List[RecruiterAssessment]): List of recruiter assessment models.
            explanations (List[Explanation]): List of explanation models.
            export_schema (List[str]): The desired schema for the export rows.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a row for export.
        """
        serialized_data = []

        assessment_map = {ra.candidate_id: ra for ra in recruiter_assessments}
        explanation_map = {exp.candidate_id: exp for exp in explanations}

        for candidate in ranked_candidates:
            row: Dict[str, Any] = {"candidate_id": candidate.candidate_id}

            # Map ranked candidate fields
            if "rank" in export_schema:
                row["rank"] = candidate.rank
            if "hybrid_score" in export_schema:
                row["hybrid_score"] = candidate.hybrid_score

            # Map recruiter assessment fields
            assessment = assessment_map.get(candidate.candidate_id)
            if assessment:
                if "recruiter_technical_score" in export_schema:
                    row["recruiter_technical_score"] = assessment.technical_score
                if "recruiter_career_score" in export_schema:
                    row["recruiter_career_score"] = assessment.career_score
                if "recruiter_behavior_score" in export_schema:
                    row["recruiter_behavior_score"] = assessment.behavior_score
                if "recruiter_risk_score" in export_schema:
                    row["recruiter_risk_score"] = assessment.risk_score
                if "recruiter_culture_fit" in export_schema:
                    row["recruiter_culture_fit"] = assessment.culture_fit
                if "recruiter_hiring_confidence" in export_schema:
                    row["recruiter_hiring_confidence"] = assessment.hiring_confidence
                if "recruiter_final_score" in export_schema:
                    row["recruiter_final_score"] = assessment.final_score

            # Map explanation fields
            explanation = explanation_map.get(candidate.candidate_id)
            if explanation:
                if "explanation_summary" in export_schema:
                    row["explanation_summary"] = explanation.summary
                if "explanation_strengths" in export_schema:
                    row["explanation_strengths"] = "; ".join(explanation.strengths) # Join list into string
                if "explanation_weaknesses" in export_schema:
                    row["explanation_weaknesses"] = "; ".join(explanation.weaknesses) # Join list into string
                if "explanation_reasoning" in export_schema:
                    row["explanation_reasoning"] = explanation.reasoning
                if "explanation_recommendation" in export_schema:
                    row["explanation_recommendation"] = explanation.recommendation
                if "explanation_confidence" in export_schema:
                    row["explanation_confidence"] = explanation.confidence

            # Ensure all schema fields are present, even if empty
            for field in export_schema:
                if field not in row:
                    row[field] = None  # Or empty string, depending on desired output for missing data

            serialized_data.append(row)

        # Sort by rank to ensure deterministic output
        serialized_data.sort(key=lambda x: x.get("rank", float("inf")))

        return serialized_data