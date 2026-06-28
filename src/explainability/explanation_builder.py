from typing import Any, Dict, List, Optional

from src.interfaces.explanation_interface import ExplanationBuilderInterface
from src.schemas.explanation_schema import Explanation, RecruiterAssessment


class ExplanationBuilder(ExplanationBuilderInterface):
    """Builds a structured Explanation object from extracted evidence."""

    def build_explanation(
        self,
        candidate_id: str,
        evidence: Dict[str, Any],
        recruiter_assessment: Optional[RecruiterAssessment] = None,
    ) -> Explanation:
        summary = self._generate_summary(evidence)
        strengths = self._identify_strengths(evidence)
        weaknesses = self._identify_weaknesses(evidence)
        reasoning = self._collate_reasoning(evidence, strengths, weaknesses)
        recommendation = self._determine_recommendation(evidence)
        confidence = self._calculate_confidence(evidence, recruiter_assessment)

        return Explanation(
            candidate_id=candidate_id,
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            reasoning=reasoning,
            recommendation=recommendation,
            confidence=confidence,
            recruiter_assessment=recruiter_assessment,
        )

    def _generate_summary(self, evidence: Dict[str, Any]) -> str:
        job_title = evidence.get("job_title", "N/A")
        candidate_name = evidence.get("candidate_name", "Candidate")
        ranked_score = evidence.get("ranked_score", 0.0)

        if ranked_score >= 0.75:
            return (
                f"{candidate_name} is a strong candidate for the {job_title} role, "
                "demonstrating excellent alignment with key requirements."
            )
        if ranked_score >= 0.5:
            return (
                f"{candidate_name} is a good candidate for the {job_title} role, "
                "with several relevant skills and experiences."
            )
        return (
            f"{candidate_name} has some relevant background for the {job_title} role, "
            "but may have gaps in critical areas."
        )

    def _identify_strengths(self, evidence: Dict[str, Any]) -> List[str]:
        strengths: List[str] = []
        structured_jd = evidence.get("structured_jd", {})
        experience_required = structured_jd.get("experience_required", 0)
        experience_years = evidence.get("experience_years", 0)

        if evidence.get("matched_skills"):
            strengths.append(
                f"Strong alignment with must-have skills: {'; '.join(evidence['matched_skills'])}."
            )
        if experience_years >= experience_required:
            strengths.append(
                f"Exceeds or meets the required {experience_required} years of experience "
                f"with {experience_years} years."
            )
        elif experience_years > 0:
            strengths.append(f"Possesses {experience_years} years of relevant experience.")
        if evidence.get("education_relevance"):
            strengths.append(
                f"Relevant educational background in: {'; '.join(evidence['education_relevance'])}."
            )
        if evidence.get("project_relevance"):
            strengths.append(
                f"Demonstrates practical experience through projects like: "
                f"{'; '.join(evidence['project_relevance'])}."
            )
        if evidence.get("leadership_evidence"):
            strengths.append(
                f"Evidence of leadership roles/activities: {'; '.join(evidence['leadership_evidence'])}."
            )
        if evidence.get("career_progression") and "No clear" not in evidence["career_progression"]:
            strengths.append(f"Shows positive career progression: {evidence['career_progression']}")

        return strengths

    def _identify_weaknesses(self, evidence: Dict[str, Any]) -> List[str]:
        weaknesses: List[str] = []
        structured_jd = evidence.get("structured_jd", {})
        experience_required = structured_jd.get("experience_required", 0)
        experience_years = evidence.get("experience_years", 0)

        if evidence.get("missing_skills"):
            weaknesses.append(
                f"Missing critical must-have skills: {'; '.join(evidence['missing_skills'])}."
            )
        if evidence.get("risk_factors"):
            weaknesses.append(
                f"Identified potential risk factors: {'; '.join(evidence['risk_factors'])}."
            )
        if experience_years < experience_required and experience_required > 0:
            weaknesses.append(
                f"Candidate has fewer years of experience ({experience_years} years) "
                f"than the required {experience_required}."
            )

        return weaknesses

    def _collate_reasoning(
        self, evidence: Dict[str, Any], strengths: List[str], weaknesses: List[str]
    ) -> str:
        reasoning_parts = [
            (
                f"The candidate {evidence.get('candidate_name', 'Candidate')} was assessed for the "
                f"{evidence.get('job_title', 'N/A')} role at {evidence.get('company', 'N/A')}."
            )
        ]

        if strengths:
            reasoning_parts.append("Strengths include: " + ". ".join(strengths) + ".")
        if weaknesses:
            reasoning_parts.append("Areas for development or concern are: " + ". ".join(weaknesses) + ".")

        if evidence.get("summary"):
            reasoning_parts.append(f"Candidate summary indicates: {evidence['summary']}.")
        if evidence.get("ranked_score") is not None:
            reasoning_parts.append(
                f"The candidate achieved a ranking score of {evidence['ranked_score']:.2f} "
                f"(Rank: {evidence.get('rank', 'N/A')})."
            )

        return " ".join(reasoning_parts)

    def _determine_recommendation(self, evidence: Dict[str, Any]) -> str:
        ranked_score = evidence.get("ranked_score", 0.0)
        risk_factors = evidence.get("risk_factors", [])

        if ranked_score >= 0.8 and not risk_factors:
            return "Strong Hire"
        if ranked_score >= 0.6 and not risk_factors:
            return "Consider for Interview"
        if ranked_score >= 0.4 and len(risk_factors) < 2:
            return "Possible Fit, requires further review"
        return "No Hire"

    def _calculate_confidence(
        self, evidence: Dict[str, Any], recruiter_assessment: Optional[RecruiterAssessment]
    ) -> float:
        confidence = 0.5

        if evidence.get("ranked_score") is not None:
            confidence += evidence["ranked_score"] * 0.2

        if recruiter_assessment:
            confidence += recruiter_assessment.hiring_confidence * 0.3

        if not evidence.get("missing_skills") and not evidence.get("risk_factors"):
            confidence += 0.1

        return min(1.0, confidence)
