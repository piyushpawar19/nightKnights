from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.schemas.jd_schema import StructuredJD
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.ranking_schema import RankedCandidate
from src.schemas.explanation_schema import RecruiterAssessment
from src.interfaces.explanation_interface import EvidenceExtractorInterface
from src.schemas.common_schema import Skill

class EvidenceExtractor(EvidenceExtractorInterface):
    """Extracts structured evidence from JD, candidate profile, and ranking data."""

    def extract_evidence(
        self,
        structured_jd: StructuredJD,
        candidate_profile: CandidateProfile,
        ranked_candidate: RankedCandidate,
        recruiter_assessment: Optional[RecruiterAssessment] = None
    ) -> Dict[str, Any]:
        """Extracts structured evidence relevant for explanation generation."""
        evidence = {
            "candidate_id": candidate_profile.candidate_id,
            "job_title": structured_jd.job_title,
            "company": structured_jd.company,
            "candidate_name": candidate_profile.name,
            "summary": candidate_profile.summary,
            "ranked_score": ranked_candidate.hybrid_score,
            "rank": ranked_candidate.rank,
            "experience_years": self._calculate_years_experience(candidate_profile.experience),
            "matched_skills": self._get_matched_skills(structured_jd.must_have_skills, candidate_profile.skills),
            "missing_skills": self._get_missing_skills(structured_jd.must_have_skills, candidate_profile.skills),
            "education_relevance": self._get_education_relevance(structured_jd.education, candidate_profile.education),
            "project_relevance": self._get_project_relevance(structured_jd.technologies, candidate_profile.projects),
            "career_progression": self._analyze_career_progression(candidate_profile.experience),
            "leadership_evidence": self._extract_leadership_evidence(candidate_profile.experience),
            "risk_factors": self._identify_risk_factors(candidate_profile, structured_jd),
            "recruiter_assessment_data": recruiter_assessment.model_dump() if recruiter_assessment else None,
            "metadata": {
                "jd_metadata": structured_jd.metadata.model_dump(),
                "candidate_metadata": candidate_profile.metadata.model_dump(),
                "ranking_metadata": ranked_candidate.metadata.model_dump()
            }
        }
        return evidence

    def _calculate_years_experience(self, experiences: List[Any]) -> float:
        """Calculates total years of experience from a list of experience entries."""
        total_years = 0.0
        for exp in experiences:
            if hasattr(exp, 'start_date') and hasattr(exp, 'end_date'):
                start = exp.start_date
                end = exp.end_date if exp.end_date else datetime.now(timezone.utc)
                delta = end - start
                total_years += delta.days / 365.25
        return round(total_years, 2)

    def _get_matched_skills(self, jd_skills: List[Skill], candidate_skills: List[Skill]) -> List[str]:
        """Identifies skills from the JD that match candidate skills."""
        jd_skill_names = {skill.name.lower() for skill in jd_skills}
        candidate_skill_names = {skill.name.lower() for skill in candidate_skills}
        return list(jd_skill_names.intersection(candidate_skill_names))

    def _get_missing_skills(self, jd_skills: List[Skill], candidate_skills: List[Skill]) -> List[str]:
        """Identifies skills from the JD that are missing from candidate skills."""
        jd_skill_names = {skill.name.lower() for skill in jd_skills}
        candidate_skill_names = {skill.name.lower() for skill in candidate_skills}
        return list(jd_skill_names.difference(candidate_skill_names))

    def _get_education_relevance(self, jd_education: List[str], candidate_education: List[Any]) -> List[str]:
        """Determines relevance of candidate education to JD requirements."""
        relevant_education = []
        jd_edu_keywords = {edu.lower() for edu in jd_education}
        for edu in candidate_education:
            if hasattr(edu, 'degree') and any(keyword in edu.degree.lower() for keyword in jd_edu_keywords):
                relevant_education.append(edu.degree)
            elif hasattr(edu, 'field_of_study') and any(keyword in edu.field_of_study.lower() for keyword in jd_edu_keywords):
                relevant_education.append(edu.field_of_study)
        return relevant_education

    def _get_project_relevance(self, jd_technologies: List[Skill], candidate_projects: List[Any]) -> List[str]:
        """Assesses relevance of candidate projects based on JD technologies."""
        relevant_projects = []
        jd_tech_names = {tech.name.lower() for tech in jd_technologies}
        for project in candidate_projects:
            if hasattr(project, 'technologies'):
                project_tech_names = {tech.name.lower() for tech in project.technologies}
                if jd_tech_names.intersection(project_tech_names):
                    relevant_projects.append(project.name)
        return relevant_projects

    def _analyze_career_progression(self, experiences: List[Any]) -> str:
        """Analyzes career progression based on experience titles and durations."""
        if not experiences:
            return "No work experience provided."

        sorted_experiences = sorted(experiences, key=lambda x: x.start_date if hasattr(x, 'start_date') else datetime.min)
        progression_points = []

        for i, exp in enumerate(sorted_experiences):
            if i > 0:
                prev_exp = sorted_experiences[i-1]
                # Simple check for title change indicating progression
                if hasattr(exp, 'title') and hasattr(prev_exp, 'title') and exp.title != prev_exp.title:
                    progression_points.append(f"Moved from {prev_exp.title} to {exp.title} at {exp.company}.")

        if progression_points:
            return "; ".join(progression_points)
        return "No clear career progression indicated by title changes."

    def _extract_leadership_evidence(self, experiences: List[Any]) -> List[str]:
        """Extracts evidence of leadership from experience descriptions and titles."""
        leadership_keywords = ["lead", "manager", "head", "director", "architect", "team lead", "mentor", "managed", "coached"]
        evidence = []
        for exp in experiences:
            if hasattr(exp, 'title') and any(kw in exp.title.lower() for kw in leadership_keywords):
                evidence.append(f"Leadership role identified in title: {exp.title} at {exp.company}.")
            if hasattr(exp, 'description') and exp.description and any(kw in exp.description.lower() for kw in leadership_keywords):
                evidence.append(f"Leadership activities mentioned in description for {exp.title} at {exp.company}.")
        return evidence

    def _identify_risk_factors(self, candidate_profile: CandidateProfile, structured_jd: StructuredJD) -> List[str]:
        """Identifies potential risk factors based on candidate profile and JD."""
        risks = []

        # Example: Gaps in employment
        if candidate_profile.experience:
            aware_max = datetime.max.replace(tzinfo=timezone.utc)

            def _sort_key(exp: Any) -> datetime:
                if exp.end_date is None:
                    return aware_max
                if exp.end_date.tzinfo is None:
                    return exp.end_date.replace(tzinfo=timezone.utc)
                return exp.end_date

            sorted_experiences = sorted(candidate_profile.experience, key=_sort_key, reverse=True)
            for i in range(len(sorted_experiences) - 1):
                current_end = sorted_experiences[i].end_date or datetime.now(timezone.utc)
                if current_end.tzinfo is None:
                    current_end = current_end.replace(tzinfo=timezone.utc)
                next_start = sorted_experiences[i + 1].start_date
                if next_start and next_start.tzinfo is None:
                    next_start = next_start.replace(tzinfo=timezone.utc)
                    risks.append(f"Potential employment gap identified between {sorted_experiences[i+1].company} and {sorted_experiences[i].company}.")

        # Example: Mismatch in required experience years
        if structured_jd.experience_required is not None:
            total_experience = self._calculate_years_experience(candidate_profile.experience)
            if total_experience < structured_jd.experience_required:
                risks.append(f"Candidate has {total_experience} years of experience, but JD requires {structured_jd.experience_required} years.")

        # Example: Too many job changes in short period (e.g., more than 3 jobs in 5 years)
        if candidate_profile.experience and len(candidate_profile.experience) > 3:
            # Filter for experiences within the last 5 years relative to the latest experience end date
            latest_end_date = max([exp.end_date for exp in candidate_profile.experience if exp.end_date is not None] + [datetime.now(timezone.utc)])
            recent_experiences = [exp for exp in candidate_profile.experience if (latest_end_date.year - exp.start_date.year) <= 5]
            if len(recent_experiences) > 3:
                risks.append("High number of job changes detected within the last 5 years.")

        # Example: Low activity score (if provided)
        if candidate_profile.activity_score is not None and candidate_profile.activity_score < 0.3:
            risks.append(f"Candidate activity score is low ({candidate_profile.activity_score}), indicating potential disengagement.")

        # Example: Missing critical must-have skills
        missing = self._get_missing_skills(structured_jd.must_have_skills, candidate_profile.skills)
        if missing:
            risks.append(f"Candidate is missing crucial must-have skills: {'; '.join(missing)}.")

        return risks