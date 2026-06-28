import re
import logging
from typing import Any, Dict, Iterable, List, Optional
from src.models.candidate_profile import CandidateProfile

logger = logging.getLogger(__name__)

# Compile whitespace pattern for reuse to optimize performance
WHITESPACE_PATTERN = re.compile(r"\s+")


class ProfileBuilderError(Exception):
    """Base exception for Candidate Profile Builder Agent errors."""
    pass


class InvalidCandidateRecordError(ProfileBuilderError):
    """Exception raised when raw candidate record is malformed or missing key fields."""
    pass


class CandidateProfileBuilder:
    """
    Stateless agent responsible for cleaning, normalizing, and formatting raw candidate dictionaries
    into validated, structured, and retrieval-optimized CandidateProfile objects.
    """

    def build_profile(self, raw_candidate: dict) -> CandidateProfile:
        """
        Transforms a single raw candidate record into a validated CandidateProfile.

        Args:
            raw_candidate: Dictionary containing the raw candidate profile payload.

        Returns:
            A validated CandidateProfile Pydantic model instance.

        Raises:
            InvalidCandidateRecordError: If the raw data is invalid or missing critical fields.
        """
        if not isinstance(raw_candidate, dict):
            raise InvalidCandidateRecordError("Raw candidate record must be a dictionary.")

        candidate_id = raw_candidate.get("candidate_id")
        if not candidate_id:
            raise InvalidCandidateRecordError("Candidate record is missing the required 'candidate_id' field.")

        try:
            # 1. Clean basic fields
            full_name = self._clean_text(self._safe_get(raw_candidate, "profile.anonymized_name", ""))
            headline = self._clean_text(self._safe_get(raw_candidate, "profile.headline", ""))
            summary = self._clean_text(self._safe_get(raw_candidate, "profile.summary", ""))

            # Format location & country
            loc = self._clean_text(self._safe_get(raw_candidate, "profile.location", ""))
            country = self._clean_text(self._safe_get(raw_candidate, "profile.country", ""))
            location = f"{loc}, {country}" if loc and country else (loc or country)

            # 2. Extract current company and role
            current_company = self._extract_current_company(raw_candidate)
            current_role = self._extract_current_role(raw_candidate)

            # 3. Extract years of experience
            years_experience = self._extract_years_experience(raw_candidate)

            # 4. Extract links & socials if present (default to None if missing)
            github = self._clean_text(self._safe_get(raw_candidate, "profile.github")) or raw_candidate.get("github")
            linkedin = self._clean_text(self._safe_get(raw_candidate, "profile.linkedin")) or raw_candidate.get("linkedin")
            portfolio = self._clean_text(self._safe_get(raw_candidate, "profile.portfolio")) or raw_candidate.get("portfolio")

            # 5. Extract projects (handle optional field gracefully)
            projects_raw = self._safe_get(raw_candidate, "projects", [])
            projects = self._normalize_list(
                projects_raw, 
                formatter=lambda p: p.get("name", "") if isinstance(p, dict) else str(p)
            )

            # 6. Normalize list fields
            skills_raw = self._safe_get(raw_candidate, "skills", [])
            skills = self._normalize_list(
                skills_raw, 
                formatter=lambda s: s.get("name", "") if isinstance(s, dict) else str(s)
            )

            career_raw = self._safe_get(raw_candidate, "career_history", [])
            experience = self._normalize_list(career_raw, formatter=self._format_job)

            education_raw = self._safe_get(raw_candidate, "education", [])
            education = self._normalize_list(education_raw, formatter=self._format_education)

            certifications_raw = self._safe_get(raw_candidate, "certifications", [])
            certifications = self._normalize_list(certifications_raw, formatter=self._format_cert)

            # 7. Formulate platform activity status
            activity = self._build_activity_summary(raw_candidate)

            # 8. Generate consolidated search optimized text block
            search_text = self._build_search_text(
                headline=headline,
                summary=summary,
                skills=skills,
                experience=experience,
                projects=projects,
                education=education,
                certifications=certifications,
                current_role=current_role,
                current_company=current_company,
                activity=activity
            )

            # 9. Compute metadata
            metadata = self._build_metadata(raw_candidate, skills, projects, certifications)

            # 10. Instantiate and validate via Pydantic
            profile = CandidateProfile(
                candidate_id=candidate_id,
                full_name=full_name,
                headline=headline,
                summary=summary,
                skills=skills,
                experience=experience,
                education=education,
                projects=projects,
                certifications=certifications,
                location=location,
                current_company=current_company,
                current_role=current_role,
                years_experience=years_experience,
                github=github or None,
                linkedin=linkedin or None,
                portfolio=portfolio or None,
                activity=activity,
                search_text=search_text,
                metadata=metadata
            )

            logger.debug(f"Successfully processed profile for candidate {candidate_id}")
            return profile

        except Exception as e:
            logger.error(f"Error parsing candidate profile {candidate_id}: {str(e)}", exc_info=True)
            raise InvalidCandidateRecordError(f"Malformed schema in candidate {candidate_id}: {str(e)}") from e

    def build_profiles(self, candidates: Iterable[dict]) -> List[CandidateProfile]:
        """
        Transforms a batch of raw candidate records, skipping invalid records gracefully.

        Args:
            candidates: Iterable of raw candidate dictionaries.

        Returns:
            List of successfully built CandidateProfile objects.
        """
        processed_profiles: List[CandidateProfile] = []
        success_count = 0
        skipped_count = 0

        for raw_candidate in candidates:
            try:
                profile = self.build_profile(raw_candidate)
                processed_profiles.append(profile)
                success_count += 1
                if success_count % 1000 == 0:
                    logger.info(f"CandidateProfileBuilder: Processed {success_count} candidate records.")
            except Exception as e:
                skipped_count += 1
                cid = raw_candidate.get("candidate_id", "Unknown") if isinstance(raw_candidate, dict) else "Non-dict"
                logger.warning(f"Failed parsing record {cid}. Reason: {str(e)}")

        logger.info(
            f"CandidateProfileBuilder complete. "
            f"Successfully built: {success_count}, Skipped: {skipped_count}"
        )
        return processed_profiles

    # --- Private Helper Methods ---

    def _safe_get(self, data: Any, keys: str | List[str], default: Any = None) -> Any:
        """Helper to navigate nested dictionary structures using dot notation or keys list."""
        if data is None:
            return default
        if isinstance(keys, str):
            keys = keys.split(".")
        
        curr = data
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            elif isinstance(curr, list):
                try:
                    curr = curr[int(k)]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return curr

    def _clean_text(self, text: Any) -> str:
        """Removes duplicate whitespaces, trims trailing space, and standardizes strings."""
        if not text or not isinstance(text, str):
            return ""
        return WHITESPACE_PATTERN.sub(" ", text).strip()

    def _normalize_list(self, items: Any, formatter: Any = None) -> List[str]:
        """Normalizes a raw list of strings or dictionaries into clean formatted text elements."""
        if not items or not isinstance(items, list):
            return []
        
        normalized: List[str] = []
        for item in items:
            if formatter:
                try:
                    formatted = formatter(item)
                    cleaned = self._clean_text(formatted)
                    if cleaned:
                        normalized.append(cleaned)
                except Exception:
                    continue
            else:
                cleaned = self._clean_text(str(item))
                if cleaned:
                    normalized.append(cleaned)
        return normalized

    def _extract_current_company(self, raw_candidate: dict) -> str:
        """Extracts candidate's current employer name from profile fields or career history."""
        # 1. Look in profile block
        company = self._safe_get(raw_candidate, "profile.current_company")
        if company:
            return self._clean_text(company)

        # 2. Backup check in career history entries
        career = self._safe_get(raw_candidate, "career_history", [])
        if isinstance(career, list):
            for job in career:
                if isinstance(job, dict) and job.get("is_current"):
                    comp = job.get("company")
                    if comp:
                        return self._clean_text(comp)
        return ""

    def _extract_current_role(self, raw_candidate: dict) -> str:
        """Extracts candidate's current job title/role from profile fields or career history."""
        # 1. Look in profile block
        title = self._safe_get(raw_candidate, "profile.current_title")
        if not title:
            title = self._safe_get(raw_candidate, "profile.current_role")
        if title:
            return self._clean_text(title)

        # 2. Backup check in career history entries
        career = self._safe_get(raw_candidate, "career_history", [])
        if isinstance(career, list):
            for job in career:
                if isinstance(job, dict) and job.get("is_current"):
                    role = job.get("title")
                    if role:
                        return self._clean_text(role)
        return ""

    def _extract_years_experience(self, raw_candidate: dict) -> float:
        """Extracts years of experience from profile or sums duration in career history."""
        yoe = self._safe_get(raw_candidate, "profile.years_of_experience")
        if yoe is None:
            yoe = self._safe_get(raw_candidate, "profile.years_experience")
        if yoe is not None:
            try:
                return round(float(yoe), 1)
            except (ValueError, TypeError):
                pass

        # Calculate from career history duration as fallback
        career = self._safe_get(raw_candidate, "career_history", [])
        if isinstance(career, list):
            total_months = 0
            for job in career:
                if isinstance(job, dict):
                    dur = job.get("duration_months")
                    if isinstance(dur, (int, float)):
                        total_months += dur
            return round(total_months / 12.0, 1)
        return 0.0

    def _format_job(self, job: dict) -> str:
        """Formats a single job history dict into a structured descriptive sentence."""
        title = job.get("title", "")
        company = job.get("company", "")
        duration = job.get("duration_months", "")
        desc = job.get("description", "")
        
        dur_str = f" ({duration} months)" if duration else ""
        desc_str = f" - {desc}" if desc else ""
        return f"{title} at {company}{dur_str}{desc_str}"

    def _format_education(self, edu: dict) -> str:
        """Formats education info into a structured descriptive string."""
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        start = edu.get("start_year")
        end = edu.get("end_year")
        tier = edu.get("tier", "")
        
        field_str = f" in {field}" if field else ""
        inst_str = f" from {inst}" if inst else ""
        tier_str = f" ({tier.replace('_', ' ').title()})" if tier and tier != "unknown" else ""
        period_str = f" ({start}-{end})" if start and end else ""
        return f"{degree}{field_str}{inst_str}{tier_str}{period_str}"

    def _format_cert(self, cert: dict) -> str:
        """Formats certification details."""
        name = cert.get("name", "")
        issuer = cert.get("issuer", "")
        year = cert.get("year", "")
        issuer_str = f" by {issuer}" if issuer else ""
        year_str = f" ({year})" if year else ""
        return f"{name}{issuer_str}{year_str}"

    def _build_activity_summary(self, raw_candidate: dict) -> str:
        """Formulates platform activity summary based on Redrob engagement metrics."""
        signals = self._safe_get(raw_candidate, "redrob_signals", {})
        if not signals:
            return ""
        
        completeness = signals.get("profile_completeness_score", 0.0)
        last_active = signals.get("last_active_date", "")
        apps = signals.get("applications_submitted_30d", 0)
        connections = signals.get("connection_count", 0)
        open_to_work = signals.get("open_to_work_flag", False)
        
        otw_str = "Open to work." if open_to_work else ""
        active_str = f"Last active on {last_active}." if last_active else ""
        apps_str = f"Submitted {apps} applications in the last 30 days." if apps else ""
        conn_str = f"Has {connections} connections on the platform." if connections else ""
        
        parts = [
            f"Profile completeness score is {completeness}%.",
            otw_str,
            active_str,
            apps_str,
            conn_str
        ]
        return " ".join([p for p in parts if p])

    def _build_search_text(
        self,
        headline: str,
        summary: str,
        skills: List[str],
        experience: List[str],
        projects: List[str],
        education: List[str],
        certifications: List[str],
        current_role: str,
        current_company: str,
        activity: str
    ) -> str:
        """Concatenates candidate details into a dense search string, avoiding duplicate components."""
        parts = []

        items = []
        if headline:
            items.append(headline)
        if summary:
            items.append(summary)
        if current_role or current_company:
            rc = f"Current Role: {current_role} at {current_company}" if current_role and current_company else (current_role or current_company)
            items.append(rc)
        if skills:
            items.append("Skills: " + ", ".join(skills))
        if experience:
            items.append("Experience:\n" + "\n".join(experience))
        if projects:
            items.append("Projects:\n" + "\n".join(projects))
        if education:
            items.append("Education:\n" + "\n".join(education))
        if certifications:
            items.append("Certifications:\n" + "\n".join(certifications))
        if activity:
            items.append("Activity: " + activity)

        seen = set()
        for item in items:
            cleaned = self._clean_text(item)
            if not cleaned:
                continue
            # Store lower-cased version to filter duplicate segments
            lowered = cleaned.lower()
            if lowered not in seen:
                seen.add(lowered)
                parts.append(cleaned)

        return "\n\n".join(parts)

    def _build_metadata(self, raw_candidate: dict, skills: List[str], projects: List[str], certifications: List[str]) -> dict:
        """Builds a metrics and completeness metadata dictionary for profiling/validation."""
        signals = self._safe_get(raw_candidate, "redrob_signals", {})
        completeness_raw = signals.get("profile_completeness_score", 0.0)
        profile_completeness = round(completeness_raw / 100.0, 2)
        
        return {
            "profile_completeness": profile_completeness,
            "num_skills": len(skills),
            "num_projects": len(projects),
            "num_certifications": len(certifications),
            "generated_by": "CandidateProfileBuilder"
        }


if __name__ == "__main__":
    import json
    
    # Simple console validation
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    sample = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Jane Doe",
            "headline": "Senior Machine Learning Engineer | PyTorch | NLP | LLMs",
            "summary": "Experienced ML Engineer with 8.5 years of experience building and deploying NLP models.",
            "location": "San Francisco",
            "country": "USA",
            "years_of_experience": 8.5,
            "current_title": "Senior ML Engineer",
            "current_company": "AI Technologies Inc.",
            "current_company_size": "201-500",
            "current_industry": "Artificial Intelligence"
        },
        "career_history": [
            {
                "company": "AI Technologies Inc.",
                "title": "Senior ML Engineer",
                "start_date": "2022-01-10",
                "end_date": None,
                "duration_months": 53,
                "is_current": True,
                "industry": "Artificial Intelligence",
                "company_size": "201-500",
                "description": "Leading development on LLM fine-tuning pipelines and Retrieval-Augmented Generation (RAG) systems."
            },
            {
                "company": "DataCorp",
                "title": "Data Scientist",
                "start_date": "2018-06-01",
                "end_date": "2021-12-31",
                "duration_months": 42,
                "is_current": False,
                "industry": "Software",
                "company_size": "10001+",
                "description": "Developed and deployed classical machine learning models for user churn prediction."
            }
        ],
        "education": [
            {
                "institution": "Stanford University",
                "degree": "M.S.",
                "field_of_study": "Computer Science",
                "start_year": 2016,
                "end_year": 2018,
                "grade": "3.9 GPA",
                "tier": "tier_1"
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 45, "duration_months": 96},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 32, "duration_months": 60},
            {"name": "Transformers", "proficiency": "advanced", "endorsements": 18, "duration_months": 36}
        ],
        "certifications": [
            {"name": "Deep Learning Specialization", "issuer": "Coursera", "year": 2019}
        ],
        "redrob_signals": {
            "profile_completeness_score": 95.0,
            "signup_date": "2020-01-01",
            "last_active_date": "2026-06-25",
            "open_to_work_flag": True,
            "profile_views_received_30d": 120,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.9,
            "avg_response_time_hours": 1.5,
            "skill_assessment_scores": {"Python": 98.0},
            "connection_count": 450,
            "endorsements_received": 75,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 50.0, "max": 75.0},
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 85.0,
            "search_appearance_30d": 210,
            "saved_by_recruiters_30d": 15,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }

    builder = CandidateProfileBuilder()
    profile = builder.build_profile(sample)
    
    print("\n--- Model Output successfully built ---")
    print(f"ID: {profile.candidate_id}")
    print(f"Full Name: {profile.full_name}")
    print(f"Location: {profile.location}")
    print(f"Experience: {profile.years_experience} years")
    print(f"Metadata: {json.dumps(profile.metadata, indent=2)}")
    print(f"Skills: {profile.skills}")
    print(f"Search Text Length: {len(profile.search_text)} chars")
    print("\n--- Search Text sample: ---")
    print(profile.search_text[:400] + "...")
