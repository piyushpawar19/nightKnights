import logging
import datetime
from typing import Dict, Any, List
from pydantic import ValidationError

from nightKnights.src.schemas.feature_schema import (
    CandidateFeatures, RawFeatureMetrics, NormalizedFeatureMetrics, FeatureMetadata
)
from nightKnights.src.preprocessing.feature_extractors import (
    extract_matched_skills, extract_missing_skills, extract_required_skill_matches,
    extract_preferred_skill_matches, extract_candidate_experience_years,
    extract_required_experience_years, extract_experience_gap_years,
    extract_matched_certifications, extract_matched_languages, extract_matched_frameworks,
    extract_matched_databases, extract_matched_cloud_platforms, extract_matched_devops_tools,
    extract_matched_ai_ml_skills, extract_matched_soft_skills, extract_keyword_matches,
    normalize_skill_overlap, normalize_required_skill_coverage, normalize_preferred_skill_coverage,
    normalize_experience_match, normalize_education_match, normalize_certification_match,
    normalize_title_similarity, normalize_seniority_match, normalize_domain_match,
    normalize_industry_match, normalize_location_match, normalize_employment_type_match,
    normalize_language_match, normalize_framework_match, normalize_database_match,
    normalize_cloud_match, normalize_devops_match, normalize_ai_ml_match, normalize_soft_skill_match,
    normalize_keyword_similarity, normalize_technology_stack_match
)
from nightKnights.src.preprocessing.feature_utils import safe_get, safe_float, calculate_overlap

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def generate_raw_metrics(self, state: Dict) -> Dict:
        parsed_jd = state.get("parsed_jd", {})
        extracted_skills = state.get("extracted_skills", {})
        candidate_profile = state.get("candidate_profile", {})

        # Helper to get lists, defaulting to empty list if key not found
        def get_list(data: Dict, keys: List[str]) -> List[str]:
            return safe_get(data, keys, [])

        # JD Skills
        jd_required_skills = get_list(extracted_skills, ["job_description", "required_skills"])
        jd_preferred_skills = get_list(extracted_skills, ["job_description", "preferred_skills"])
        jd_other_skills = get_list(extracted_skills, ["job_description", "other_skills"])
        jd_all_skills = list(set(jd_required_skills + jd_preferred_skills + jd_other_skills))

        # Candidate Skills
        candidate_hard_skills = get_list(extracted_skills, ["candidate_profile", "hard_skills"])
        candidate_soft_skills_list = get_list(extracted_skills, ["candidate_profile", "soft_skills"])
        candidate_all_skills = list(set(candidate_hard_skills + candidate_soft_skills_list))

        # Experience
        candidate_exp_years = extract_candidate_experience_years(candidate_profile)
        required_exp_years = extract_required_experience_years(parsed_jd)

        # Certifications, Languages, Frameworks, etc. (from parsed_jd and candidate_profile)
        jd_certifications = get_list(parsed_jd, ["certifications"])
        candidate_certifications = get_list(candidate_profile, ["certifications"])

        jd_languages = get_list(parsed_jd, ["technologies", "languages"])
        candidate_languages = get_list(candidate_profile, ["technologies", "languages"])

        jd_frameworks = get_list(parsed_jd, ["technologies", "frameworks"])
        candidate_frameworks = get_list(candidate_profile, ["technologies", "frameworks"])

        jd_databases = get_list(parsed_jd, ["technologies", "databases"])
        candidate_databases = get_list(candidate_profile, ["technologies", "databases"])

        jd_cloud_platforms = get_list(parsed_jd, ["technologies", "cloud_platforms"])
        candidate_cloud_platforms = get_list(candidate_profile, ["technologies", "cloud_platforms"])

        jd_devops_tools = get_list(parsed_jd, ["technologies", "devops_tools"])
        candidate_devops_tools = get_list(candidate_profile, ["technologies", "devops_tools"])

        jd_ai_ml_skills = get_list(parsed_jd, ["technologies", "ai_ml_skills"])
        candidate_ai_ml_skills = get_list(candidate_profile, ["technologies", "ai_ml_skills"])

        jd_soft_skills = get_list(parsed_jd, ["soft_skills"])

        jd_description_text = safe_get(parsed_jd, ["job_description", "full_text"], "")
        candidate_summary_text = safe_get(candidate_profile, ["summary"], "")
        jd_keywords = get_list(parsed_jd, ["keywords"])

        raw_metrics = {
            "matched_skills": extract_matched_skills(jd_all_skills, candidate_all_skills),
            "missing_skills": extract_missing_skills(jd_required_skills, candidate_all_skills),
            "required_skill_matches": extract_required_skill_matches(jd_required_skills, candidate_all_skills),
            "preferred_skill_matches": extract_preferred_skill_matches(jd_preferred_skills, candidate_all_skills),
            "candidate_experience_years": candidate_exp_years,
            "required_experience_years": required_exp_years,
            "experience_gap_years": extract_experience_gap_years(candidate_exp_years, required_exp_years),
            "matched_certifications": extract_matched_certifications(jd_certifications, candidate_certifications),
            "matched_languages": extract_matched_languages(jd_languages, candidate_languages),
            "matched_frameworks": extract_matched_frameworks(jd_frameworks, candidate_frameworks),
            "matched_databases": extract_matched_databases(jd_databases, candidate_databases),
            "matched_cloud_platforms": extract_matched_cloud_platforms(jd_cloud_platforms, candidate_cloud_platforms),
            "matched_devops_tools": extract_matched_devops_tools(jd_devops_tools, candidate_devops_tools),
            "matched_ai_ml_skills": extract_matched_ai_ml_skills(jd_ai_ml_skills, candidate_ai_ml_skills),
            "matched_soft_skills": extract_matched_soft_skills(jd_soft_skills, candidate_soft_skills_list),
            "keyword_matches": extract_keyword_matches(jd_description_text, candidate_summary_text, jd_keywords),
        }
        return raw_metrics

    def generate_normalized_metrics(self, raw_metrics: Dict, state: Dict) -> Dict:
        parsed_jd = state.get("parsed_jd", {})
        extracted_skills = state.get("extracted_skills", {})
        candidate_profile = state.get("candidate_profile", {})

        # Helper to get lists, defaulting to empty list if key not found
        def get_list(data: Dict, keys: List[str]) -> List[str]:
            return safe_get(data, keys, [])

        # JD Skills
        jd_required_skills = get_list(extracted_skills, ["job_description", "required_skills"])
        jd_preferred_skills = get_list(extracted_skills, ["job_description", "preferred_skills"])
        jd_other_skills = get_list(extracted_skills, ["job_description", "other_skills"])
        jd_all_skills = list(set(jd_required_skills + jd_preferred_skills + jd_other_skills))

        # Candidate Skills
        """Changed to include all relevant candidate skills to calculate a more accurate skill_overlap."""
        candidate_hard_skills = get_list(extracted_skills, ["candidate_profile", "hard_skills"])
        candidate_soft_skills_list = get_list(extracted_skills, ["candidate_profile", "soft_skills"])
        candidate_ml_related_skills = get_list(candidate_profile, ["technologies", "ai_ml_skills"])
        # Also consider if TensorFlow/Keras are present in hard_skills as an indicator for ML
        if any(s.lower() in [term.lower() for term in candidate_hard_skills] for s in ["tensorflow", "keras", "pytorch"]):
            candidate_ml_related_skills.append("Machine Learning")
        
        candidate_all_skills = list(set(candidate_hard_skills + candidate_soft_skills_list + candidate_ml_related_skills))

        # Experience
        candidate_exp_years = raw_metrics["candidate_experience_years"]
        required_exp_years = raw_metrics["required_experience_years"]

        # Certifications, Languages, Frameworks, etc.
        jd_certifications = get_list(parsed_jd, ["certifications"])
        candidate_certifications = get_list(candidate_profile, ["certifications"])

        jd_languages = get_list(parsed_jd, ["technologies", "languages"])
        candidate_languages = get_list(candidate_profile, ["technologies", "languages"])

        jd_frameworks = get_list(parsed_jd, ["technologies", "frameworks"])
        candidate_frameworks = get_list(candidate_profile, ["technologies", "frameworks"])

        jd_databases = get_list(parsed_jd, ["technologies", "databases"])
        candidate_databases = get_list(candidate_profile, ["technologies", "databases"])

        jd_cloud_platforms = get_list(parsed_jd, ["technologies", "cloud_platforms"])
        candidate_cloud_platforms = get_list(candidate_profile, ["technologies", "cloud_platforms"])

        jd_devops_tools = get_list(parsed_jd, ["technologies", "devops_tools"])
        candidate_devops_tools = get_list(candidate_profile, ["technologies", "devops_tools"])

        jd_ai_ml_skills = get_list(parsed_jd, ["technologies", "ai_ml_skills"])
        # candidate_ai_ml_skills is defined locally in this method, so pass it here.

        jd_soft_skills = get_list(parsed_jd, ["soft_skills"])

        jd_description_text = safe_get(parsed_jd, ["job_description", "full_text"], "")
        candidate_summary_text = safe_get(candidate_profile, ["summary"], "")

        # For technology_stack_match, combine all tech-related skills/tools
        jd_tech_stack = list(set(
            jd_all_skills + jd_languages + jd_frameworks + jd_databases + 
            jd_cloud_platforms + jd_devops_tools + jd_ai_ml_skills
        ))
        candidate_tech_stack = list(set(
            candidate_all_skills + candidate_languages + candidate_frameworks + 
            candidate_databases + candidate_cloud_platforms + candidate_devops_tools +
            candidate_ml_related_skills # Corrected: use candidate_ml_related_skills instead of undefined candidate_ai_ml_skills
        ))

        normalized_metrics = {
            "skill_overlap": calculate_overlap(list(set(jd_all_skills)), list(set(candidate_all_skills))),
            "required_skill_coverage": normalize_required_skill_coverage(jd_required_skills, candidate_all_skills),
            "preferred_skill_coverage": normalize_preferred_skill_coverage(jd_preferred_skills, candidate_all_skills),
            "experience_match": normalize_experience_match(candidate_exp_years, required_exp_years),
            "education_match": normalize_education_match(
                get_list(parsed_jd, ["education_requirements", "degrees"]),
                get_list(candidate_profile, ["education", "degrees"])
            ),
            "certification_match": normalize_certification_match(jd_certifications, candidate_certifications),
            "title_similarity": normalize_title_similarity(
                safe_get(parsed_jd, ["job_title"], ""),
                safe_get(candidate_profile, ["current_title"], "")
            ),
            "seniority_match": normalize_seniority_match(
                safe_get(parsed_jd, ["seniority_level"], ""),
                safe_get(candidate_profile, ["seniority_level"], "")
            ),
            "domain_match": normalize_domain_match(
                get_list(parsed_jd, ["domain"]),
                get_list(candidate_profile, ["domains"])
            ),
            "industry_match": normalize_industry_match(
                get_list(parsed_jd, ["industry"]),
                get_list(candidate_profile, ["industries"])
            ),
            "location_match": normalize_location_match(
                get_list(parsed_jd, ["location"]),
                get_list(candidate_profile, ["locations"])
            ),
            "employment_type_match": normalize_employment_type_match(
                get_list(parsed_jd, ["employment_type"]),
                get_list(candidate_profile, ["employment_types"])
            ),
            "language_match": normalize_language_match(jd_languages, candidate_languages),
            "framework_match": normalize_framework_match(jd_frameworks, candidate_frameworks),
            "database_match": normalize_database_match(jd_databases, candidate_databases),
            "cloud_match": normalize_cloud_match(jd_cloud_platforms, candidate_cloud_platforms),
            "devops_match": normalize_devops_match(jd_devops_tools, candidate_devops_tools),
            "ai_ml_match": normalize_ai_ml_match(jd_ai_ml_skills, candidate_ml_related_skills), # Pass the correct local variable
            "soft_skill_match": normalize_soft_skill_match(jd_soft_skills, candidate_soft_skills_list),
            "keyword_similarity": normalize_keyword_similarity(jd_description_text, candidate_summary_text),
            "technology_stack_match": normalize_technology_stack_match(jd_tech_stack, candidate_tech_stack),
        }
        return normalized_metrics

    def generate_metadata(self, raw_metrics: Dict) -> Dict:
        return {
            "feature_count": len(raw_metrics),
            "generation_timestamp": datetime.datetime.now(datetime.timezone.utc), # Use timezone-aware UTC datetime
            "schema_version": "1.0.0",
        }

    def construct_candidate_features(self, raw_metrics: Dict, normalized_metrics: Dict, metadata: Dict) -> CandidateFeatures:
        return CandidateFeatures(
            raw=RawFeatureMetrics(**raw_metrics),
            normalized=NormalizedFeatureMetrics(**normalized_metrics),
            metadata=FeatureMetadata(**metadata)
        )
