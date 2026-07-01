from typing import List, Dict, Any
from src.preprocessing.feature_utils import (
    calculate_overlap, calculate_percentage, normalize_score, safe_get, safe_float, cosine_similarity
)
from functools import lru_cache
import os
from joblib import Memory

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

@memory.cache
def extract_matched_skills(jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> List[str]:
    """Identifies and returns skills present in both JD and candidate profiles."""
    jd_skills_unique = set(jd_skills)
    candidate_skills_lower = {s.lower() for s in candidate_skills}

    # Direct matches
    matched_direct = {s for s in jd_skills_unique if s.lower() in candidate_skills_lower}

    # Conceptual matching for \"Machine Learning\" in candidate skills
    conceptual_matches = set()
    if "machine learning" in {s.lower() for s in jd_skills_unique}:
        if any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
            conceptual_matches.add("Machine Learning")

    matched = list(matched_direct | conceptual_matches)
    return sorted(matched)

@memory.cache
def extract_missing_skills(required_jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> List[str]:
    """Identifies and returns required JD skills missing from candidate profile."""
    missing = []
    candidate_skills_lower = {s.lower() for s in candidate_skills}
    for skill in required_jd_skills:
        if skill.lower() == "machine learning":
            if not any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
                missing.append(skill)
        elif skill.lower() not in candidate_skills_lower:
            missing.append(skill)
    return sorted(missing)

@memory.cache
def extract_required_skill_matches(required_jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> int:
    """Counts the number of required JD skills matched in candidate profile."""
    matches = 0
    candidate_skills_lower = {s.lower() for s in candidate_skills}
    for skill in required_jd_skills:
        if skill.lower() == "machine learning":
            if any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
                matches += 1
        elif skill.lower() in candidate_skills_lower:
            matches += 1
    return matches

@memory.cache
def extract_preferred_skill_matches(preferred_jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> int:
    """Counts the number of preferred JD skills matched in candidate profile."""
    return len(set(preferred_jd_skills) & set(candidate_skills))

@memory.cache
def extract_candidate_experience_years(candidate_profile: Dict) -> float:
    """Extracts candidate\"s total years of experience."""
    return safe_float(safe_get(candidate_profile, ["experience", "total_years"], 0.0))

@memory.cache
def extract_required_experience_years(parsed_jd: Dict) -> float:
    """Extracts required years of experience from JD."""
    return safe_float(safe_get(parsed_jd, ["experience_requirements", "min_years"], 0.0))

@memory.cache
def extract_experience_gap_years(candidate_exp: float, required_exp: float) -> float:
    """Calculates the experience gap in years."""
    return max(0.0, required_exp - candidate_exp)

@memory.cache
def extract_matched_certifications(jd_certs: tuple[str, ...], candidate_certs: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched certifications."""
    return list(set(jd_certs) & set(candidate_certs))

@memory.cache
def extract_matched_languages(jd_languages: tuple[str, ...], candidate_languages: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched programming languages."""
    return list(set(jd_languages) & set(candidate_languages))

@memory.cache
def extract_matched_frameworks(jd_frameworks: tuple[str, ...], candidate_frameworks: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched frameworks."""
    return list(set(jd_frameworks) & set(candidate_frameworks))

@memory.cache
def extract_matched_databases(jd_databases: tuple[str, ...], candidate_databases: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched databases."""
    return list(set(jd_databases) & set(candidate_databases))

@memory.cache
def extract_matched_cloud_platforms(jd_cloud: tuple[str, ...], candidate_cloud: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched cloud platforms."""
    return list(set(jd_cloud) & set(candidate_cloud))

@memory.cache
def extract_matched_devops_tools(jd_devops: tuple[str, ...], candidate_devops: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched DevOps tools."""
    return list(set(jd_devops) & set(candidate_devops))

@memory.cache
def extract_matched_ai_ml_skills(jd_aiml: tuple[str, ...], candidate_aiml: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched AI/ML skills."""
    return list(set(jd_aiml) & set(candidate_aiml))

@memory.cache
def extract_matched_soft_skills(jd_soft_skills: tuple[str, ...], candidate_soft_skills: tuple[str, ...]) -> List[str]:
    """Identifies and returns matched soft skills."""
    return list(set(jd_soft_skills) & set(candidate_soft_skills))

@memory.cache
def extract_keyword_matches(jd_text: str, candidate_text: str, keywords: tuple[str, ...]) -> List[str]:
    """Identifies and returns keywords found in both JD and candidate texts."""
    matched = []
    jd_lower = jd_text.lower()
    candidate_lower = candidate_text.lower()
    for keyword in keywords:
        # Directly check if the keyword is present in both texts
        if keyword.lower() in jd_lower and keyword.lower() in candidate_lower:
            matched.append(keyword)
    return sorted(matched)

# --- Normalized Feature Extractors ---

@memory.cache
def normalize_skill_overlap(jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> float:
    """Normalizes skill overlap between JD and candidate."""
    return calculate_overlap(list(jd_skills), list(candidate_skills))

@memory.cache
def normalize_required_skill_coverage(required_jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> float:
    """Normalizes coverage of required skills."""
    if not required_jd_skills:
        return 0.0
    matched = 0
    candidate_skills_lower = {s.lower() for s in candidate_skills}
    for skill in required_jd_skills:
        if skill.lower() == "machine learning":
            if any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
                matched += 1
        elif skill.lower() in candidate_skills_lower:
            matched += 1
    return calculate_percentage(matched, len(required_jd_skills))

@memory.cache
def normalize_preferred_skill_coverage(preferred_jd_skills: tuple[str, ...], candidate_skills: tuple[str, ...]) -> float:
    """Normalizes coverage of preferred skills."""
    if not preferred_jd_skills:
        return 0.0
    matched = len(set(preferred_jd_skills).intersection(candidate_skills))
    return calculate_percentage(matched, len(preferred_jd_skills))

@memory.cache
def normalize_experience_match(candidate_exp: float, required_exp: float, max_deviation: float = 5.0) -> float:
    """Normalizes experience match score."""
    if required_exp == 0:
        return 1.0 if candidate_exp >= 0 else 0.0 # If no experience is required, any experience is a match
    
    # If candidate experience meets or exceeds required experience, it\"s a perfect match (1.0)
    if candidate_exp >= required_exp:
        return 1.0
    
    # If candidate experience is less than required, calculate a score based on the gap
    # The score decreases as the gap increases, up to max_deviation
    gap = required_exp - candidate_exp
    if gap >= max_deviation:
        return 0.0
    
    # Linear interpolation between 0.0 (at max_deviation) and 1.0 (at required_exp)
    return normalize_score(candidate_exp, required_exp, required_exp - max_deviation)

@memory.cache
def normalize_education_match(jd_education: tuple[str, ...], candidate_education: tuple[str, ...]) -> float:
    """Normalizes education match score."""
    # A simple approach: check if candidate has any of the required/preferred JD education levels
    if not jd_education or not candidate_education:
        return 0.0
    
    jd_education_lower = {edu.lower() for edu in jd_education}
    candidate_education_lower = {edu.lower() for edu in candidate_education}
    
    for edu in candidate_education_lower:
        if edu in jd_education_lower:
            return 1.0
    return 0.0

@memory.cache
def normalize_certification_match(jd_certs: tuple[str, ...], candidate_certs: tuple[str, ...]) -> float:
    """Normalizes certification match score."""
    return calculate_overlap(list(jd_certs), list(candidate_certs))

@memory.cache
def normalize_title_similarity(jd_title: str, candidate_title: str) -> float:
    """Normalizes job title similarity."""
    return cosine_similarity(jd_title, candidate_title)

@memory.cache
def normalize_seniority_match(jd_seniority: str, candidate_seniority: str) -> float:
    """Normalizes seniority level match."""
    if not jd_seniority or not candidate_seniority:
        return 0.0
    return 1.0 if jd_seniority.lower() == candidate_seniority.lower() else 0.0

@memory.cache
def normalize_domain_match(jd_domain: tuple[str, ...], candidate_domains: tuple[str, ...]) -> float:
    """Normalizes domain match score."""
    return calculate_overlap(list(jd_domain), list(candidate_domains))

@memory.cache
def normalize_industry_match(jd_industry: tuple[str, ...], candidate_industries: tuple[str, ...]) -> float:
    """Normalizes industry match score."""
    return calculate_overlap(list(jd_industry), list(candidate_industries))

@memory.cache
def normalize_location_match(jd_location: tuple[str, ...], candidate_location: tuple[str, ...]) -> float:
    """Normalizes location match score."""
    # Assuming a match if any location in candidate list matches any in JD list
    if not jd_location or not candidate_location:
        return 0.0
    
    jd_loc_lower = {loc.lower() for loc in jd_location}
    candidate_loc_lower = {loc.lower() for loc in candidate_location}
    
    for loc in candidate_loc_lower:
        if loc in jd_loc_lower:
            return 1.0
    return 0.0

@memory.cache
def normalize_employment_type_match(jd_emp_type: tuple[str, ...], candidate_emp_type: tuple[str, ...]) -> float:
    """Normalizes employment type match score."""
    # Assuming a match if any employment type in candidate list matches any in JD list
    if not jd_emp_type or not candidate_emp_type:
        return 0.0
    
    jd_emp_lower = {et.lower() for et in jd_emp_type}
    candidate_emp_lower = {et.lower() for et in candidate_emp_type}
    
    for et in candidate_emp_lower:
        if et in jd_emp_lower:
            return 1.0
    return 0.0

@memory.cache
def normalize_language_match(jd_languages: tuple[str, ...], candidate_languages: tuple[str, ...]) -> float:
    """Normalizes programming language match score."""
    return calculate_overlap(list(jd_languages), list(candidate_languages))

@memory.cache
def normalize_framework_match(jd_frameworks: tuple[str, ...], candidate_frameworks: tuple[str, ...]) -> float:
    """Normalizes framework match score."""
    return calculate_overlap(list(jd_frameworks), list(candidate_frameworks))

@memory.cache
def normalize_database_match(jd_databases: tuple[str, ...], candidate_databases: tuple[str, ...]) -> float:
    """Normalizes database match score."""
    return calculate_overlap(list(jd_databases), list(candidate_databases))

@memory.cache
def normalize_cloud_match(jd_cloud: tuple[str, ...], candidate_cloud: tuple[str, ...]) -> float:
    """Normalizes cloud platform match score."""
    return calculate_overlap(list(jd_cloud), list(candidate_cloud))

@memory.cache
def normalize_devops_match(jd_devops: tuple[str, ...], candidate_devops: tuple[str, ...]) -> float:
    """Normalizes DevOps tools match score."""
    return calculate_overlap(list(jd_devops), list(candidate_devops))

@memory.cache
def normalize_ai_ml_match(jd_aiml: tuple[str, ...], candidate_aiml: tuple[str, ...]) -> float:
    """Normalizes AI/ML skills match score."""
    return calculate_overlap(list(jd_aiml), list(candidate_aiml))

@memory.cache
def normalize_soft_skill_match(jd_soft_skills: tuple[str, ...], candidate_soft_skills: tuple[str, ...]) -> float:
    """Normalizes soft skills match score."""
    return calculate_overlap(list(jd_soft_skills), list(candidate_soft_skills))

@memory.cache
def normalize_keyword_similarity(jd_text: str, candidate_text: str) -> float:
    """Normalizes keyword similarity between JD and candidate texts."""
    return cosine_similarity(jd_text, candidate_text)

@memory.cache
def normalize_technology_stack_match(jd_tech_stack: tuple[str, ...], candidate_tech_stack: tuple[str, ...]) -> float:
    """Normalizes technology stack match score."""
    return calculate_overlap(list(jd_tech_stack), list(candidate_tech_stack))
