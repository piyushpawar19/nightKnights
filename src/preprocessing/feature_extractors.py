from typing import List, Dict, Any
from nightKnights.src.preprocessing.feature_utils import (
    calculate_overlap, calculate_percentage, normalize_score, safe_get, safe_float, cosine_similarity
)

def extract_matched_skills(jd_skills: List[str], candidate_skills: List[str]) -> List[str]:
    """Identifies and returns skills present in both JD and candidate profiles."""
    # Ensure unique skills for accurate matching
    jd_skills_unique = list(set(jd_skills))
    candidate_skills_unique = list(set(candidate_skills))
    
    # Conceptual matching for \"Machine Learning\" in candidate skills
    # If \"Machine Learning\" is a JD skill, and candidate has TensorFlow/Keras/PyTorch, consider it a match
    conceptual_matches = set()
    jd_skills_lower = [s.lower() for s in jd_skills_unique]
    candidate_skills_lower = [s.lower() for s in candidate_skills_unique]

    # Direct matches
    matched_direct = set(s for s in jd_skills_unique if s.lower() in candidate_skills_lower)
    
    if "machine learning" in jd_skills_lower:
        if any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
            conceptual_matches.add("Machine Learning")

    matched = list(matched_direct | conceptual_matches)
    return sorted(matched)

def extract_missing_skills(required_jd_skills: List[str], candidate_skills: List[str]) -> List[str]:
    """Identifies and returns required JD skills missing from candidate profile."""
    # Consider conceptual matches for ML skills
    missing = []
    candidate_skills_lower = [s.lower() for s in candidate_skills]
    for skill in required_jd_skills:
        if skill.lower() == "machine learning":
            if not any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
                missing.append(skill)
        elif skill.lower() not in candidate_skills_lower:
            missing.append(skill)
    return sorted(missing)

def extract_required_skill_matches(required_jd_skills: List[str], candidate_skills: List[str]) -> int:
    """Counts the number of required JD skills matched in candidate profile."""
    matches = 0
    candidate_skills_lower = [s.lower() for s in candidate_skills]
    for skill in required_jd_skills:
        if skill.lower() == "machine learning":
            if any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
                matches += 1
        elif skill.lower() in candidate_skills_lower:
            matches += 1
    return matches

def extract_preferred_skill_matches(preferred_jd_skills: List[str], candidate_skills: List[str]) -> int:
    """Counts the number of preferred JD skills matched in candidate profile."""
    return len(set(preferred_jd_skills) & set(candidate_skills))

def extract_candidate_experience_years(candidate_profile: Dict) -> float:
    """Extracts candidate\"s total years of experience."""
    return safe_float(safe_get(candidate_profile, ["experience", "total_years"], 0.0))

def extract_required_experience_years(parsed_jd: Dict) -> float:
    """Extracts required years of experience from JD."""
    return safe_float(safe_get(parsed_jd, ["experience_requirements", "min_years"], 0.0))

def extract_experience_gap_years(candidate_exp: float, required_exp: float) -> float:
    """Calculates the experience gap in years."""
    return max(0.0, required_exp - candidate_exp)

def extract_matched_certifications(jd_certs: List[str], candidate_certs: List[str]) -> List[str]:
    """Identifies and returns matched certifications."""
    return list(set(jd_certs) & set(candidate_certs))

def extract_matched_languages(jd_languages: List[str], candidate_languages: List[str]) -> List[str]:
    """Identifies and returns matched programming languages."""
    return list(set(jd_languages) & set(candidate_languages))

def extract_matched_frameworks(jd_frameworks: List[str], candidate_frameworks: List[str]) -> List[str]:
    """Identifies and returns matched frameworks."""
    return list(set(jd_frameworks) & set(candidate_frameworks))

def extract_matched_databases(jd_databases: List[str], candidate_databases: List[str]) -> List[str]:
    """Identifies and returns matched databases."""
    return list(set(jd_databases) & set(candidate_databases))

def extract_matched_cloud_platforms(jd_cloud: List[str], candidate_cloud: List[str]) -> List[str]:
    """Identifies and returns matched cloud platforms."""
    return list(set(jd_cloud) & set(candidate_cloud))

def extract_matched_devops_tools(jd_devops: List[str], candidate_devops: List[str]) -> List[str]:
    """Identifies and returns matched DevOps tools."""
    return list(set(jd_devops) & set(candidate_devops))

def extract_matched_ai_ml_skills(jd_aiml: List[str], candidate_aiml: List[str]) -> List[str]:
    """Identifies and returns matched AI/ML skills."""
    return list(set(jd_aiml) & set(candidate_aiml))

def extract_matched_soft_skills(jd_soft_skills: List[str], candidate_soft_skills: List[str]) -> List[str]:
    """Identifies and returns matched soft skills."""
    return list(set(jd_soft_skills) & set(candidate_soft_skills))

def extract_keyword_matches(jd_text: str, candidate_text: str, keywords: List[str]) -> List[str]:
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

def normalize_skill_overlap(jd_skills: List[str], candidate_skills: List[str]) -> float:
    """Normalizes skill overlap between JD and candidate."""
    return calculate_overlap(jd_skills, candidate_skills)

def normalize_required_skill_coverage(required_jd_skills: List[str], candidate_skills: List[str]) -> float:
    """Normalizes coverage of required skills."""
    if not required_jd_skills:
        return 0.0
    matched = 0
    candidate_skills_lower = [s.lower() for s in candidate_skills]
    for skill in required_jd_skills:
        if skill.lower() == "machine learning":
            if any(ml_skill in candidate_skills_lower for ml_skill in ["tensorflow", "keras", "pytorch"]):
                matched += 1
        elif skill.lower() in candidate_skills_lower:
            matched += 1
    return calculate_percentage(matched, len(required_jd_skills))

def normalize_preferred_skill_coverage(preferred_jd_skills: List[str], candidate_skills: List[str]) -> float:
    """Normalizes coverage of preferred skills."""
    if not preferred_jd_skills:
        return 0.0
    matched = len(set(preferred_jd_skills).intersection(candidate_skills))
    return calculate_percentage(matched, len(preferred_jd_skills))

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

def normalize_education_match(jd_education: List[str], candidate_education: List[str]) -> float:
    """Normalizes education match score."""
    # A simple approach: check if candidate has any of the required/preferred JD education levels
    if not jd_education or not candidate_education:
        return 0.0
    
    jd_education_lower = [edu.lower() for edu in jd_education]
    candidate_education_lower = [edu.lower() for edu in candidate_education]
    
    for edu in candidate_education_lower:
        if edu in jd_education_lower:
            return 1.0
    return 0.0

def normalize_certification_match(jd_certs: List[str], candidate_certs: List[str]) -> float:
    """Normalizes certification match score."""
    return calculate_overlap(jd_certs, candidate_certs)

def normalize_title_similarity(jd_title: str, candidate_title: str) -> float:
    """Normalizes job title similarity."""
    return cosine_similarity(jd_title, candidate_title)

def normalize_seniority_match(jd_seniority: str, candidate_seniority: str) -> float:
    """Normalizes seniority level match."""
    if not jd_seniority or not candidate_seniority:
        return 0.0
    return 1.0 if jd_seniority.lower() == candidate_seniority.lower() else 0.0

def normalize_domain_match(jd_domain: List[str], candidate_domains: List[str]) -> float:
    """Normalizes domain match score."""
    return calculate_overlap(jd_domain, candidate_domains)

def normalize_industry_match(jd_industry: List[str], candidate_industries: List[str]) -> float:
    """Normalizes industry match score."""
    return calculate_overlap(jd_industry, candidate_industries)

def normalize_location_match(jd_location: List[str], candidate_location: List[str]) -> float:
    """Normalizes location match score."""
    # Assuming a match if any location in candidate list matches any in JD list
    if not jd_location or not candidate_location:
        return 0.0
    
    jd_loc_lower = [loc.lower() for loc in jd_location]
    candidate_loc_lower = [loc.lower() for loc in candidate_location]
    
    for loc in candidate_loc_lower:
        if loc in jd_loc_lower:
            return 1.0
    return 0.0

def normalize_employment_type_match(jd_emp_type: List[str], candidate_emp_type: List[str]) -> float:
    """Normalizes employment type match score."""
    # Assuming a match if any employment type in candidate list matches any in JD list
    if not jd_emp_type or not candidate_emp_type:
        return 0.0
    
    jd_emp_lower = [et.lower() for et in jd_emp_type]
    candidate_emp_lower = [et.lower() for et in candidate_emp_type]
    
    for et in candidate_emp_lower:
        if et in jd_emp_lower:
            return 1.0
    return 0.0

def normalize_language_match(jd_languages: List[str], candidate_languages: List[str]) -> float:
    """Normalizes programming language match score."""
    return calculate_overlap(jd_languages, candidate_languages)

def normalize_framework_match(jd_frameworks: List[str], candidate_frameworks: List[str]) -> float:
    """Normalizes framework match score."""
    return calculate_overlap(jd_frameworks, candidate_frameworks)

def normalize_database_match(jd_databases: List[str], candidate_databases: List[str]) -> float:
    """Normalizes database match score."""
    return calculate_overlap(jd_databases, candidate_databases)

def normalize_cloud_match(jd_cloud: List[str], candidate_cloud: List[str]) -> float:
    """Normalizes cloud platform match score."""
    return calculate_overlap(jd_cloud, candidate_cloud)

def normalize_devops_match(jd_devops: List[str], candidate_devops: List[str]) -> float:
    """Normalizes DevOps tools match score."""
    return calculate_overlap(jd_devops, candidate_devops)

def normalize_ai_ml_match(jd_aiml: List[str], candidate_aiml: List[str]) -> float:
    """Normalizes AI/ML skills match score."""
    return calculate_overlap(jd_aiml, candidate_aiml)

def normalize_soft_skill_match(jd_soft_skills: List[str], candidate_soft_skills: List[str]) -> float:
    """Normalizes soft skills match score."""
    return calculate_overlap(jd_soft_skills, candidate_soft_skills)

def normalize_keyword_similarity(jd_text: str, candidate_text: str) -> float:
    """Normalizes keyword similarity between JD and candidate texts."""
    return cosine_similarity(jd_text, candidate_text)

def normalize_technology_stack_match(jd_tech_stack: List[str], candidate_tech_stack: List[str]) -> float:
    """Normalizes technology stack match score."""
    return calculate_overlap(jd_tech_stack, candidate_tech_stack)
