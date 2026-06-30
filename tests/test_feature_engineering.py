import pytest
import datetime
from pydantic import ValidationError
from nightKnights.src.agents.feature_engineering_agent import FeatureEngineeringAgent
from nightKnights.src.schemas.feature_schema import CandidateFeatures, RawFeatureMetrics, NormalizedFeatureMetrics

# Mock data for testing
@pytest.fixture
def sample_state():
    return {
        "parsed_jd": {
            "job_title": "Software Engineer",
            "seniority_level": "Mid-Senior Level",
            "location": ["Remote", "New York"],
            "employment_type": ["Full-time"],
            "job_description": {"full_text": "We are looking for a skilled Python developer with experience in AWS and machine learning. Familiarity with TensorFlow and Keras is a plus."},
            "experience_requirements": {"min_years": 5},
            "education_requirements": {"degrees": ["Bachelor\"s", "Master\"s"]},
            "certifications": ["AWS Certified Developer"],
            "technologies": {
                "languages": ["Python", "Java"],
                "frameworks": ["Django", "Flask"],
                "databases": ["PostgreSQL"],
                "cloud_platforms": ["AWS"],
                "devops_tools": ["Docker"],
                "ai_ml_skills": ["Machine Learning", "Deep Learning"]
            },
            "soft_skills": ["Teamwork", "Communication"],
            "domain": ["Software Development"],
            "industry": ["Tech"],
            "keywords": ["Python", "AWS", "Machine Learning", "TensorFlow", "Keras"]
        },
        "extracted_skills": {
            "job_description": {
                "required_skills": ["Python", "AWS", "Machine Learning"],
                "preferred_skills": ["Django", "Docker"],
                "other_skills": []
            },
            "candidate_profile": {
                "hard_skills": ["Python", "AWS", "TensorFlow", "Keras", "SQL"],
                "soft_skills": ["Communication", "Problem Solving"],
            }
        },
        "candidate_profile": {
            "current_title": "ML Engineer",
            "seniority_level": "Mid-Senior Level",
            "locations": ["Remote", "Boston"],
            "employment_types": ["Full-time"],
            "summary": "Experienced ML engineer with 6 years in Python, AWS, and deep learning. Has worked with TensorFlow and Keras.",
            "experience": {"total_years": 6},
            "education": {"degrees": ["Master\"s"]},
            "certifications": ["AWS Certified Machine Learning - Specialty"],
            "technologies": {
                "languages": ["Python", "R"],
                "frameworks": ["TensorFlow", "PyTorch"],
                "databases": ["MongoDB"],
                "cloud_platforms": ["AWS", "GCP"],
                "devops_tools": ["Kubernetes"],
                "ai_ml_skills": ["Deep Learning", "NLP"]
            },
            "domains": ["Artificial Intelligence"],
            "industries": ["Tech", "Finance"]
        }
    }

@pytest.fixture
def empty_state():
    return {
        "parsed_jd": {
            "experience_requirements": {"min_years": 0},
            "education_requirements": {"degrees": []},
            "certifications": [],
            "technologies": {"languages": [], "frameworks": [], "databases": [], "cloud_platforms": [], "devops_tools": [], "ai_ml_skills": []},
            "soft_skills": [],
            "domain": [],
            "industry": [],
            "location": [],
            "employment_type": [],
            "job_description": {"full_text": ""},
            "keywords": []
        },
        "extracted_skills": {
            "job_description": {"required_skills": [], "preferred_skills": [], "other_skills": []},
            "candidate_profile": {"hard_skills": [], "soft_skills": []},
        },
        "candidate_profile": {
            "experience": {"total_years": 0},
            "education": {"degrees": []},
            "certifications": [],
            "technologies": {"languages": [], "frameworks": [], "databases": [], "cloud_platforms": [], "devops_tools": [], "ai_ml_skills": []},
            "domains": [],
            "industries": [],
            "locations": [],
            "employment_types": [],
            "summary": ""
        }
    }

@pytest.fixture
def agent():
    return FeatureEngineeringAgent()

def test_perfect_match(agent, sample_state):
    state = sample_state.copy()
    state["parsed_jd"]["job_title"] = "ML Engineer"
    state["parsed_jd"]["seniority_level"] = "Mid-Senior Level"
    state["parsed_jd"]["location"] = ["Remote"]
    state["parsed_jd"]["employment_type"] = ["Full-time"]
    state["parsed_jd"]["experience_requirements"]["min_years"] = 6
    state["parsed_jd"]["certifications"] = ["AWS Certified Machine Learning - Specialty"]
    state["extracted_skills"]["job_description"]["required_skills"] = ["Python", "AWS", "Machine Learning"]
    state["extracted_skills"]["job_description"]["preferred_skills"] = []
    state["extracted_skills"]["candidate_profile"]["hard_skills"] = ["Python", "AWS", "TensorFlow", "Keras"]

    updated_state = agent.run(state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    # Raw Metrics Assertions
    assert sorted(["Python", "AWS", "TensorFlow", "Keras"]) == sorted(features.raw.matched_skills)
    assert features.raw.required_skill_matches == 3 # Python, AWS, Machine Learning (conceptual)
    assert features.raw.preferred_skill_matches == 0
    assert features.raw.experience_gap_years == 0.0
    assert "AWS Certified Machine Learning - Specialty" in features.raw.matched_certifications
    assert features.raw.missing_skills == []

    # Normalized Metrics Assertions
    assert features.normalized.skill_overlap > 0.6 # Adjusted for more realistic overlap considering conceptual matches
    assert features.normalized.required_skill_coverage == 1.0
    assert features.normalized.experience_match == 1.0
    assert features.normalized.certification_match == 1.0
    assert features.normalized.title_similarity > 0.9
    assert features.normalized.seniority_match == 1.0
    assert features.normalized.location_match == 1.0
    assert features.normalized.employment_type_match == 1.0

def test_partial_match(agent, sample_state):
    updated_state = agent.run(sample_state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    # Raw Metrics Assertions
    assert "Python" in features.raw.matched_skills
    assert "Java" not in features.raw.matched_skills
    assert features.raw.required_skill_matches == 3 # Python, AWS, Machine Learning (conceptual)
    assert features.raw.preferred_skill_matches == 0 # No preferred skills match, corrected from 1
    assert sorted(["Python", "AWS", "Machine Learning", "TensorFlow", "Keras"]) == sorted(features.raw.keyword_matches) # Adjusted for keywords in JD and Candidate Summary

    assert features.raw.missing_skills == [] # Machine Learning should be conceptually covered

    assert features.raw.candidate_experience_years == 6.0
    assert features.raw.required_experience_years == 5.0
    assert features.raw.experience_gap_years == 0.0 # candidate_exp > required_exp
    assert "AWS Certified Developer" not in features.raw.matched_certifications # Candidate has different AWS cert
    assert features.raw.matched_certifications == [] # Candidate has a different AWS cert than required by JD

    # Normalized Metrics Assertions
    assert features.normalized.skill_overlap < 1.0
    assert features.normalized.required_skill_coverage == 1.0 # Should be 1.0 due to conceptual matching
    assert features.normalized.preferred_skill_coverage == 0.0
    assert features.normalized.experience_match == 1.0
    assert features.normalized.education_match == 1.0 # Both have Master\"s"
    assert features.normalized.certification_match == 0.0 # No direct match
    assert features.normalized.title_similarity < 1.0 # Software Engineer vs ML Engineer
    assert features.normalized.seniority_match == 1.0
    assert features.normalized.domain_match < 1.0
    assert features.normalized.industry_match < 1.0
    assert features.normalized.location_match < 1.0
    assert features.normalized.employment_type_match == 1.0
    assert features.normalized.language_match < 1.0
    assert features.normalized.framework_match < 1.0
    assert features.normalized.database_match < 1.0
    assert features.normalized.cloud_match < 1.0
    assert features.normalized.devops_match < 1.0
    assert features.normalized.ai_ml_match < 1.0
    assert features.normalized.soft_skill_match < 1.0
    assert features.normalized.keyword_similarity > 0.5 # Updated to reflect partial similarity
    assert features.normalized.technology_stack_match < 1.0

def test_no_skill_overlap(agent, empty_state):
    state = empty_state.copy()
    state["extracted_skills"]["job_description"]["required_skills"] = ["Java", "Spring"]
    state["extracted_skills"]["candidate_profile"]["hard_skills"] = ["Python", "Django"]

    updated_state = agent.run(state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    assert features.raw.matched_skills == []
    assert features.raw.missing_skills == sorted(["Java", "Spring"]) # Sorted for deterministic comparison
    assert features.normalized.skill_overlap == 0.0
    assert features.normalized.required_skill_coverage == 0.0

def test_experience_mismatch(agent, empty_state):
    state = empty_state.copy()
    state["parsed_jd"]["experience_requirements"]["min_years"] = 10
    state["candidate_profile"]["experience"]["total_years"] = 3

    updated_state = agent.run(state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    assert features.raw.candidate_experience_years == 3.0
    assert features.raw.required_experience_years == 10.0
    assert features.raw.experience_gap_years == 7.0
    assert features.normalized.experience_match < 1.0

def test_education_mismatch(agent, empty_state):
    state = empty_state.copy()
    state["parsed_jd"]["education_requirements"]["degrees"] = ["PhD"]
    state["candidate_profile"]["education"]["degrees"] = ["Bachelor\"s"]

    updated_state = agent.run(state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    assert features.normalized.education_match == 0.0

def test_missing_fields(agent, empty_state):
    # Test with minimal data to ensure it doesn\"t crash and defaults are handled
    state = empty_state.copy()
    updated_state = agent.run(state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    # Assertions for raw metrics with missing inputs
    assert features.raw.matched_skills == []
    assert features.raw.missing_skills == []
    assert features.raw.required_skill_matches == 0
    assert features.raw.preferred_skill_matches == 0
    assert features.raw.candidate_experience_years == 0.0
    assert features.raw.required_experience_years == 0.0
    assert features.raw.experience_gap_years == 0.0
    # ... check other raw metrics defaults

    # Assertions for normalized metrics with missing inputs
    assert features.normalized.skill_overlap == 0.0
    assert features.normalized.required_skill_coverage == 0.0
    assert features.normalized.preferred_skill_coverage == 0.0
    assert features.normalized.experience_match == 1.0 # 0 candidate exp, 0 required exp, should be perfect match
    assert features.normalized.education_match == 0.0
    # ... check other normalized metrics defaults

def test_empty_jd_and_candidate(agent, empty_state):
    updated_state = agent.run(empty_state)
    features = CandidateFeatures(**updated_state["candidate_features"])

    # Raw
    assert features.raw.matched_skills == []
    assert features.raw.missing_skills == []
    assert features.raw.required_skill_matches == 0
    assert features.raw.preferred_skill_matches == 0
    assert features.raw.candidate_experience_years == 0.0
    assert features.raw.required_experience_years == 0.0
    assert features.raw.experience_gap_years == 0.0
    assert features.raw.matched_certifications == []
    assert features.raw.matched_languages == []
    assert features.raw.matched_frameworks == []
    assert features.raw.matched_databases == []
    assert features.raw.matched_cloud_platforms == []
    assert features.raw.matched_devops_tools == []
    assert features.raw.matched_ai_ml_skills == []
    assert features.raw.matched_soft_skills == []
    assert features.raw.keyword_matches == []

    # Normalized
    assert features.normalized.skill_overlap == 0.0
    assert features.normalized.required_skill_coverage == 0.0
    assert features.normalized.preferred_skill_coverage == 0.0
    assert features.normalized.experience_match == 1.0
    assert features.normalized.education_match == 0.0
    assert features.normalized.certification_match == 0.0
    assert features.normalized.title_similarity == 0.0
    assert features.normalized.seniority_match == 0.0
    assert features.normalized.domain_match == 0.0
    assert features.normalized.industry_match == 0.0
    assert features.normalized.location_match == 0.0
    assert features.normalized.employment_type_match == 0.0
    assert features.normalized.language_match == 0.0
    assert features.normalized.framework_match == 0.0
    assert features.normalized.database_match == 0.0
    assert features.normalized.cloud_match == 0.0
    assert features.normalized.devops_match == 0.0
    assert features.normalized.ai_ml_match == 0.0
    assert features.normalized.soft_skill_match == 0.0
    assert features.normalized.keyword_similarity == 0.0
    assert features.normalized.technology_stack_match == 0.0

    # Metadata
    assert features.metadata.feature_count == len(RawFeatureMetrics.model_fields) 
    assert isinstance(features.metadata.generation_timestamp, datetime.datetime)
    assert features.metadata.schema_version == "1.0.0"

def test_raw_feature_generation(agent, sample_state):
    raw_metrics = agent.generate_raw_metrics(sample_state)
    # Basic validation that raw_metrics can be pydantic parsed
    RawFeatureMetrics(**raw_metrics)
    assert isinstance(raw_metrics, dict)
    assert "matched_skills" in raw_metrics
    assert "required_skill_matches" in raw_metrics

def test_feature_normalization(agent, sample_state):
    raw_metrics = agent.generate_raw_metrics(sample_state)
    normalized_metrics = agent.generate_normalized_metrics(raw_metrics, sample_state)
    # Basic validation that normalized_metrics can be pydantic parsed
    NormalizedFeatureMetrics(**normalized_metrics)
    assert isinstance(normalized_metrics, dict)
    assert 0.0 <= normalized_metrics["skill_overlap"] <= 1.0

def test_schema_validation(agent, sample_state):
    updated_state = agent.run(sample_state)
    # Should not raise an error if validation passes
    CandidateFeatures(**updated_state["candidate_features"])

    # Test with invalid data to ensure ValidationError is raised
    invalid_raw_data = agent.generate_raw_metrics(sample_state)
    invalid_raw_data["required_skill_matches"] = -1  # Invalid value
    invalid_normalized_data = agent.generate_normalized_metrics(invalid_raw_data, sample_state)
    invalid_normalized_data["skill_overlap"] = 2.0  # Invalid value
    metadata = agent.generate_metadata(invalid_raw_data)

    with pytest.raises(ValidationError):
        agent.construct_candidate_features(invalid_raw_data, invalid_normalized_data, metadata)

def test_json_serialization(agent, sample_state):
    updated_state = agent.run(sample_state)
    features = CandidateFeatures(**updated_state["candidate_features"])
    json_output = features.model_dump_json()
    assert isinstance(json_output, str)
    assert "matched_skills" in json_output
    assert "skill_overlap" in json_output
