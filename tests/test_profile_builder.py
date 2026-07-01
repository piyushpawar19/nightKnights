import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
from datetime import datetime, timezone

from src.jd_parser.jd_parser import JDParser
from src.retrieval.profile_builder_agent import CandidateProfileBuilder
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.jd_schema import StructuredJD


@pytest.fixture
def mock_jd_parser():
    parser = MagicMock(spec=JDParser)
    parser.parse_job_description.return_value = StructuredJD(
        job_title="Software Engineer",
        company="Example Corp",
        industry="Tech",
        seniority="Mid-level",
        experience_required=3,
        must_have_skills=[],
        nice_to_have_skills=[],
        responsibilities=[],
        technologies=[]
    )
    return parser

@pytest.fixture
def profile_builder():
    return CandidateProfileBuilder()

@pytest.fixture
def sample_raw_candidate_data():
    return {
        "candidate_id": "CAND_001",
        "profile": {
            "anonymized_name": "John Doe",
            "headline": "Software Developer",
            "summary": "Experienced in Python.",
            "years_of_experience": 5.0,
            "current_title": "Developer",
            "current_company": "ABC Inc",
            "location": "Remote",
            "country": "USA",
        },
        "career_history": [
            {
                "company": "ABC Inc",
                "title": "Developer",
                "duration_months": 60,
                "is_current": True,
                "description": "Developed software."
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert"}
        ],
        "education": [],
        "certifications": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2020-01-01",
            "last_active_date": "2023-01-01",
            "open_to_work_flag": True,
            "applications_submitted_30d": 5,
            "connection_count": 100
        }
    }


@pytest.mark.skip(reason="Outdated")
def test_build_candidate_profile(profile_builder, sample_raw_candidate_data):
    profile = profile_builder.build_profile(sample_raw_candidate_data)

    assert isinstance(profile, CandidateProfile)
    assert profile.candidate_id == "CAND_001"
    assert profile.full_name == "John Doe"
    assert "python" in [s.lower() for s in profile.skills]
    assert profile.years_experience == 5.0
    assert profile.search_text is not None

@pytest.mark.skip(reason="Outdated")
def test_build_candidate_profile_missing_data(profile_builder, sample_raw_candidate_data):
    incomplete_data = sample_raw_candidate_data.copy()
    del incomplete_data["profile"]["anonymized_name"]
    # The profile builder should handle missing optional fields gracefully
    profile = profile_builder.build_profile(incomplete_data)
    assert profile.full_name == ""
    assert profile.summary == "Experienced in Python."

@pytest.mark.skip(reason="Outdated")
def test_build_candidate_profile_empty_skills(profile_builder, sample_raw_candidate_data):
    empty_skills_data = sample_raw_candidate_data.copy()
    empty_skills_data["skills"] = []
    profile = profile_builder.build_profile(empty_skills_data)
    assert profile.skills == []

@pytest.mark.skip(reason="Outdated")
def test_build_candidate_profile_search_text_generation(profile_builder, sample_raw_candidate_data):
    profile = profile_builder.build_profile(sample_raw_candidate_data)
    assert "john doe" in profile.search_text.lower()
    assert "software developer" in profile.search_text.lower()
    assert "python" in profile.search_text.lower()
    assert "abc inc" in profile.search_text.lower()

@pytest.mark.skip(reason="Outdated")
def test_build_candidate_profile_redrob_signals(profile_builder, sample_raw_candidate_data):
    profile = profile_builder.build_profile(sample_raw_candidate_data)
    assert profile.metadata["profile_completeness"] == 0.9
    assert profile.metadata["signup_date"] == datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert profile.metadata["applications_submitted_30d"] == 5

@pytest.mark.skip(reason="Outdated")
def test_build_profile_batch(profile_builder, sample_raw_candidate_data):
    data_batch = [
        sample_raw_candidate_data,
        {
            **sample_raw_candidate_data,
            "candidate_id": "CAND_002",
            "profile": {
                **sample_raw_candidate_data["profile"],
                "anonymized_name": "Jane Smith",
                "years_of_experience": 2.0,
            },
            "skills": [
                {"name": "Java", "proficiency": "intermediate"}
            ]
        }
    ]
    profiles = profile_builder.build_profiles(data_batch)
    assert len(profiles) == 2
    assert profiles[0].candidate_id == "CAND_001"
    assert profiles[1].candidate_id == "CAND_002"
    assert profiles[1].years_experience == 2.0
    assert "java" in [s.lower() for s in profiles[1].skills]


