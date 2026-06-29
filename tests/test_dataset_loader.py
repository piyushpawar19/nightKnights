import pytest
from pathlib import Path
from typing import List

from src.retrieval.dataset_loader import DatasetLoader
from schemas.candidate_schema import CandidateProfile
from utils.logger import get_logger

logger = get_logger(__name__)

@pytest.fixture
def sample_json_data(tmp_path):
    # Create a temporary JSON file for testing
    data = [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Jane Doe",
                "headline": "Senior Software Engineer",
                "summary": "Experienced engineer with a passion for clean code.",
                "location": "Boston",
                "country": "USA",
                "years_of_experience": 5.0,
                "current_title": "Software Engineer II",
                "current_company": "TechCorp"
            },
            "career_history": [
                {
                    "company": "TechCorp",
                    "title": "Software Engineer II",
                    "duration_months": 24,
                    "is_current": True,
                    "description": "Full-stack development using Python and React."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 10},
                {"name": "React", "proficiency": "advanced", "endorsements": 5}
            ],
            "education": [
                {
                    "institution": "Boston University",
                    "degree": "B.S.",
                    "field_of_study": "Computer Science",
                    "start_year": 2016,
                    "end_year": 2020,
                    "tier": "tier_2"
                }
            ],
            "certifications": [
                {"name": "AWS Solutions Architect", "issuer": "Amazon", "year": 2021}
            ],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2020-05-01",
                "last_active_date": "2026-06-25",
                "open_to_work_flag": True,
                "applications_submitted_30d": 3,
                "connection_count": 150
            }
        },
        {
            "candidate_id": "CAND_0000002",
            "profile": {
                "anonymized_name": "John Smith",
                "headline": "Junior Data Scientist",
                "summary": "Recent graduate eager to learn and contribute.",
                "location": "New York",
                "country": "USA",
                "years_of_experience": 1.0,
                "current_title": "Data Analyst",
                "current_company": "DataCo"
            },
            "career_history": [
                {
                    "company": "DataCo",
                    "title": "Data Analyst",
                    "duration_months": 12,
                    "is_current": True,
                    "description": "Assisted in data cleaning and visualization."
                }
            ],
            "skills": [
                {"name": "R", "proficiency": "intermediate", "endorsements": 3},
                {"name": "SQL", "proficiency": "advanced", "endorsements": 7}
            ],
            "education": [
                {
                    "institution": "NYU",
                    "degree": "M.S.",
                    "field_of_study": "Data Science",
                    "start_year": 2021,
                    "end_year": 2023,
                    "tier": "tier_1"
                }
            ],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 80.0,
                "signup_date": "2023-01-15",
                "last_active_date": "2026-06-20",
                "open_to_work_flag": False,
                "applications_submitted_30d": 1,
                "connection_count": 50
            }
        }
    ]
    file_path = tmp_path / "sample_candidates.json"
    with open(file_path, "w") as f:
        import json
        json.dump(data, f, indent=4)
    return file_path

@pytest.fixture
def dataset_loader():
    return DatasetLoader()

def test_load_candidates_from_json(dataset_loader, sample_json_data):
    candidates_data = dataset_loader.load_candidates_from_json(sample_json_data)

    assert isinstance(candidates_data, list)
    assert len(candidates_data) == 2

    # Check structure of loaded data
    candidate1 = candidates_data[0]
    assert "candidate_id" in candidate1
    assert "profile" in candidate1
    assert candidate1["profile"]["anonymized_name"] == "Jane Doe"

def test_load_candidates_from_json_non_existent_file(dataset_loader, tmp_path):
    non_existent_file = tmp_path / "non_existent.json"
    with pytest.raises(IOError, match=f"File not found: {non_existent_file}"):
        dataset_loader.load_candidates_from_json(non_existent_file)

def test_load_candidates_from_json_invalid_json_format(dataset_loader, tmp_path):
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("{ \"key\": \"value\" }") # Malformed JSON
    with pytest.raises(ValueError, match="Invalid JSON format in file:.*"):
        dataset_loader.load_candidates_from_json(invalid_json_file)

def test_load_candidates_from_json_empty_json_file(dataset_loader, tmp_path):
    empty_json_file = tmp_path / "empty.json"
    empty_json_file.write_text("[]")
    candidates_data = dataset_loader.load_candidates_from_json(empty_json_file)
    assert isinstance(candidates_data, list)
    assert len(candidates_data) == 0

def test_load_candidates_from_json_with_malformed_records(dataset_loader, tmp_path):
    malformed_data = [
        {
            "candidate_id": "CAND_003",
            "profile": {"anonymized_name": "Charlie"}
            # Missing other required fields to simulate malformed record
        },
        {
            "candidate_id": "CAND_004",
            "profile": {"anonymized_name": "David"}
        }
    ]
    file_path = tmp_path / "malformed_candidates.json"
    with open(file_path, "w") as f:
        import json
        json.dump(malformed_data, f, indent=4)
    
    # Expect that malformed records might be filtered or raise an error depending on strictness
    # For this test, we\"ll assume it attempts to load and might result in fewer valid candidates
    candidates_data = dataset_loader.load_candidates_from_json(file_path)
    assert isinstance(candidates_data, list)
    # The loader is designed to load raw dicts, validation happens later in profile_builder
    assert len(candidates_data) == 2
