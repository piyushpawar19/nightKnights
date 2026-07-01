import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import os
from datetime import datetime, timezone

# Adjust the path to import modules from the src directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import run_pipeline # This might need adjustment based on actual main.py structure
from graph.graph import build_graph, run_pipeline as graph_run_pipeline
from state.pipeline_state import create_initial_state, PipelineState
from schemas.graph_schema import CandidateRecord, NodeError # Import NodeError
from schemas.jd_schema import JobInfo, Requirements, Skills, Responsibilities, Preferences, ParsingMetadata
from schemas.jd_schema import ParsedJD
from utils.validate_submission import validate_submission as validate_submission_csv

# Mock data for testing
VALID_JD = """
Job Title: Software Engineer
Company: Example Corp
Location: Remote
Responsibilities:
- Develop and maintain web applications.
- Collaborate with cross-functional teams.
Skills:
- Python (5+ years)
- Django
- AWS
"""

VALID_CANDIDATE_1 = {
    "candidate_id": "CAND_0000001",
    "headline": "Experienced Python Developer",
    "summary": "5 years of experience in Python and Django.",
    "years_of_experience": 5.0,
    "current_title": "Software Engineer",
    "current_company": "Tech Solutions",
    "skills": ["Python", "Django", "AWS"],
    "location": "Remote",
    "raw_data": {},
}

VALID_CANDIDATE_2 = {
    "candidate_id": "CAND_0000002",
    "headline": "Cloud Engineer",
    "summary": "Experienced in AWS and cloud technologies.",
    "years_of_experience": 3.0,
    "current_title": "Cloud Specialist",
    "current_company": "Cloud Innovations",
    "skills": ["AWS", "DevOps"],
    "location": "New York",
    "raw_data": {},
}

# --- Fixtures ---
@pytest.fixture
def mock_pipeline_modules():
    with (
        patch("graph.nodes.parse_jd_node") as mock_parse_jd,
        patch("graph.nodes.retrieve_candidates_node") as mock_retrieve_candidates,
        patch("graph.nodes.feature_engineering_node") as mock_feature_engineering,
        patch("graph.nodes.hybrid_ranking_node") as mock_hybrid_ranking,
        patch("graph.nodes.reranking_node") as mock_reranking,
        patch("graph.nodes.explanation_node") as mock_explanation,
        patch("graph.nodes.evaluation_node") as mock_evaluation,
        patch("graph.nodes.csv_generation_node") as mock_csv_generation,
        patch("utils.logger.get_logger", return_value=MagicMock()) as mock_get_logger,
    ):
        # Configure mock return values for successful pipeline run
        mock_parse_jd.return_value = lambda state: {**state, "parsed_jd": ParsedJD(
            job_info=JobInfo(title="Software Engineer", company="Example Corp", location="Remote"),
            requirements=Requirements(mandatory_requirements=["Develop web applications"], certifications=[], education=[]),
            skills=Skills(programming_languages=["Python", "Django"], cloud=["AWS"], technical_skills=[]),
            responsibilities=Responsibilities(responsibilities_list=["Develop and maintain web applications", "Collaborate with cross-functional teams"]),
            preferences=Preferences(),
            metadata=ParsingMetadata(parse_timestamp="2023-01-01T00:00:00Z", parser_version="1.0")
        )}
        mock_retrieve_candidates.return_value = lambda state: {**state, "retrieved_candidates": [VALID_CANDIDATE_1, VALID_CANDIDATE_2]}
        mock_feature_engineering.return_value = lambda state: {**state, "feature_vectors": []}
        mock_hybrid_ranking.return_value = lambda state: {**state, "ranked_candidates": [{"candidate_id": "CAND_0000001", "rank": 1, "overall_score": 0.9, "component_scores": {}}, {"candidate_id": "CAND_0000002", "rank": 2, "overall_score": 0.8, "component_scores": {}}]},
        mock_reranking.return_value = lambda state: {**state, "reranked_candidates": [{"candidate_id": "CAND_0000001", "rank": 1, "overall_score": 0.95, "component_scores": {}}, {"candidate_id": "CAND_0000002", "rank": 2, "overall_score": 0.85, "component_scores": {}}]},
        mock_explanation.return_value = lambda state: {**state, "explanations": []}
        mock_evaluation.return_value = lambda state: {**state, "evaluation_metrics": {}}
        mock_csv_generation.return_value = lambda state: {**state, "submission_path": "mock_submission.csv"}

        yield # This yields control to the test function


# --- Tests ---

def test_normal_execution(mock_pipeline_modules):
    """Test the pipeline with valid inputs for normal execution."""
    # Use the run_pipeline function from main.py, which is the entry point
    final_state = run_pipeline(VALID_JD)

    assert final_state is not None
    assert final_state.get("parsed_jd") is not None
    assert len(final_state.get("retrieved_candidates", [])) > 0
    assert len(final_state.get("reranked_candidates", [])) > 0
    # Assert that the submission_path ends with a .csv file and is not the mock path
    assert final_state.get("submission_path") is not None and final_state.get("submission_path").endswith(".csv")
    assert not final_state.get("errors")


def test_missing_required_jd_fields(mock_pipeline_modules):
    """Test pipeline robustness when JD parsing fails due to missing fields."""
    malformed_jd = """
    Company: Example Corp
    Skills:
    - Python
    """ # Missing Job Title and Responsibilities
    
    # Mock the jd_parser_agent to simulate a parsing failure
    with (
        patch("graph.nodes.parse_jd_node") as mock_parse_jd,
        patch("utils.logger.get_logger") as mock_get_logger,
    ):
        mock_get_logger.return_value = MagicMock()
        # Simulate an error or inability to parse due to missing fields
        mock_parse_jd.return_value = lambda state: {**state, "parsed_jd": None}

        final_state = run_pipeline(malformed_jd)
        
        assert final_state is not None
        assert final_state.get("parsed_jd") is None
        assert any(err.get("error_type") == "InvalidJDFormat" for err in final_state.get("errors", []))


def test_empty_candidate_dataset(mock_pipeline_modules):
    """Test pipeline behavior with an empty candidate dataset."""
    with (
        patch("graph.nodes.retrieve_candidates_node") as mock_retrieve_candidates,
        patch("utils.logger.get_logger") as mock_get_logger,
    ):
            mock_get_logger.return_value = MagicMock()
            # Make retrieve_candidates_node return an empty list for retrieved_candidates, but do not override retrieved_candidates_raw
            # The internal logic of retrieve_candidates_node will handle the empty list if it's not provided via retrieved_candidates_raw
            mock_retrieve_candidates.return_value = lambda state: {**state, "retrieved_candidates": [], "retrieved_candidates_raw": []}

            final_state = run_pipeline(VALID_JD)

            assert final_state is not None
            assert final_state.get("parsed_jd") is not None
            assert not final_state.get("retrieved_candidates")
            assert not final_state.get("ranked_candidates")
            assert not final_state.get("reranked_candidates")
            assert not final_state.get("explanations")
            # Expect no errors for gracefully handling empty data
            assert final_state.get("errors") == [] or not final_state.get("errors") # Explicitly check for empty list or None


def test_malformed_candidate_profiles(mock_pipeline_modules):
    """Test pipeline handling of malformed candidate profiles."""
    malformed_candidate = {
        "candidate_id": "INVALID_ID", # Malformed ID
        "headline": "Tester",
        "years_of_experience": "five", # Invalid type
        "skills": ["Python"],
        "raw_data": {},
    }

    with (
        patch("graph.nodes.retrieve_candidates_node") as mock_retrieve_candidates,
        patch("utils.logger.get_logger") as mock_get_logger,
    ):
        mock_get_logger.return_value = MagicMock()
        # Simulate retrieval of one good and one malformed candidate
        mock_retrieve_candidates.return_value = lambda state: {**state, "retrieved_candidates_raw": [VALID_CANDIDATE_1, malformed_candidate]}
        
        final_state = run_pipeline(VALID_JD)

        assert final_state is not None
        assert any(err.get("error_type") == "MalformedCandidateProfile" for err in final_state.get("errors", []))
        assert len(final_state.get("retrieved_candidates", [])) == 1 # The malformed candidate should be filtered


def test_invalid_jds(mock_pipeline_modules):
    """Test pipeline behavior with completely invalid job descriptions."""
    invalid_jd_content = "This is not a valid job description format at all. Just some random text."

    with (
        patch("graph.nodes.parse_jd_node") as mock_parse_jd,
        patch("utils.logger.get_logger") as mock_get_logger,
    ):
        mock_get_logger.return_value = MagicMock()
        mock_parse_jd.return_value = lambda state: {**state, "parsed_jd": None, "errors": state.get("errors", []) + [NodeError(
            node_name="parse_jd",
            error_type="InvalidJDFormat",
            error_message="JD could not be parsed",
            timestamp=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        ).model_dump()]}
        final_state = run_pipeline(invalid_jd_content)

        assert final_state is not None
        assert final_state.get("parsed_jd") is None
        assert any(err.get("error_type") == "InvalidJDFormat" for err in final_state.get("errors", []))


def test_duplicate_candidate_ids_in_retrieval(mock_pipeline_modules):
    """Test handling of duplicate candidate IDs in the retrieved candidates list."""
    duplicate_candidate_list = [VALID_CANDIDATE_1, VALID_CANDIDATE_1, VALID_CANDIDATE_2]

    with (
        patch("graph.nodes.retrieve_candidates_node") as mock_retrieve_candidates,
        patch("utils.logger.get_logger") as mock_get_logger,
    ):
        mock_get_logger.return_value = MagicMock()
        mock_retrieve_candidates.return_value = lambda state: {**state, "retrieved_candidates_raw": duplicate_candidate_list}
        
        # Expect that the pipeline or a downstream node will log an error or de-duplicate
        # For now, let\"s just check for an error being logged and deduplication.
        final_state = run_pipeline(VALID_JD)

        assert final_state is not None
        assert any(err.get("error_type") == "DuplicateCandidateID" for err in final_state.get("errors", []))
        assert len(final_state.get("retrieved_candidates", [])) == 2 # Expect a deduplicated list


def test_submission_validation_utility():
    """Test the standalone submission validation utility with various invalid scenarios."""
    # Create a dummy CSV file for testing
    test_csv_path = Path("nightKnights/tests/test_submission.csv")
    
    # Test 1: Invalid Header
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,score,rank,reasoning\n") # Incorrect order
        f.write("CAND_0000001,0.9,1,Good fit\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert any("Row 1 (header) must be exactly" in err for err in errors)
    test_csv_path.unlink()

    # Test 2: Incorrect number of data rows
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for i in range(50):
            f.write(f"CAND_{i:07d},{i+1},{(100-i)/100:.2f},Reason {i}\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert any(f"must be exactly {100} data rows" in err for err in errors)
    test_csv_path.unlink()

    # Test 3: Duplicate Candidate ID
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for i in range(99):
            f.write(f"CAND_{i:07d},{i+1},{(100-i)/100:.2f},Reason {i}\n")
        f.write(f"CAND_0000000,100,0.01,Reason 99\n") # Duplicate with CAND_0000000
    errors = validate_submission_csv(str(test_csv_path))
    assert any("duplicate candidate_id CAND_0000000" in err for err in errors)
    test_csv_path.unlink()

    # Test 4: Duplicate Rank
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        # Generate 99 unique candidates
        for i in range(99):
            f.write(f"CAND_{i:07d},{i+1},{(100-i)/100:.2f},Reason {i}\n")
        # Add one more candidate with a duplicate rank (e.g., rank 1)
        f.write(f"CAND_0000099,1,0.01,Reason 99\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert any("duplicate rank 1" in err for err in errors)
    test_csv_path.unlink()

    # Test 5: Invalid Rank (not an integer)
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        f.write("CAND_0000001,1.5,0.9,Reason\n")
        for i in range(2, 101):
            f.write(f"CAND_{i:07d},{i},{(100-i)/100:.2f},Reason {i}\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert any("rank must be an integer" in err for err in errors)
    test_csv_path.unlink()

    # Test 6: Score not non-increasing
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        f.write("CAND_0000001,1,0.5,Reason\n")
        f.write("CAND_0000002,2,0.6,Reason\n") # Score increased
        for i in range(3, 101):
            f.write(f"CAND_{i:07d},{i},{(100-i)/100:.2f},Reason {i}\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert any("score must be non-increasing by rank" in err for err in errors)
    test_csv_path.unlink()

    # Test 7: Tie-break violation (equal scores, candidate_id not ascending)
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        f.write("CAND_0000002,1,0.8,Reason\n")
        f.write("CAND_0000001,2,0.8,Reason\n") # Equal scores, ID not ascending
        for i in range(3, 101):
            f.write(f"CAND_{i:07d},{i},{(100-i)/100:.2f},Reason {i}\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert any("tie-break requires candidate_id ascending" in err for err in errors)
    test_csv_path.unlink()

    # Test 8: Valid submission
    with open(test_csv_path, "w") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for i in range(100):
            f.write(f"CAND_{i:07d},{i+1},{(100-i)/100:.2f},Reason {i}\n")
    errors = validate_submission_csv(str(test_csv_path))
    assert not errors
    test_csv_path.unlink()
