import csv
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from src.evaluation.csv_generator import CSVGenerator
from src.evaluation.serializers import SubmissionSerializer
from src.evaluation.validators import ExportValidationError, SubmissionValidator
from src.models.domain_models import Explanation, RankedCandidate, RecruiterAssessment
from src.schemas.retrieval_schema import RetrievalResult


def _mock_retrieval(candidate_id: str) -> RetrievalResult:
    return RetrievalResult(
        candidate_id=candidate_id,
        dense_score=0.8,
        retrieval_source="mock",
        rank=1,
    )


MOCK_RANKED_CANDIDATES = [
    RankedCandidate(candidate_id="candidate1", hybrid_score=0.9, rank=1, retrieval_result=_mock_retrieval("candidate1")),
    RankedCandidate(candidate_id="candidate2", hybrid_score=0.8, rank=2, retrieval_result=_mock_retrieval("candidate2")),
]
MOCK_RECRUITER_ASSESSMENTS = [
    RecruiterAssessment(candidate_id="candidate1", technical_score=0.9, career_score=0.8, behavior_score=0.7, risk_score=0.1, culture_fit=0.9, hiring_confidence=0.95, final_score=0.9, metadata={}),
    RecruiterAssessment(candidate_id="candidate2", technical_score=0.8, career_score=0.7, behavior_score=0.6, risk_score=0.2, culture_fit=0.8, hiring_confidence=0.90, final_score=0.8, metadata={}),
]
MOCK_EXPLANATIONS = [
    Explanation(candidate_id="candidate1", summary="Strong technical skills.", strengths=["Python", "ML"], weaknesses=["SQL"], reasoning="Experience aligns.", recommendation="Hire", confidence=0.9, metadata={}),
    Explanation(candidate_id="candidate2", summary="Good communication.", strengths=["Communication"], weaknesses=["Leadership"], reasoning="Potential fit.", recommendation="Consider", confidence=0.8, metadata={}),
]

MOCK_EXPORT_SCHEMA = [
    "candidate_id",
    "rank",
    "hybrid_score",
    "recruiter_technical_score",
    "recruiter_career_score",
    "recruiter_behavior_score",
    "recruiter_risk_score",
    "recruiter_culture_fit",
    "recruiter_hiring_confidence",
    "recruiter_final_score",
    "explanation_summary",
    "explanation_strengths",
    "explanation_weaknesses",
    "explanation_reasoning",
    "explanation_recommendation",
    "explanation_confidence",
]


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def csv_generator():
    return CSVGenerator()


@pytest.fixture
def submission_validator():
    return SubmissionValidator()


@pytest.fixture
def submission_serializer():
    return SubmissionSerializer()


def test_write_csv_valid_data(csv_generator, temp_dir):
    filepath = temp_dir / "test.csv"
    data = [
        {"candidate_id": "1", "rank": 1, "hybrid_score": 0.9},
        {"candidate_id": "2", "rank": 2, "hybrid_score": 0.8},
    ]
    schema = ["candidate_id", "rank", "hybrid_score"]
    csv_generator.write_csv(filepath, data, schema)

    assert filepath.exists()
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["candidate_id"] == "1"
        assert rows[1]["hybrid_score"] == "0.8"


def test_write_csv_empty_data_raises_error(csv_generator, temp_dir):
    filepath = temp_dir / "empty.csv"
    with pytest.raises(ValueError, match="Data to write cannot be empty."):
        csv_generator.write_csv(filepath, [], MOCK_EXPORT_SCHEMA)


def test_write_csv_no_overwrite_raises_error(csv_generator, temp_dir):
    filepath = temp_dir / "existing.csv"
    filepath.touch()
    data = [
        {"candidate_id": "1", "rank": 1, "hybrid_score": 0.9},
    ]
    with pytest.raises(FileExistsError):
        csv_generator.write_csv(filepath, data, MOCK_EXPORT_SCHEMA, overwrite=False)


def test_write_csv_overwrite_succeeds(csv_generator, temp_dir):
    filepath = temp_dir / "existing.csv"
    filepath.touch()
    data = [
        {"candidate_id": "1", "rank": 1, "hybrid_score": 0.9},
    ]
    csv_generator.write_csv(filepath, data, MOCK_EXPORT_SCHEMA, overwrite=True)
    assert filepath.exists()


def test_write_csv_custom_delimiter(csv_generator, temp_dir):
    filepath = temp_dir / "delimited.csv"
    data = [
        {"candidate_id": "1", "rank": 1, "hybrid_score": 0.9},
    ]
    schema = ["candidate_id", "rank", "hybrid_score"]
    csv_generator.write_csv(filepath, data, schema, delimiter=";")
    with open(filepath, "r") as f:
        content = f.read()
        assert ";" in content
        assert "," not in content.split("\n")[0]  # Ensure comma isn't used as delimiter


def test_serialization_produces_correct_schema(submission_serializer):
    serialized_data = submission_serializer.serialize(
        MOCK_RANKED_CANDIDATES,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    assert isinstance(serialized_data, list)
    assert len(serialized_data) == len(MOCK_RANKED_CANDIDATES)
    for row in serialized_data:
        assert set(row.keys()) == set(MOCK_EXPORT_SCHEMA)


def test_serialization_preserves_order(submission_serializer):
    # Reverse order for ranked candidates to test if serializer sorts them
    reversed_ranked_candidates = list(reversed(MOCK_RANKED_CANDIDATES))
    serialized_data = submission_serializer.serialize(
        reversed_ranked_candidates,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    assert serialized_data[0]["rank"] == 1
    assert serialized_data[1]["rank"] == 2


def test_validator_valid_data(submission_validator, submission_serializer):
    serialized_data = submission_serializer.serialize(
        MOCK_RANKED_CANDIDATES,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    # Should not raise any error
    submission_validator.validate(serialized_data, MOCK_EXPORT_SCHEMA)


def test_validator_empty_data_raises_error(submission_validator):
    with pytest.raises(ExportValidationError, match="Export data cannot be empty."):
        submission_validator.validate([], MOCK_EXPORT_SCHEMA)


def test_validator_duplicate_ids_raises_error(submission_validator, submission_serializer):
    duplicate_candidates = [
        RankedCandidate(candidate_id="candidate1", hybrid_score=0.9, rank=1, retrieval_result=_mock_retrieval("candidate1")),
        RankedCandidate(candidate_id="candidate1", hybrid_score=0.8, rank=2, retrieval_result=_mock_retrieval("candidate1")),
    ]
    serialized_data = submission_serializer.serialize(
        duplicate_candidates,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    with pytest.raises(ExportValidationError, match="Duplicate candidate IDs found: candidate1"):
        submission_validator.validate(serialized_data, MOCK_EXPORT_SCHEMA)


def test_validator_invalid_ranks_raises_error(submission_validator, submission_serializer):
    invalid_ranked_candidates = [
        RankedCandidate(candidate_id="candidate1", hybrid_score=0.9, rank=1, retrieval_result=_mock_retrieval("candidate1")),
        RankedCandidate(candidate_id="candidate2", hybrid_score=0.8, rank=3, retrieval_result=_mock_retrieval("candidate2")),
    ]
    serialized_data = submission_serializer.serialize(
        invalid_ranked_candidates,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    with pytest.raises(ExportValidationError, match="Invalid or non-sequential ranks found."):
        submission_validator.validate(serialized_data, MOCK_EXPORT_SCHEMA)


def test_validator_invalid_scores_raises_error(submission_validator, submission_serializer):
    serialized_data = submission_serializer.serialize(
        MOCK_RANKED_CANDIDATES,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    serialized_data[0]["hybrid_score"] = 1.1
    with pytest.raises(ExportValidationError, match="Invalid hybrid_score in row 1: 1.1. Must be between 0.0 and 1.0."):
        submission_validator.validate(serialized_data, MOCK_EXPORT_SCHEMA)


def test_validator_schema_mismatch_raises_error(submission_validator, submission_serializer):
    invalid_schema = ["candidate_id", "rank", "non_existent_field"]
    serialized_data = submission_serializer.serialize(
        MOCK_RANKED_CANDIDATES,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    with pytest.raises(ExportValidationError, match="Schema mismatch in row 1."):
        submission_validator.validate(serialized_data, invalid_schema)


def test_validator_null_values_raises_error(submission_validator, submission_serializer):
    serialized_data = submission_serializer.serialize(
        MOCK_RANKED_CANDIDATES[:1],
        MOCK_RECRUITER_ASSESSMENTS[:1],
        MOCK_EXPLANATIONS[:1],
        MOCK_EXPORT_SCHEMA,
    )
    serialized_data[0]["recruiter_final_score"] = None
    with pytest.raises(ExportValidationError, match='Null or empty value found in row 1, field "recruiter_final_score".'):
        submission_validator.validate(serialized_data, MOCK_EXPORT_SCHEMA)


def test_validator_empty_explanation_raises_error(submission_validator, submission_serializer):
    serialized_data = submission_serializer.serialize(
        MOCK_RANKED_CANDIDATES,
        MOCK_RECRUITER_ASSESSMENTS,
        MOCK_EXPLANATIONS,
        MOCK_EXPORT_SCHEMA,
    )
    serialized_data[0]["explanation_summary"] = ""
    with pytest.raises(ExportValidationError, match='Null or empty value found in row 1, field "explanation_summary".'):
        submission_validator.validate(serialized_data, MOCK_EXPORT_SCHEMA)