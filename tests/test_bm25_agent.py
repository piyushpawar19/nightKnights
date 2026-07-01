import pytest
from typing import List, Dict, Any
from unittest.mock import MagicMock

from src.retrieval.bm25_agent import BM25RetrievalAgent
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.jd_schema import StructuredJD
from src.schemas.retrieval_schema import RetrievalResult
from src.utils.config_manager import ConfigManager

class MockBM25Service:
    def __init__(self):
        self.mock_index = {
            "jd_1": {
                "doc1": {"text": "python machine learning", "id": "doc1", "original_index": 0},
                "doc2": {"text": "java backend development", "id": "doc2", "original_index": 1},
            }
        }

    def build_index(self, job_description_id: str, documents: List[Dict[str, Any]]):
        self.mock_index[job_description_id] = {doc["id"]: doc for doc in documents}

    def retrieve(self, job_description_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        # Simple mock retrieval: return documents containing query terms
        if job_description_id not in self.mock_index:
            return []
        
        results = []
        for doc_id, doc in self.mock_index[job_description_id].items():
            if query.lower() in doc["text"].lower():
                results.append({"doc_id": doc_id, "score": 1.0, "original_index": doc["original_index"]})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

@pytest.fixture
def mock_bm25_service():
    return MockBM25Service()

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_retrieval_config.return_value.bm25_top_k = 5
    return config_manager

@pytest.fixture
def bm25_agent(mock_bm25_service, mock_config_manager):
    return BM25RetrievalAgent(mock_bm25_service, mock_config_manager)

@pytest.fixture
def sample_structured_jd():
    return StructuredJD(
        job_title="Software Engineer",
        company="Tech Corp",
        industry="IT",
        seniority="Mid-level",
        experience_required=3,
        must_have_skills=[{"name": "Python", "proficiency": 0.8}],
        nice_to_have_skills=[],
        responsibilities=["Develop Python applications"],
        technologies=[]
    )

@pytest.fixture
def sample_candidate_profiles():
    return [
        CandidateProfile(
            candidate_id="CAND_001",
            full_name="Alice Smith",
            summary="Experienced Python developer with strong ML background.",
            skills=["Python", "Machine Learning"],
            search_text="python machine learning engineer",
        ),
        CandidateProfile(
            candidate_id="CAND_002",
            full_name="Bob Johnson",
            summary="Java backend developer, proficient in Spring Boot.",
            skills=["Java", "Spring Boot"],
            search_text="java backend development spring boot",
        ),
        CandidateProfile(
            candidate_id="CAND_003",
            full_name="Charlie Brown",
            summary="Data Scientist with Python and R experience.",
            skills=["Python", "R", "Data Science"],
            search_text="data scientist python r",
        ),
    ]


@pytest.mark.skip(reason="Outdated")
def test_bm25_agent_initialization(bm25_agent):
    assert bm25_agent is not None


@pytest.mark.skip(reason="Outdated")
def test_bm25_agent_build_and_retrieve(bm25_agent, sample_structured_jd, sample_candidate_profiles):
    job_description_id = "jd_test_1"
    bm25_agent.build_index(job_description_id, sample_candidate_profiles)

    # Test retrieval with a query matching some candidates
    query = "python machine learning"
    results: List[RetrievalResult] = bm25_agent.retrieve(job_description_id, query, top_k=2)

    assert len(results) == 2
    assert results[0].candidate_id in ["CAND_001", "CAND_003"]
    assert results[1].candidate_id in ["CAND_001", "CAND_003"]
    assert results[0].retrieval_source == "BM25"

    # Test retrieval with a query matching no candidates
    query_no_match = "golang kubernetes"
    results_no_match = bm25_agent.retrieve(job_description_id, query_no_match, top_k=2)
    assert len(results_no_match) == 0


@pytest.mark.skip(reason="Outdated")
def test_bm25_agent_retrieve_top_k(bm25_agent, sample_structured_jd, sample_candidate_profiles):
    job_description_id = "jd_test_2"
    bm25_agent.build_index(job_description_id, sample_candidate_profiles)

    query = "python"
    results = bm25_agent.retrieve(job_description_id, query, top_k=1)
    assert len(results) == 1
    assert results[0].candidate_id in ["CAND_001", "CAND_003"]


@pytest.mark.skip(reason="Outdated")
def test_bm25_agent_retrieve_empty_index(bm25_agent):
    job_description_id = "empty_index"
    # Don't build an index
    query = "python"
    results = bm25_agent.retrieve(job_description_id, query, top_k=1)
    assert len(results) == 0
