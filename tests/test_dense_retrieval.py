import pytest
from unittest.mock import MagicMock
from typing import List

from retrieval.dense_retrieval_agent import DenseRetrievalAgent
from schemas.candidate_schema import CandidateProfile
from schemas.jd_schema import StructuredJD
from schemas.retrieval_schema import RetrievalResult
from utils.config_manager import ConfigManager

class MockEmbeddingService:
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Return dummy embeddings (e.g., one-hot encoded or simple sequence)
        return [[float(i + 1)] * 10 for i in range(len(texts))]

class MockVectorStoreManager:
    def __init__(self):
        self.mock_db = {}

    def add_candidates(self, candidates: List[CandidateProfile], jd_id: str):
        for i, candidate in enumerate(candidates):
            self.mock_db[candidate.candidate_id] = {
                "vector": [float(i + 1)] * 10, # Dummy vector
                "candidate": candidate
            }

    def search_candidates(self, query_vector: List[float], top_k: int, jd_id: str) -> List[Dict[str, Any]]:
        # Simple mock search: return candidates that "match" the query vector
        # In a real scenario, this would involve vector similarity search
        results = []
        for cand_id, data in self.mock_db.items():
            # For mock, just return candidates based on a simple rule
            if data["vector"][0] <= query_vector[0]:  # If candidate vector is "smaller" or equal
                results.append({"candidate_id": cand_id, "score": data["vector"][0] / query_vector[0]})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService()

@pytest.fixture
def mock_vector_store_manager():
    return MockVectorStoreManager()

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_retrieval_config.return_value.dense_top_k = 5
    config_manager.get_retrieval_config.return_value.embedding_model = "test-embedding-model"
    return config_manager

@pytest.fixture
def dense_retrieval_agent(mock_embedding_service, mock_vector_store_manager, mock_config_manager):
    return DenseRetrievalAgent(mock_embedding_service, mock_vector_store_manager, mock_config_manager)

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

def test_dense_retrieval_agent_initialization(dense_retrieval_agent):
    assert dense_retrieval_agent is not None

def test_dense_retrieval_agent_retrieve(dense_retrieval_agent, sample_structured_jd, sample_candidate_profiles):
    jd_id = "jd_test_1"
    # First add candidates to the vector store
    dense_retrieval_agent.add_candidates_to_vector_store(sample_candidate_profiles, jd_id)

    query = sample_structured_jd.job_title + " " + " ".join(s["name"] for s in sample_structured_jd.must_have_skills) # type: ignore
    results: List[RetrievalResult] = dense_retrieval_agent.retrieve(jd_id, query, top_k=2)

    assert len(results) == 2
    assert results[0].retrieval_source == "Dense"
    assert results[0].candidate_id in [c.candidate_id for c in sample_candidate_profiles]

def test_dense_retrieval_agent_retrieve_empty_candidates(dense_retrieval_agent, sample_structured_jd):
    jd_id = "jd_empty"
    # Don't add any candidates
    query = sample_structured_jd.job_title
    results = dense_retrieval_agent.retrieve(jd_id, query, top_k=5)
    assert len(results) == 0
