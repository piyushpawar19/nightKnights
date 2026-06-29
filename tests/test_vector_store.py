import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any
from pathlib import Path
import faiss
import numpy as np

from retrieval.vector_store import VectorStoreManager
from schemas.candidate_schema import CandidateProfile
from utils.config_manager import ConfigManager

# Mock FAISS index for testing
class MockFAISSIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        self.vectors.extend(vectors.tolist())
        self.metadata.extend(metadata)

    def search(self, query_vectors: np.ndarray, k: int) -> List[Dict[str, Any]]:
        # Simple mock search: find the closest vector by L2 distance
        results = []
        for query_vec in query_vectors:
            min_dist = float("inf")
            closest_idx = -1
            for i, vec in enumerate(self.vectors):
                dist = np.linalg.norm(np.array(vec) - query_vec)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            if closest_idx != -1:
                results.append({
                    "candidate_id": self.metadata[closest_idx]["candidate_id"],
                    "score": float(min_dist), # Inverted for similarity
                    "original_index": self.metadata[closest_idx]["original_index"],
                })
        return sorted(results, key=lambda x: x["score"])[:k] # Sort by score ascending (distance)

@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    service.get_embeddings.side_effect = lambda texts: [[1.0] * 10 for _ in texts]
    return service

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_retrieval_config.return_value.embedding_dimension = 10
    config_manager.get_retrieval_config.return_value.faiss_index_type = "Flat"
    config_manager.get_retrieval_config.return_value.cache_embeddings = False
    return config_manager

@pytest.fixture
def vector_store_manager(mock_embedding_service, mock_config_manager, tmp_path):
    # Ensure the cache_dir is within the temporary path
    mock_config_manager.get_retrieval_config.return_value.embedding_cache_dir = tmp_path / "embeddings_cache"
    return VectorStoreManager(mock_embedding_service, mock_config_manager)

@pytest.fixture
def sample_candidate_profiles():
    return [
        CandidateProfile(
            candidate_id="CAND_001",
            full_name="Alice",
            summary="Python developer",
            search_text="Python development",
            skills=["Python"],
        ),
        CandidateProfile(
            candidate_id="CAND_002",
            full_name="Bob",
            summary="Java developer",
            search_text="Java development",
            skills=["Java"],
        ),
    ]

def test_vector_store_manager_initialization(vector_store_manager):
    assert vector_store_manager is not None
    assert isinstance(vector_store_manager._embedding_service, MagicMock)
    assert isinstance(vector_store_manager._config_manager, MagicMock)

def test_add_candidates(vector_store_manager, sample_candidate_profiles):
    jd_id = "JD_001"
    vector_store_manager.add_candidates(sample_candidate_profiles, jd_id)

    assert jd_id in vector_store_manager._faiss_indexes
    assert jd_id in vector_store_manager._candidate_metadata
    assert len(vector_store_manager._candidate_metadata[jd_id]) == 2
    assert vector_store_manager._faiss_indexes[jd_id].ntotal == 2  # Assuming 2 candidates added

def test_search_candidates(vector_store_manager, sample_candidate_profiles, mock_embedding_service):
    jd_id = "JD_002"
    vector_store_manager.add_candidates(sample_candidate_profiles, jd_id)

    query = "Python developer"
    mock_embedding_service.get_embeddings.return_value = [[1.0] * 10] # Mock query embedding
    
    results = vector_store_manager.search_candidates(query, top_k=1, jd_id=jd_id)
    assert len(results) == 1
    assert results[0]["candidate_id"] == "CAND_001" # Assuming CAND_001 is closest to query [1.0...]
    assert results[0]["score"] == 0.0 # Distance should be 0 with mocked embeddings and search

def test_search_candidates_empty_index(vector_store_manager):
    jd_id = "JD_empty"
    query = "some query"
    results = vector_store_manager.search_candidates(query, top_k=1, jd_id=jd_id)
    assert len(results) == 0

def test_remove_index(vector_store_manager, sample_candidate_profiles):
    jd_id = "JD_003"
    vector_store_manager.add_candidates(sample_candidate_profiles, jd_id)
    assert jd_id in vector_store_manager._faiss_indexes

    vector_store_manager.remove_index(jd_id)
    assert jd_id not in vector_store_manager._faiss_indexes
    assert jd_id not in vector_store_manager._candidate_metadata

def test_get_embedding_cache_path(vector_store_manager, mock_config_manager, tmp_path):
    # Test with cache enabled
    mock_config_manager.get_retrieval_config.return_value.cache_embeddings = True
    mock_config_manager.get_retrieval_config.return_value.embedding_cache_dir = tmp_path / "cache_enabled"
    cache_path = vector_store_manager._get_embedding_cache_path("test_text")
    assert cache_path.parent == tmp_path / "cache_enabled"
    assert cache_path.name.endswith(".npy")

    # Test with cache disabled
    mock_config_manager.get_retrieval_config.return_value.cache_embeddings = False
    cache_path = vector_store_manager._get_embedding_cache_path("test_text")
    assert cache_path is None


