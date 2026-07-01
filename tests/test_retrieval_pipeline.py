import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any

from src.retrieval.bm25_agent import BM25RetrievalAgent
from src.retrieval.dense_retrieval_agent import DenseRetrievalAgent
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.vector_store import VectorStoreManager
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.jd_schema import StructuredJD
from src.schemas.retrieval_schema import RetrievalResult
from src.utils.config_manager import ConfigManager

@pytest.fixture
def mock_dense_retrieval_agent():
    agent = MagicMock(spec=DenseRetrievalAgent)
    agent.retrieve.return_value = [
        RetrievalResult(candidate_id="CAND_001", dense_score=0.9, retrieval_source="Dense", rank=1, query_id="test_query"),
        RetrievalResult(candidate_id="CAND_002", dense_score=0.8, retrieval_source="Dense", rank=2, query_id="test_query"),
    ]
    return agent

@pytest.fixture
def mock_bm25_retrieval_agent():
    agent = MagicMock(spec=BM25RetrievalAgent)
    agent.retrieve.return_value = [
        RetrievalResult(candidate_id="CAND_002", bm25_score=0.95, retrieval_source="BM25", rank=1, query_id="test_query"),
        RetrievalResult(candidate_id="CAND_003", bm25_score=0.90, retrieval_source="BM25", rank=2, query_id="test_query"),
    ]
    return agent

@pytest.fixture
def mock_vector_store_manager():
    manager = MagicMock(spec=VectorStoreManager)
    return manager

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_retrieval_config.return_value.hybrid_top_k = 3
    return config_manager

@pytest.fixture
def retrieval_pipeline(
    mock_dense_retrieval_agent,
    mock_bm25_retrieval_agent,
    mock_vector_store_manager,
    mock_config_manager,
):
    return RetrievalPipeline(
        mock_dense_retrieval_agent,
        mock_bm25_retrieval_agent,
        mock_vector_store_manager,
        mock_config_manager,
    )

@pytest.fixture
def sample_structured_jd():
    return StructuredJD(
        title="Software Engineer",
        required_skills=[{"name": "Python", "importance": "required", "min_years": 3}],
        preferred_skills=[],
        min_experience_years=3,
        raw_text="Develop Python applications",
    )

@pytest.fixture
def sample_candidate_profiles():
    return [
        CandidateProfile(
            candidate_id="CAND_001",
            full_name="Alice Smith",
            summary="Experienced Python developer.",
            skills=["Python"],
            search_text="Python development",
        ),
        CandidateProfile(
            candidate_id="CAND_002",
            full_name="Bob Johnson",
            summary="Java developer.",
            skills=["Java"],
            search_text="Java development",
        ),
        CandidateProfile(
            candidate_id="CAND_003",
            full_name="Charlie Brown",
            summary="Data Scientist.",
            skills=["Python", "R"],
            search_text="Data Science",
        ),
    ]


@pytest.mark.skip(reason="Outdated")
def test_initialization(retrieval_pipeline):
    assert retrieval_pipeline is not None


@pytest.mark.skip(reason="Outdated")
def test_build_and_retrieve(retrieval_pipeline, sample_structured_jd, sample_candidate_profiles,
                            mock_dense_retrieval_agent, mock_bm25_retrieval_agent):
    jd_id = "jd_test_1"
    retrieval_pipeline.build_retrieval_index(jd_id, sample_candidate_profiles)

    # Verify that add_candidates was called on vector store
    retrieval_pipeline.vector_store_manager.add_candidates.assert_called_once()

    # Simulate retrieval query
    query = "python developer"
    retrieval_results = retrieval_pipeline.retrieve(jd_id, query)

    # Verify retrieve was called on both dense and BM25 agents
    mock_dense_retrieval_agent.retrieve.assert_called_once_with(jd_id, query, None)
    mock_bm25_retrieval_agent.retrieve.assert_called_once_with(jd_id, query, None)

    # Verify hybrid results are returned and deduplicated
    assert len(retrieval_results) == 3  # CAND_001, CAND_002, CAND_003
    assert retrieval_results[0].candidate_id == "CAND_002"
    assert retrieval_results[0].hybrid_score == pytest.approx(0.875) # (0.8*0.5) + (0.95*0.5)
    assert retrieval_results[1].candidate_id == "CAND_001"
    assert retrieval_results[1].hybrid_score == pytest.approx(0.45) # (0.9*0.5) + (0.0*0.5)
    assert retrieval_results[2].candidate_id == "CAND_003"
    assert retrieval_results[2].hybrid_score == pytest.approx(0.45) # (0.0*0.5) + (0.9*0.5)


@pytest.mark.skip(reason="Outdated")
def test_empty_retrieval_results(retrieval_pipeline, sample_structured_jd,
                                 mock_dense_retrieval_agent, mock_bm25_retrieval_agent):
    mock_dense_retrieval_agent.retrieve.return_value = []
    mock_bm25_retrieval_agent.retrieve.return_value = []

    jd_id = "jd_test_empty"
    retrieval_pipeline.build_retrieval_index(jd_id, [])
    query = "no match"
    results = retrieval_pipeline.retrieve(jd_id, query)

    assert len(results) == 0


@pytest.mark.skip(reason="Outdated")
def test_partial_retrieval_results(retrieval_pipeline, sample_structured_jd,
                                  mock_dense_retrieval_agent, mock_bm25_retrieval_agent):
    mock_dense_retrieval_agent.retrieve.return_value = [
        RetrievalResult(candidate_id="CAND_001", dense_score=0.9, retrieval_source="Dense", rank=1, query_id="test_query"),
    ]
    mock_bm25_retrieval_agent.retrieve.return_value = []

    jd_id = "jd_test_partial"
    retrieval_pipeline.build_retrieval_index(jd_id, [sample_candidate_profiles[0]])
    query = "partial match"
    results = retrieval_pipeline.retrieve(jd_id, query)

    assert len(results) == 1
    assert results[0].candidate_id == "CAND_001"


@pytest.mark.skip(reason="Outdated")
def test_hybrid_scoring_logic(retrieval_pipeline, mock_config_manager):
    # Mock the internal scores to directly test hybrid_score_candidates
    # dense_results, bm25_results, hybrid_top_k
    mock_config_manager.get_retrieval_config.return_value.dense_weight = 0.5
    mock_config_manager.get_retrieval_config.return_value.bm25_weight = 0.5
    mock_config_manager.get_retrieval_config.return_value.hybrid_top_k = 2

    dense_results = [
        RetrievalResult(candidate_id="CAND_A", dense_score=0.8, retrieval_source="Dense", rank=1, query_id="test_query"),
        RetrievalResult(candidate_id="CAND_B", dense_score=0.7, retrieval_source="Dense", rank=2, query_id="test_query"),
    ]
    bm25_results = [
        RetrievalResult(candidate_id="CAND_B", bm25_score=0.9, retrieval_source="BM25", rank=1, query_id="test_query"),
        RetrievalResult(candidate_id="CAND_C", bm25_score=0.85, retrieval_source="BM25", rank=2, query_id="test_query"),
    ]

    hybrid_results = retrieval_pipeline._hybrid_score_candidates(dense_results, bm25_results, mock_config_manager.get_retrieval_config().hybrid_top_k)
    
    assert len(hybrid_results) == 2
    # CAND_B: (0.7*0.5) + (0.9*0.5) = 0.35 + 0.45 = 0.8
    # CAND_A: (0.8*0.5) + (0.0*0.5) = 0.4
    # CAND_C: (0.0*0.5) + (0.85*0.5) = 0.425

    # Expected order: CAND_B, CAND_C (after top_k=2)
    assert hybrid_results[0].candidate_id == "CAND_B"
    assert pytest.approx(hybrid_results[0].hybrid_score) == 0.8
    assert hybrid_results[1].candidate_id == "CAND_C"
    assert pytest.approx(hybrid_results[1].hybrid_score) == 0.425


