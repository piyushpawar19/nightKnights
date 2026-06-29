from typing import List
from src.schemas.retrieval_schema import RetrievalResult
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.jd_schema import StructuredJD
from src.retrieval.dense_retrieval_agent import DenseRetrievalAgent
from src.retrieval.bm25_agent import BM25RetrievalAgent
from src.retrieval.vector_store import VectorStoreManager
from src.utils.config_manager import ConfigManager
from collections import defaultdict
import math


class RetrievalPipeline:
    def __init__(
        self,
        dense_retrieval_agent: DenseRetrievalAgent,
        bm25_retrieval_agent: BM25RetrievalAgent,
        vector_store_manager: VectorStoreManager,
        config_manager: ConfigManager,
    ) -> None:
        self.dense_retrieval_agent = dense_retrieval_agent
        self.bm25_retrieval_agent = bm25_retrieval_agent
        self.vector_store_manager = vector_store_manager
        self.config_manager = config_manager

    def build_retrieval_index(
        self, jd_id: str, candidate_profiles: List[CandidateProfile]
    ) -> None:
        self.vector_store_manager.add_candidates(jd_id, candidate_profiles)
        self.bm25_retrieval_agent.build_index(jd_id, candidate_profiles)

    def retrieve(
        self,
        jd_id: str,
        query: str,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        config = self.config_manager.get_retrieval_config()
        if top_k is None:
            top_k = config.hybrid_top_k

        dense_results = self.dense_retrieval_agent.retrieve(jd_id, query, config.dense_top_k)
        bm25_results = self.bm25_retrieval_agent.retrieve(jd_id, query, config.bm25_top_k)

        return self._hybrid_score_candidates(dense_results, bm25_results, top_k)

    def _hybrid_score_candidates(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        config = self.config_manager.get_retrieval_config()
        candidate_scores = defaultdict(lambda: {"dense_score": 0.0, "bm25_score": 0.0})

        for result in dense_results:
            candidate_scores[result.candidate_id]["dense_score"] = result.dense_score or 0.0

        for result in bm25_results:
            candidate_scores[result.candidate_id]["bm25_score"] = result.bm25_score or 0.0

        hybrid_results = []
        for candidate_id, scores in candidate_scores.items():
            hybrid_score = (
                (scores["dense_score"] * config.dense_weight)
                + (scores["bm25_score"] * config.bm25_weight)
            )
            # Create a new RetrievalResult with the hybrid score
            # For simplicity, we'll take the rank from the higher scoring component or default to 1
            # The retrieval_source could also be "hybrid"
            hybrid_results.append(
                RetrievalResult(
                    candidate_id=candidate_id,
                    dense_score=scores["dense_score"],
                    bm25_score=scores["bm25_score"],
                    retrieval_source="hybrid",
                    rank=1,  # Rank will be determined after sorting
                    hybrid_score=hybrid_score,
                    query_id="test_query",
                )
            )

        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Assign ranks and apply top_k
        for i, result in enumerate(hybrid_results):
            result.rank = i + 1
        return hybrid_results[:top_k]
