import math
from typing import Any, Dict, List, Set

from src.interfaces.evaluation_interface import Metric
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RankingMetrics:
    """Calculates ranking-focused metrics like MRR, NDCG, MAP, Precision@K, Recall@K, and Top-K Accuracy."""

    @staticmethod
    def mrr(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculates Mean Reciprocal Rank (MRR)."""
        for i, doc_id in enumerate(ranked_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Calculates Normalized Discounted Cumulative Gain (NDCG) at K."""
        if k <= 0:
            return 0.0

        idcg = RankingMetrics._ideal_dcg(relevant_ids, k)
        if idcg == 0:
            return 0.0

        dcg = 0.0
        for i, doc_id in enumerate(ranked_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)
        return dcg / idcg

    @staticmethod
    def _ideal_dcg(relevant_ids: Set[str], k: int) -> float:
        """Helper to calculate Ideal Discounted Cumulative Gain (IDCG)."""
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / math.log2(i + 2)
        return idcg

    @staticmethod
    def average_precision(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculates Average Precision (AP)."""
        if not relevant_ids or not ranked_ids:
            return 0.0

        num_relevant_retrieved = 0
        sum_precisions = 0.0
        for i, doc_id in enumerate(ranked_ids):
            if doc_id in relevant_ids:
                num_relevant_retrieved += 1
                sum_precisions += num_relevant_retrieved / (i + 1)
        return sum_precisions / len(relevant_ids)

    @staticmethod
    def precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Calculates Precision@K (re-implementation for ranking context)."""
        if k <= 0 or not ranked_ids:
            return 0.0

        hits = 0
        for doc_id in ranked_ids[:k]:
            if doc_id in relevant_ids:
                hits += 1
        return hits / k

    @staticmethod
    def recall_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Calculates Recall@K (re-implementation for ranking context)."""
        if not relevant_ids or k <= 0:
            return 0.0

        hits = 0
        for doc_id in ranked_ids[:k]:
            if doc_id in relevant_ids:
                hits += 1
        return hits / len(relevant_ids)

    @staticmethod
    def top_k_accuracy(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Calculates Top-K Accuracy.

        This metric checks if *any* of the relevant items are within the top K.
        It's equivalent to Hit Rate if k is the length of retrieved_ids.
        """
        if k <= 0 or not ranked_ids or not relevant_ids:
            return 0.0

        for doc_id in ranked_ids[:k]:
            if doc_id in relevant_ids:
                return 1.0
        return 0.0


class MRRMetric(Metric):
    def calculate(self, ranked_ids: List[str], relevant_ids: Set[str]) -> Dict[str, Any]:
        score = RankingMetrics.mrr(ranked_ids, relevant_ids)
        logger.debug("Calculated MRR: %.4f", score)
        return {"mrr": score}


class NDCGMetric(Metric):
    def calculate(self, ranked_ids: List[str], relevant_ids: Set[str], k: int) -> Dict[str, Any]:
        score = RankingMetrics.ndcg_at_k(ranked_ids, relevant_ids, k)
        logger.debug("Calculated NDCG@%d: %.4f", k, score)
        return {"ndcg_at_k": score, "k": k}


class MAPMetric(Metric):
    def calculate(self, ranked_ids: List[str], relevant_ids: Set[str]) -> Dict[str, Any]:
        score = RankingMetrics.average_precision(ranked_ids, relevant_ids)
        logger.debug("Calculated MAP: %.4f", score)
        return {"map": score}


class RankingPrecisionMetric(Metric):
    def calculate(self, ranked_ids: List[str], relevant_ids: Set[str], k: int) -> Dict[str, Any]:
        score = RankingMetrics.precision_at_k(ranked_ids, relevant_ids, k)
        logger.debug("Calculated Ranking Precision@%d: %.4f", k, score)
        return {"ranking_precision_at_k": score, "k": k}


class RankingRecallMetric(Metric):
    def calculate(self, ranked_ids: List[str], relevant_ids: Set[str], k: int) -> Dict[str, Any]:
        score = RankingMetrics.recall_at_k(ranked_ids, relevant_ids, k)
        logger.debug("Calculated Ranking Recall@%d: %.4f", k, score)
        return {"ranking_recall_at_k": score, "k": k}


class TopKAccuracyMetric(Metric):
    def calculate(self, ranked_ids: List[str], relevant_ids: Set[str], k: int) -> Dict[str, Any]:
        score = RankingMetrics.top_k_accuracy(ranked_ids, relevant_ids, k)
        logger.debug("Calculated Top-K Accuracy@%d: %.4f", k, score)
        return {"top_k_accuracy": score, "k": k}
