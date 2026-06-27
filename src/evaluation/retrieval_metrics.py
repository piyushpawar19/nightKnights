from typing import Any, Dict, List, Set

from src.interfaces.evaluation_interface import Metric
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalMetrics:
    """Calculates retrieval-focused metrics like Recall@K, Precision@K, Hit Rate, and Coverage."""

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str], relevant_ids: Set[str], k: int
    ) -> float:
        """Calculates Recall@K.

        Args:
            retrieved_ids (List[str]): Ordered list of retrieved candidate IDs.
            relevant_ids (Set[str]): Set of truly relevant candidate IDs.
            k (int): The cutoff point for evaluation.

        Returns:
            float: The Recall@K score.
        """
        if not relevant_ids or k <= 0:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        hits = len(retrieved_at_k.intersection(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str], relevant_ids: Set[str], k: int
    ) -> float:
        """Calculates Precision@K.

        Args:
            retrieved_ids (List[str]): Ordered list of retrieved candidate IDs.
            relevant_ids (Set[str]): Set of truly relevant candidate IDs.
            k (int): The cutoff point for evaluation.

        Returns:
            float: The Precision@K score.
        """
        if k <= 0 or not retrieved_ids:
            return 0.0

        retrieved_at_k = set(retrieved_ids[:k])
        hits = len(retrieved_at_k.intersection(relevant_ids))
        return hits / k

    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculates Hit Rate (whether any relevant item is retrieved).

        Args:
            retrieved_ids (List[str]): Ordered list of retrieved candidate IDs.
            relevant_ids (Set[str]): Set of truly relevant candidate IDs.

        Returns:
            float: 1.0 if any relevant item is in retrieved_ids, 0.0 otherwise.
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0

        if set(retrieved_ids).intersection(relevant_ids):
            return 1.0
        return 0.0

    @staticmethod
    def coverage(
        retrieved_ids_all_queries: List[Set[str]], all_possible_relevant_ids: Set[str]
    ) -> float:
        """Calculates Coverage (proportion of all relevant items that are ever retrieved).

        Args:
            retrieved_ids_all_queries (List[Set[str]]): List of sets of retrieved IDs for each query.
            all_possible_relevant_ids (Set[str]): Set of all unique relevant candidate IDs across all queries.

        Returns:
            float: The Coverage score.
        """
        if not all_possible_relevant_ids:
            return 0.0

        unique_retrieved_ids = set()
        for retrieved_set in retrieved_ids_all_queries:
            unique_retrieved_ids.update(retrieved_set)

        covered_relevant_ids = unique_retrieved_ids.intersection(all_possible_relevant_ids)
        return len(covered_relevant_ids) / len(all_possible_relevant_ids)


class RecallMetric(Metric):
    def calculate(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> Dict[str, Any]:
        score = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        logger.debug("Calculated Recall@%d: %.4f", k, score)
        return {"recall_at_k": score, "k": k}


class PrecisionMetric(Metric):
    def calculate(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> Dict[str, Any]:
        score = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        logger.debug("Calculated Precision@%d: %.4f", k, score)
        return {"precision_at_k": score, "k": k}


class HitRateMetric(Metric):
    def calculate(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> Dict[str, Any]:
        score = RetrievalMetrics.hit_rate(retrieved_ids, relevant_ids)
        logger.debug("Calculated Hit Rate: %.4f", score)
        return {"hit_rate": score}


class CoverageMetric(Metric):
    def calculate(
        self, retrieved_ids_all_queries: List[Set[str]], all_possible_relevant_ids: Set[str]
    ) -> Dict[str, Any]:
        score = RetrievalMetrics.coverage(retrieved_ids_all_queries, all_possible_relevant_ids)
        logger.debug("Calculated Coverage: %.4f", score)
        return {"coverage": score}
