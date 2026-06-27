from typing import Any, Dict, List, Optional

from src.interfaces.evaluation_interface import Metric
from src.models.domain_models import Explanation, RecruiterAssessment
from src.schemas.graph_schema import NodeTimestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityMetrics:
    """Calculates metrics related to the explanation service performance."""

    @staticmethod
    def explanation_success_rate(explanations: List[Explanation]) -> float:
        """Calculates the rate of successfully generated explanations."""
        if not explanations:
            return 0.0
        successful_explanations = [e for e in explanations if e.summary and e.reasoning]
        rate = len(successful_explanations) / len(explanations)
        logger.debug("Explanation success rate: %.4f", rate)
        return rate

    @staticmethod
    def fallback_rate(explanations: List[Explanation]) -> float:
        """Calculates the rate at which fallback explanations were used."""
        if not explanations:
            return 0.0
        # Assuming a 'fallback_used' flag or similar in Explanation metadata
        # For now, we'll assume explanations with generic/short summaries might indicate fallback
        # In a real system, this would be explicitly tracked.
        fallback_explanations = [
            e for e in explanations
            if "fallback" in e.metadata.data.get("source", "").lower()
        ]
        rate = len(fallback_explanations) / len(explanations)
        logger.debug("Fallback rate: %.4f", rate)
        return rate

    @staticmethod
    def explanation_confidence_distribution(explanations: List[Explanation]) -> Dict[str, Any]:
        """Provides statistics on the distribution of explanation confidence scores."""
        if not explanations:
            return {"min": None, "max": None, "mean": None, "std": None, "count": 0}

        confidences = [e.confidence for e in explanations if e.confidence is not None]
        if not confidences:
            return {"min": None, "max": None, "mean": None, "std": None, "count": 0}

        stats = {
            "min": min(confidences),
            "max": max(confidences),
            "mean": sum(confidences) / len(confidences),
            "count": len(confidences),
        }
        # std dev calculation requires numpy or a more involved manual calculation
        # For simplicity, we'll skip std here or assume it's calculated elsewhere
        logger.debug("Explanation confidence distribution: %s", stats)
        return stats


class ExplanationSuccessRateMetric(Metric):
    def calculate(self, explanations: List[Explanation]) -> Dict[str, Any]:
        score = ExplainabilityMetrics.explanation_success_rate(explanations)
        return {"explanation_success_rate": score}


class FallbackRateMetric(Metric):
    def calculate(self, explanations: List[Explanation]) -> Dict[str, Any]:
        score = ExplainabilityMetrics.fallback_rate(explanations)
        return {"fallback_rate": score}


class ExplanationConfidenceDistributionMetric(Metric):
    def calculate(self, explanations: List[Explanation]) -> Dict[str, Any]:
        stats = ExplainabilityMetrics.explanation_confidence_distribution(explanations)
        return {"explanation_confidence_distribution": stats}
