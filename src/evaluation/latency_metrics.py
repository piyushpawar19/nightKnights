import time
from typing import Any, Dict, List, Optional

from src.interfaces.evaluation_interface import Metric
from src.schemas.graph_schema import NodeTimestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LatencyMetrics:
    """Calculates latency and throughput metrics for pipeline components."""

    @staticmethod
    def calculate_node_latency(timestamps: List[NodeTimestamp], node_name: str) -> Optional[float]:
        """Calculates the latency for a specific node in milliseconds."""
        for ts in timestamps:
            if ts.node_name == node_name:
                logger.debug("Calculated latency for node %s: %.2fms", node_name, ts.duration_ms)
                return ts.duration_ms
        logger.warning("Node %s timestamps not found.", node_name)
        return None

    @staticmethod
    def calculate_total_pipeline_runtime(timestamps: List[NodeTimestamp]) -> Optional[float]:
        """Calculates the total pipeline runtime from start of first node to end of last node."""
        if not timestamps:
            return None

        start_time = min(ts.started_at for ts in timestamps)
        end_time = max(ts.completed_at for ts in timestamps)
        duration_ms = round((end_time - start_time).total_seconds() * 1000, 2)
        logger.debug("Calculated total pipeline runtime: %.2fms", duration_ms)
        return duration_ms

    @staticmethod
    def calculate_throughput(total_runtime_ms: float, num_candidates: int) -> float:
        """Calculates throughput in candidates per second."""
        if total_runtime_ms <= 0 or num_candidates <= 0:
            return 0.0
        throughput = num_candidates / (total_runtime_ms / 1000)
        logger.debug("Calculated throughput: %.2f candidates/sec", throughput)
        return throughput


class RetrievalLatencyMetric(Metric):
    def calculate(self, timestamps: List[NodeTimestamp]) -> Dict[str, Any]:
        latency = LatencyMetrics.calculate_node_latency(timestamps, "retrieval") # Assuming 'retrieval' is the node name
        return {"retrieval_latency_ms": latency} if latency is not None else {}


class RankingLatencyMetric(Metric):
    def calculate(self, timestamps: List[NodeTimestamp]) -> Dict[str, Any]:
        latency = LatencyMetrics.calculate_node_latency(timestamps, "ranking") # Assuming 'ranking' is the node name
        return {"ranking_latency_ms": latency} if latency is not None else {}


class ExplanationLatencyMetric(Metric):
    def calculate(self, timestamps: List[NodeTimestamp]) -> Dict[str, Any]:
        latency = LatencyMetrics.calculate_node_latency(timestamps, "explanation") # Assuming 'explanation' is the node name
        return {"explanation_latency_ms": latency} if latency is not None else {}


class ExportLatencyMetric(Metric):
    def calculate(self, timestamps: List[NodeTimestamp]) -> Dict[str, Any]:
        latency = LatencyMetrics.calculate_node_latency(timestamps, "export") # Assuming 'export' is the node name
        return {"export_latency_ms": latency} if latency is not None else {}


class TotalPipelineRuntimeMetric(Metric):
    def calculate(self, timestamps: List[NodeTimestamp]) -> Dict[str, Any]:
        runtime = LatencyMetrics.calculate_total_pipeline_runtime(timestamps)
        return {"total_pipeline_runtime_ms": runtime} if runtime is not None else {}


class ThroughputMetric(Metric):
    def calculate(self, total_runtime_ms: float, num_candidates: int) -> Dict[str, Any]:
        throughput = LatencyMetrics.calculate_throughput(total_runtime_ms, num_candidates)
        return {"throughput_candidates_per_sec": throughput}
