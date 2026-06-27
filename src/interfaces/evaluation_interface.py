from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from src.models.domain_models import Explanation, RankedCandidate, RetrievalResult


class Metric(ABC):
    """Abstract base class for a single evaluation metric."""

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Calculates the metric and returns the result as a dictionary."""
        pass


class Evaluator(ABC):
    """Abstract base class for an evaluator that runs multiple metrics."""

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs a suite of evaluations and returns aggregated results."""
        pass


class ReportGenerator(ABC):
    """Abstract base class for generating evaluation reports."""

    @abstractmethod
    def generate_report(
        self, results: Dict[str, Any], output_path: Path, format: str
    ) -> Path:
        """Generates an evaluation report in the specified format."""
        pass


class BenchmarkRunner(ABC):
    """Abstract base class for running benchmarks across multiple configurations."""

    @abstractmethod
    def run_benchmark(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs a benchmark comparing multiple configurations."""
        pass
