import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml
from src.evaluation.evaluation_agent import EvaluationAgent
from src.interfaces.evaluation_interface import BenchmarkRunner as AbstractBenchmarkRunner
from src.models.domain_models import Explanation, RankedCandidate
from src.schemas.graph_schema import NodeStatus, NodeTimestamp
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRunner(AbstractBenchmarkRunner):
    """Runs benchmarks across different pipeline configurations and collects results."""

    def __init__(
        self,
        config_manager: ConfigManager,
        evaluation_agent: EvaluationAgent,
        base_output_dir: Optional[Path] = None,
    ) -> None:
        self.config_manager = config_manager
        self.evaluation_agent = evaluation_agent
        self.base_output_dir = base_output_dir or Path("outputs/benchmarks")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        pipeline_callable: Callable[..., Any],
        benchmark_configs: Dict[str, Dict[str, Any]],
        ground_truth_data: Dict[str, Set[str]],
        initial_candidates_count: int,
        k_values: Dict[str, int],
        report_format: str = "json",
    ) -> Dict[str, Any]:
        logger.info("Starting benchmark run with %d configurations.", len(benchmark_configs))
        benchmark_results: Dict[str, Any] = {
            "benchmark_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "individual_runs": {},
            "summary": "",
        }

        for bench_name, config_overrides in benchmark_configs.items():
            logger.info("Running benchmark for configuration: %s", bench_name)
            run_output_dir = self.base_output_dir / bench_name
            run_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                pipeline_result = pipeline_callable(config_overrides)
                if isinstance(pipeline_result, tuple) and len(pipeline_result) == 4:
                    ranked_candidates, retrieval_results_per_query, explanations, timestamps = pipeline_result
                else:
                    raise ValueError(
                        "pipeline_callable must return "
                        "(ranked_candidates, retrieval_results_per_query, explanations, timestamps)"
                    )

                evaluation_results = self.evaluation_agent.evaluate(
                    ranked_candidates=ranked_candidates,
                    ground_truth_relevant_ids=ground_truth_data,
                    retrieval_results_per_query=retrieval_results_per_query,
                    explanations=explanations,
                    timestamps=timestamps,
                    num_initial_candidates=initial_candidates_count,
                    k_values=k_values,
                    report_name=f"{bench_name}_evaluation",
                    report_format=report_format,
                    output_dir=run_output_dir,
                    summary=f"Evaluation for benchmark: {bench_name}",
                )
                benchmark_results["individual_runs"][bench_name] = evaluation_results
            except Exception as e:
                logger.error("Error running benchmark for %s: %s", bench_name, e)
                benchmark_results["individual_runs"][bench_name] = {"error": str(e)}

        logger.info("Benchmark run completed.")
        return benchmark_results

    def _apply_config_overrides(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        config = base_config.copy()
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key] = self._apply_config_overrides(config[key], value)
            else:
                config[key] = value
        return config
