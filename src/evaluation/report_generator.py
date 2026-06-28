import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from src.interfaces.evaluation_interface import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReportGenerator(ReportGenerator):
    """Generates evaluation reports in various formats (JSON, CSV, Markdown)."""

    def generate_report(
        self, results: Dict[str, Any], output_path: Path, format: str
    ) -> Path:
        """Generates an evaluation report in the specified format.

        Args:
            results (Dict[str, Any]): The aggregated evaluation results.
            output_path (Path): The base path and filename for the report.
            format (str): The desired report format (json, csv, md).

        Returns:
            Path: The path to the generated report file.

        Raises:
            ValueError: If an unsupported format is requested.
            IOError: If there is an error writing the file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_filepath = output_path.parent / f"{output_path.stem}.{format}"

        try:
            if format == "json":
                self._write_json_report(report_filepath, results)
            elif format == "csv":
                self._write_csv_report(report_filepath, results)
            elif format == "md":
                self._write_markdown_report(report_filepath, results)
            else:
                raise ValueError(f"Unsupported report format: {format}")

            logger.info("Evaluation report generated successfully at %s", report_filepath)
            return report_filepath
        except IOError as e:
            logger.error("Error writing report to %s: %s", report_filepath, e)
            raise
        except Exception as e:
            logger.error(
                "An unexpected error occurred during report generation: %s", e
            )
            raise

    def _write_json_report(self, filepath: Path, results: Dict[str, Any]) -> None:
        """Writes the report in JSON format."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def _write_csv_report(self, filepath: Path, results: Dict[str, Any]) -> None:
        """Writes the report in CSV format (flattening results as needed)."""
        # This is a simplified flattening. For complex nested results, a more robust flattening
        # logic would be needed, possibly converting nested dicts to dot-notation keys.
        headers = []
        rows = []

        # Extract headers and flatten values for CSV
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    headers.append(f"{key}_{sub_key}")
                    # Ensure sub_value is string or simple type for CSV
                    rows.append(str(sub_value) if not isinstance(sub_value, (list, dict)) else json.dumps(sub_value))
            else:
                headers.append(key)
                rows.append(str(value) if not isinstance(value, (list, dict)) else json.dumps(value))
        
        # For now, we'll assume a single row for aggregated results.
        # If individual pipeline runs are part of the results, this needs to be adapted.
        data_to_write = [dict(zip(headers, rows))]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_to_write)

    def _write_markdown_report(self, filepath: Path, results: Dict[str, Any]) -> None:
        """Writes the report in Markdown format."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"**Generated At:** {results.get("timestamp", "N/A")}\n")
            f.write(f"**Pipeline Version:** {results.get("pipeline_version", "N/A")}\n")
            f.write(f"**Configuration Snapshot:**\n")
            f.write("```yaml\n")
            f.write(yaml.dump(results.get("configuration_snapshot", {}), indent=2))
            f.write("```\n\n")

            f.write("## Summary\n\n")
            f.write(f"{results.get("summary", "No summary provided.")}\n\n")

            f.write("## Metrics\n\n")
            for category, metrics in results.items():
                if isinstance(metrics, dict) and category not in ["configuration_snapshot", "summary", "timestamp", "pipeline_version"]:
                    f.write(f"### {category.replace("_", " ").title()}\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("| :----- | :---- |\n")
                    for metric_name, value in metrics.items():
                        f.write(f"| {metric_name.replace("_", " ").title()} | {value} |\n")
                    f.write("\n")