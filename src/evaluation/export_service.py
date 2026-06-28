import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.evaluation.csv_generator import CSVGenerator
from src.evaluation.serializers import SubmissionSerializer
from src.evaluation.validators import ExportValidationError, SubmissionValidator
from src.interfaces.export_interface import ExportService as AbstractExportService
from src.models.domain_models import Explanation, RankedCandidate, RecruiterAssessment
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExportService(AbstractExportService):
    """Service for exporting validated ranking results to various formats."""

    def __init__(
        self,
        validator: SubmissionValidator,
        serializer: SubmissionSerializer,
        csv_writer: CSVGenerator,
        config_manager: ConfigManager,
    ) -> None:
        self.validator = validator
        self.serializer = serializer
        self.csv_writer = csv_writer
        self.config_manager = config_manager

    def _generate_metadata(
        self,
        start_time: float,
        output_path: Path,
        record_count: int,
        config_version: str = "1.0", # Assuming a default or retrieve from a more specific config
    ) -> Dict[str, Any]:
        """Generates export metadata."""
        end_time = time.perf_counter()
        duration_ms = round((end_time - start_time) * 1000, 2)
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        # In a real system, pipeline_version would likely come from a build system or config
        # and config_version from a specific export-related config model.
        return {
            "timestamp": timestamp,
            "pipeline_version": "1.0.0",  # Placeholder
            "total_candidates": record_count,
            "export_duration_ms": duration_ms,
            "configuration_version": config_version,
            "output_path": str(output_path),
        }

    def _save_metadata(self, output_path: Path, metadata: Dict[str, Any]) -> None:
        """Saves export metadata to a YAML file alongside the export."""
        metadata_filepath = output_path.parent / f"{output_path.stem}_metadata.yaml"
        try:
            with open(metadata_filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(metadata, f, indent=4)
            logger.info("Export metadata saved to %s", metadata_filepath)
        except IOError as e:
            logger.error(
                "Failed to save export metadata to %s: %s", metadata_filepath, e
            )
            raise

    def export_csv(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        output_dir: Path,
        filename: str,
        export_schema: List[str],
        delimiter: str = ",",
        quotechar: str = ",",
        quoting: int = 0,  # csv.QUOTE_MINIMAL
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> Path:
        """Exports validated ranking results to a CSV file."""
        start_time = time.perf_counter()
        output_path = output_dir / filename
        record_count = 0

        logger.info("Exporting data to CSV: %s", output_path)

        try:
            serialized_data = self.serializer.serialize(
                ranked_candidates,
                recruiter_assessments,
                explanations,
                export_schema,
            )
            self.validator.validate(serialized_data, export_schema)

            record_count = len(serialized_data)

            self.csv_writer.write_csv(
                filepath=output_path,
                data=serialized_data,
                schema=export_schema,
                delimiter=delimiter,
                quotechar=quotechar,
                quoting=quoting,
                encoding=encoding,
                overwrite=overwrite,
            )

            metadata = self._generate_metadata(start_time, output_path, record_count)
            self._save_metadata(output_path, metadata)

            logger.info(
                "CSV export completed successfully. Records: %d, File: %s, Duration: %.2fms",
                record_count,
                output_path,
                metadata["export_duration_ms"],
            )
            return output_path
        except ExportValidationError as e:
            logger.error("CSV Export validation failed: %s", e)
            raise
        except FileExistsError as e:
            logger.error("CSV Export failed: %s", e)
            raise
        except Exception as e:
            logger.error("An unexpected error occurred during CSV export: %s", e)
            raise

    def export_json(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        output_dir: Path,
        filename: str,
        overwrite: bool = False,
    ) -> Path:
        """Exports validated ranking results to a JSON file."""
        output_path = output_dir / filename
        logger.info("JSON export requested for %s (not yet fully implemented).", output_path)
        # Placeholder for future implementation
        raise NotImplementedError("JSON export is not yet implemented.")

    def export_parquet(
        self,
        ranked_candidates: List[RankedCandidate],
        recruiter_assessments: List[RecruiterAssessment],
        explanations: List[Explanation],
        output_dir: Path,
        filename: str,
        overwrite: bool = False,
    ) -> Path:
        """Exports validated ranking results to a Parquet file."""
        output_path = output_dir / filename
        logger.info("Parquet export requested for %s (not yet fully implemented).", output_path)
        # Placeholder for future implementation
        raise NotImplementedError("Parquet export is not yet implemented.")