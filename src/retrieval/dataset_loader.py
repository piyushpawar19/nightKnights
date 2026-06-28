"""
Dataset Loader Agent — entry point of the Retrieval Engine.

Loads candidate datasets from multiple formats, validates records, and exposes
lazy, batch-oriented iteration for downstream retrieval agents.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]

# Lightweight candidate ID pattern aligned with candidate_schema.json
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetLoaderError(Exception):
    """Base exception for Dataset Loader errors."""


class DatasetNotFoundError(DatasetLoaderError):
    """Raised when the source dataset file does not exist."""


class UnsupportedFormatError(DatasetLoaderError):
    """Raised when the dataset format cannot be handled."""


class EmptyDatasetError(DatasetLoaderError):
    """Raised when the dataset contains no valid records after loading."""


class BatchIndexError(DatasetLoaderError):
    """Raised when a requested batch index is out of range."""


class RecordNotFoundError(DatasetLoaderError):
    """Raised when a candidate ID cannot be found in the materialized dataset."""


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _load_retrieval_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load dataset-loader defaults from ``configs/retrieval.yaml`` if present."""
    path = config_path or (_project_root() / "configs" / "retrieval.yaml")
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read retrieval config at %s: %s", path, exc)
        return {}


def _resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """Resolve *path* relative to *base* (defaults to project root)."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base or _project_root()) / candidate


# ---------------------------------------------------------------------------
# Format handlers (extensible for future SQL / S3 / HuggingFace sources)
# ---------------------------------------------------------------------------


class _FormatHandler(ABC):
    """Abstract reader for a single on-disk dataset format."""

    @abstractmethod
    def iter_records(self, path: Path) -> Iterator[CandidateRecord]:
        """Yield raw records from *path* using streaming I/O where possible."""


class _CsvHandler(_FormatHandler):
    def iter_records(self, path: Path) -> Iterator[CandidateRecord]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                if row is None:
                    continue
                try:
                    yield _parse_csv_row(row)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Skipping corrupted CSV row %d in %s: %s", row_number, path, exc)


class _JsonHandler(_FormatHandler):
    def iter_records(self, path: Path) -> Iterator[CandidateRecord]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    yield item
                else:
                    logger.warning("Skipping non-dict JSON array item at index %d in %s", index, path)
            return

        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                for index, item in enumerate(payload["records"]):
                    if isinstance(item, dict):
                        yield item
                    else:
                        logger.warning(
                            "Skipping non-dict record at index %d in %s", index, path
                        )
                return
            yield payload
            return

        raise UnsupportedFormatError(
            f"JSON file {path} must be an array of objects or a single object."
        )


class _JsonlHandler(_FormatHandler):
    def iter_records(self, path: Path) -> Iterator[CandidateRecord]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping invalid JSON on line %d in %s: %s", line_number, path, exc
                    )
                    continue
                if isinstance(record, dict):
                    yield record
                else:
                    logger.warning(
                        "Skipping non-dict JSONL record on line %d in %s", line_number, path
                    )


class _ParquetHandler(_FormatHandler):
    def iter_records(self, path: Path) -> Iterator[CandidateRecord]:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]
        except ImportError as exc:
            raise UnsupportedFormatError(
                "Parquet support requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    yield row
                else:
                    logger.warning("Skipping non-dict Parquet row in %s", path)


_FORMAT_HANDLERS: dict[str, _FormatHandler] = {
    "csv": _CsvHandler(),
    "json": _JsonHandler(),
    "jsonl": _JsonlHandler(),
    "parquet": _ParquetHandler(),
}


def _parse_csv_row(row: dict[str, str]) -> CandidateRecord:
    """Convert a CSV row to a candidate record, parsing nested JSON columns."""
    record: CandidateRecord = {}
    for key, value in row.items():
        if value is None or value == "":
            record[key] = None
            continue
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                record[key] = json.loads(stripped)
            except json.JSONDecodeError:
                record[key] = value
        else:
            record[key] = value
    return record


def _detect_format(path: Path) -> str:
    """Detect dataset format from file extension."""
    ext = path.suffix.lower()
    fmt = SUPPORTED_EXTENSIONS.get(ext)
    if fmt is None:
        raise UnsupportedFormatError(
            f"Unsupported dataset format '{ext}' for file {path}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return fmt


def _source_fingerprint(path: Path) -> str:
    """Fingerprint a source file for cache invalidation."""
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


class DatasetLoader:
    """
    Production-ready loader for candidate datasets.

    Responsibilities:
        - Detect and read CSV, JSON, JSONL, and Parquet (when pyarrow is available)
        - Validate and normalize records without stopping the pipeline
        - Expose lazy batch iteration for large (~100k) datasets
        - Optionally cache processed records under ``data/cache/``

    This agent does **not** build profiles, generate embeddings, or perform retrieval.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        batch_size: int = 256,
        cache_enabled: bool = True,
        validate_records: bool = True,
        *,
        cache_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        strict_id_format: bool = False,
    ) -> None:
        config = _load_retrieval_config(
            Path(config_path) if config_path is not None else None
        )
        loader_cfg = config.get("dataset_loader", {}) if isinstance(config, dict) else {}

        resolved_path = dataset_path
        if resolved_path in ("", None) and loader_cfg.get("default_dataset_path"):
            resolved_path = loader_cfg["default_dataset_path"]
        self.dataset_path = _resolve_path(resolved_path)

        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.validate_records = validate_records
        self.strict_id_format = strict_id_format

        cache_default = _project_root() / "data" / "cache"
        configured_cache = loader_cfg.get("cache_dir", cache_dir or cache_default)
        self.cache_dir = _resolve_path(configured_cache)

        self._format: Optional[str] = None
        self._records: Optional[list[CandidateRecord]] = None
        self._id_index: Optional[dict[str, int]] = None
        self._stats: dict[str, Any] = {
            "total_raw": 0,
            "valid": 0,
            "invalid": 0,
            "duplicates": 0,
            "load_time_seconds": 0.0,
        }

        if not self.dataset_path.is_file():
            raise DatasetNotFoundError(f"Dataset file not found: {self.dataset_path}")

        self._format = _detect_format(self.dataset_path)
        logger.info(
            "DatasetLoader initialized for %s (format=%s, batch_size=%d, cache=%s)",
            self.dataset_path,
            self._format,
            self.batch_size,
            self.cache_enabled,
        )

    @classmethod
    def from_config(
        cls,
        dataset_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> "DatasetLoader":
        """
        Construct a loader using defaults from ``configs/retrieval.yaml``.

        Explicit *dataset_path* overrides ``dataset_loader.default_dataset_path``.
        """
        config = _load_retrieval_config(
            Path(config_path) if config_path is not None else None
        )
        loader_cfg = config.get("dataset_loader", {}) if isinstance(config, dict) else {}
        path = dataset_path or loader_cfg.get("default_dataset_path", "candidates.jsonl")
        return cls(
            dataset_path=path,
            batch_size=int(loader_cfg.get("batch_size", 256)),
            cache_enabled=bool(loader_cfg.get("cache_enabled", True)),
            validate_records=bool(loader_cfg.get("validate_records", True)),
            cache_dir=loader_cfg.get("cache_dir"),
            config_path=config_path,
            strict_id_format=bool(loader_cfg.get("strict_id_format", False)),
        )

    # ------------------------------------------------------------------
    # Public API — loading
    # ------------------------------------------------------------------

    def load(self) -> list[CandidateRecord]:
        """
        Load and materialize the full validated dataset.

        Uses cache when enabled and the source fingerprint matches.

        Returns:
            List of validated candidate record dictionaries.

        Raises:
            EmptyDatasetError: If no valid records remain after validation.
        """
        start = time.perf_counter()
        if self._records is not None:
            logger.debug("Dataset already materialized in memory (%d records).", len(self._records))
            return self._records

        if self.cache_enabled and self._try_load_cache():
            elapsed = time.perf_counter() - start
            self._stats["load_time_seconds"] = elapsed
            logger.info(
                "Dataset loaded from cache: %d records in %.2fs",
                len(self._records or []),
                elapsed,
            )
            return self._records or []

        logger.info("Loading dataset from source: %s", self.dataset_path)
        records = self._materialize_from_source()
        elapsed = time.perf_counter() - start
        self._stats["load_time_seconds"] = elapsed

        if not records:
            raise EmptyDatasetError(f"No valid records found in {self.dataset_path}")

        if self.cache_enabled:
            self.cache_dataset()

        logger.info(
            "Dataset loaded: %d valid records (%d invalid, %d duplicates skipped) in %.2fs",
            self._stats["valid"],
            self._stats["invalid"],
            self._stats["duplicates"],
            elapsed,
        )
        return records

    def load_csv(self) -> list[CandidateRecord]:
        """Load dataset assuming CSV format."""
        self._format = "csv"
        return self._load_with_format("csv")

    def load_json(self) -> list[CandidateRecord]:
        """Load dataset assuming JSON format."""
        self._format = "json"
        return self._load_with_format("json")

    def load_jsonl(self) -> list[CandidateRecord]:
        """Load dataset assuming JSON Lines format."""
        self._format = "jsonl"
        return self._load_with_format("jsonl")

    def load_parquet(self) -> list[CandidateRecord]:
        """Load dataset assuming Parquet format (requires pyarrow)."""
        self._format = "parquet"
        return self._load_with_format("parquet")

    def _load_with_format(self, fmt: str) -> list[CandidateRecord]:
        previous = self._format
        self._format = fmt
        self._reset_materialized()
        try:
            return self.load()
        finally:
            self._format = previous

    # ------------------------------------------------------------------
    # Public API — iteration & access
    # ------------------------------------------------------------------

    def iter_batches(self) -> Iterator[list[CandidateRecord]]:
        """
        Yield validated records in fixed-size batches.

        Streams from the source when the dataset is not yet materialized,
        keeping memory usage bounded for large files (~100k records).
        """
        if self._records is not None:
            for batch_number in range(0, len(self._records), self.batch_size):
                yield self._records[batch_number : batch_number + self.batch_size]
            return

        batch: list[CandidateRecord] = []
        seen_ids: set[str] = set()
        batch_index = 0

        for raw in self._iter_raw_records():
            self._stats["total_raw"] += 1
            record = self._process_record(raw, seen_ids)
            if record is None:
                continue
            batch.append(record)
            if len(batch) >= self.batch_size:
                batch_index += 1
                logger.debug("Yielding batch %d (%d records)", batch_index, len(batch))
                yield batch
                batch = []

        if batch:
            batch_index += 1
            logger.debug("Yielding final batch %d (%d records)", batch_index, len(batch))
            yield batch

    def get_batch(self, batch_number: int) -> list[CandidateRecord]:
        """
        Return a specific batch by zero-based index.

        Args:
            batch_number: Zero-based batch index.

        Raises:
            BatchIndexError: If the batch index is out of range.
        """
        if batch_number < 0:
            raise BatchIndexError(f"Batch index must be non-negative, got {batch_number}")

        self._ensure_materialized()
        assert self._records is not None

        start = batch_number * self.batch_size
        if start >= len(self._records):
            max_batch = max(0, (len(self._records) - 1) // self.batch_size)
            raise BatchIndexError(
                f"Batch index {batch_number} out of range (max batch index: {max_batch})."
            )
        return self._records[start : start + self.batch_size]

    def get_record(self, candidate_id: str) -> CandidateRecord:
        """
        Retrieve a single candidate record by ID.

        Raises:
            RecordNotFoundError: If the candidate ID is absent.
        """
        self._ensure_materialized()
        assert self._id_index is not None
        assert self._records is not None

        index = self._id_index.get(candidate_id)
        if index is None:
            raise RecordNotFoundError(f"Candidate ID not found: {candidate_id}")
        return self._records[index]

    def shuffle(self, seed: Optional[int] = None) -> list[CandidateRecord]:
        """
        Return a shuffled copy of the materialized dataset.

        Args:
            seed: Optional random seed for reproducibility.
        """
        records = list(self.load())
        rng = random.Random(seed)
        rng.shuffle(records)
        logger.info("Shuffled %d records (seed=%s).", len(records), seed)
        return records

    def sample(self, n: int, seed: Optional[int] = None) -> list[CandidateRecord]:
        """
        Return *n* random records without replacement.

        Uses reservoir sampling when the dataset has not been materialized yet.
        """
        if n <= 0:
            return []

        if self._records is not None:
            rng = random.Random(seed)
            population = self._records
            if n >= len(population):
                return list(population)
            return rng.sample(population, n)

        return list(self._reservoir_sample(n, seed))

    def dataset_size(self) -> int:
        """Return the number of valid records in the dataset."""
        if self._records is not None:
            return len(self._records)
        if self.cache_enabled:
            meta_path = self._cache_meta_path()
            if meta_path.is_file():
                try:
                    with meta_path.open("r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                    if meta.get("fingerprint") == _source_fingerprint(self.dataset_path):
                        return int(meta.get("record_count", 0))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
        return self._count_valid_records()

    # ------------------------------------------------------------------
    # Public API — caching
    # ------------------------------------------------------------------

    def cache_dataset(self) -> Path:
        """
        Persist the materialized dataset to ``data/cache/``.

        Returns:
            Path to the cache data file.
        """
        self._ensure_materialized()
        assert self._records is not None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data_path = self._cache_data_path()
        meta_path = self._cache_meta_path()
        fingerprint = _source_fingerprint(self.dataset_path)

        with data_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        metadata = {
            "source_path": str(self.dataset_path.resolve()),
            "fingerprint": fingerprint,
            "record_count": len(self._records),
            "format": self._format,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        logger.info("Cached %d records to %s", len(self._records), data_path)
        return data_path

    def clear_cache(self) -> None:
        """Remove cache artifacts associated with this dataset source."""
        for path in (self._cache_data_path(), self._cache_meta_path()):
            if path.is_file():
                path.unlink()
                logger.info("Removed cache file: %s", path)

    # ------------------------------------------------------------------
    # Internal — materialization & validation
    # ------------------------------------------------------------------

    def _ensure_materialized(self) -> None:
        if self._records is None:
            self.load()

    def _reset_materialized(self) -> None:
        self._records = None
        self._id_index = None
        self._stats = {
            "total_raw": 0,
            "valid": 0,
            "invalid": 0,
            "duplicates": 0,
            "load_time_seconds": 0.0,
        }

    def _materialize_from_source(self) -> list[CandidateRecord]:
        records: list[CandidateRecord] = []
        seen_ids: set[str] = set()

        for raw in self._iter_raw_records():
            self._stats["total_raw"] += 1
            record = self._process_record(raw, seen_ids)
            if record is not None:
                records.append(record)

        self._records = records
        self._build_id_index()
        return records

    def _count_valid_records(self) -> int:
        count = 0
        seen_ids: set[str] = set()
        for raw in self._iter_raw_records():
            if self._process_record(raw, seen_ids) is not None:
                count += 1
        return count

    def _reservoir_sample(self, n: int, seed: Optional[int]) -> Iterator[CandidateRecord]:
        rng = random.Random(seed)
        reservoir: list[CandidateRecord] = []
        seen_ids: set[str] = set()
        valid_index = 0

        for raw in self._iter_raw_records():
            record = self._process_record(raw, seen_ids)
            if record is None:
                continue
            valid_index += 1
            if len(reservoir) < n:
                reservoir.append(record)
            else:
                replace_at = rng.randint(1, valid_index)
                if replace_at <= n:
                    reservoir[replace_at - 1] = record

        return iter(reservoir)

    def _iter_raw_records(self) -> Iterator[CandidateRecord]:
        fmt = self._format or _detect_format(self.dataset_path)
        handler = _FORMAT_HANDLERS.get(fmt)
        if handler is None:
            raise UnsupportedFormatError(f"No handler registered for format '{fmt}'.")
        yield from handler.iter_records(self.dataset_path)

    def _process_record(
        self,
        raw: Any,
        seen_ids: set[str],
    ) -> Optional[CandidateRecord]:
        if not self.validate_records:
            if not isinstance(raw, dict):
                self._stats["invalid"] += 1
                logger.warning("Skipping non-dict record.")
                return None
            normalized = _normalize_record(raw)
            candidate_id = normalized.get("candidate_id")
            if candidate_id and candidate_id in seen_ids:
                self._stats["duplicates"] += 1
                logger.warning("Skipping duplicate candidate_id: %s", candidate_id)
                return None
            if candidate_id:
                seen_ids.add(str(candidate_id))
            self._stats["valid"] += 1
            return normalized

        validated = self._validate_record(raw)
        if validated is None:
            self._stats["invalid"] += 1
            return None

        candidate_id = str(validated["candidate_id"])
        if candidate_id in seen_ids:
            self._stats["duplicates"] += 1
            logger.warning("Skipping duplicate candidate_id: %s", candidate_id)
            return None

        seen_ids.add(candidate_id)
        self._stats["valid"] += 1
        return validated

    def _validate_record(self, raw: Any) -> Optional[CandidateRecord]:
        if not isinstance(raw, dict):
            logger.warning("Invalid record skipped: expected dict, got %s.", type(raw).__name__)
            return None

        normalized = _normalize_record(raw)
        candidate_id = normalized.get("candidate_id")
        if not candidate_id:
            logger.warning("Invalid record skipped: missing 'candidate_id'.")
            return None

        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            logger.warning("Invalid record skipped: empty 'candidate_id'.")
            return None

        if self.strict_id_format and not CANDIDATE_ID_PATTERN.match(candidate_id):
            logger.warning(
                "Invalid record skipped: candidate_id '%s' does not match expected pattern.",
                candidate_id,
            )
            return None

        normalized["candidate_id"] = candidate_id
        return normalized

    def _build_id_index(self) -> None:
        self._id_index = {}
        if self._records is None:
            return
        for index, record in enumerate(self._records):
            candidate_id = record.get("candidate_id")
            if candidate_id:
                self._id_index[str(candidate_id)] = index

    # ------------------------------------------------------------------
    # Internal — cache
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        fingerprint = _source_fingerprint(self.dataset_path)
        name = self.dataset_path.stem
        return f"{name}_{fingerprint[:16]}"

    def _cache_data_path(self) -> Path:
        return self.cache_dir / f"{self._cache_key()}.jsonl"

    def _cache_meta_path(self) -> Path:
        return self.cache_dir / f"{self._cache_key()}.meta.json"

    def _try_load_cache(self) -> bool:
        data_path = self._cache_data_path()
        meta_path = self._cache_meta_path()
        if not data_path.is_file() or not meta_path.is_file():
            logger.debug("Cache miss: artifacts not found for %s", self.dataset_path)
            return False

        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache miss: unreadable metadata (%s).", exc)
            return False

        fingerprint = _source_fingerprint(self.dataset_path)
        if meta.get("fingerprint") != fingerprint:
            logger.info("Cache miss: source file changed since cache was written.")
            return False

        records: list[CandidateRecord] = []
        seen_ids: set[str] = set()
        duplicates = 0
        invalid = 0

        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    invalid += 1
                    logger.warning("Invalid cached JSON on line %d: %s", line_number, exc)
                    continue
                if not isinstance(record, dict):
                    invalid += 1
                    continue
                candidate_id = record.get("candidate_id")
                if not candidate_id:
                    invalid += 1
                    continue
                candidate_id = str(candidate_id)
                if candidate_id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(candidate_id)
                records.append(record)

        self._records = records
        self._stats["valid"] = len(records)
        self._stats["invalid"] = invalid
        self._stats["duplicates"] = duplicates
        self._build_id_index()
        logger.info("Cache hit: loaded %d records from %s", len(records), data_path)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return loader statistics from the most recent load operation."""
        stats = dict(self._stats)
        stats["dataset_size"] = len(self._records) if self._records is not None else None
        stats["format"] = self._format
        stats["dataset_path"] = str(self.dataset_path)
        return stats


def _normalize_record(record: CandidateRecord) -> CandidateRecord:
    """
    Normalize a raw record in place (shallow copy).

    Missing values are preserved as ``None``; string fields are stripped.
    """
    normalized: CandidateRecord = {}
    for key, value in record.items():
        if value is None:
            normalized[key] = None
        elif isinstance(value, str):
            stripped = value.strip()
            normalized[key] = stripped if stripped else None
        elif isinstance(value, dict):
            normalized[key] = _normalize_record(value)  # type: ignore[arg-type]
        elif isinstance(value, list):
            normalized[key] = [
                _normalize_record(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            normalized[key] = value
    return normalized


# ---------------------------------------------------------------------------
# Demonstration entry point
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Demonstrate DatasetLoader capabilities on sample project data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    root = _project_root()

    json_path = root / "sample_candidates.json"
    jsonl_path = root / "candidates.jsonl"

    print("\n=== Dataset Loader Agent Demo ===\n")

    # --- JSON ---
    if json_path.is_file():
        print(f"--- Loading JSON: {json_path.name} ---")
        json_loader = DatasetLoader(json_path, batch_size=64, cache_enabled=False)
        json_records = json_loader.load()
        print(f"Loaded {len(json_records)} records from JSON.")
        print(f"Stats: {json_loader.get_stats()}")
    else:
        print(f"Skipping JSON demo; file not found: {json_path}")

    # --- CSV (synthetic minimal example) ---
    csv_demo_path = root / "data" / "cache" / "_demo_candidates.csv"
    csv_demo_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_demo_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "profile", "skills"])
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "CAND_0009999",
                "profile": json.dumps({"headline": "Demo CSV Candidate"}),
                "skills": json.dumps([{"name": "Python"}]),
            }
        )

    print(f"\n--- Loading CSV: {csv_demo_path.name} ---")
    csv_loader = DatasetLoader(csv_demo_path, batch_size=32, cache_enabled=False)
    csv_records = csv_loader.load_csv()
    print(f"Loaded {len(csv_records)} records from CSV.")
    print(f"First record ID: {csv_records[0]['candidate_id']}")

    # --- JSONL batch iteration (first 3 batches only for demo speed) ---
    if jsonl_path.is_file():
        print(f"\n--- Batch iteration on JSONL: {jsonl_path.name} ---")
        jsonl_loader = DatasetLoader(jsonl_path, batch_size=256, cache_enabled=True)
        total_batches = 0
        total_records = 0
        for batch_idx, batch in enumerate(jsonl_loader.iter_batches()):
            total_batches += 1
            total_records += len(batch)
            if batch_idx < 2:
                print(f"  Batch {batch_idx}: {len(batch)} records")
            if batch_idx >= 2:
                break

        print(f"Processed {total_batches} batch(es), {total_records} records (demo truncated).")
        print(f"Estimated dataset size: {jsonl_loader.dataset_size():,} valid records")

        if json_path.is_file():
            lookup_loader = DatasetLoader(json_path, batch_size=64, cache_enabled=False)
            lookup_loader.load()
            target_id = lookup_loader.get_record("CAND_0000001")["candidate_id"]
            record = lookup_loader.get_record(target_id)
            print(f"\n--- Single record lookup ---")
            print(f"  candidate_id: {record['candidate_id']}")
            profile = record.get("profile") or {}
            print(f"  headline: {profile.get('headline', 'N/A')}")
    else:
        print(f"Skipping JSONL demo; file not found: {jsonl_path}")

    print("\n=== Demo complete ===\n")


if __name__ == "__main__":
    _demo()
