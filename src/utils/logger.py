"""Structured logging utilities for the candidate-ranking pipeline.

Provides:
* ``get_logger(name)`` — factory for named loggers with structured formatting.
* ``NodeExecutionLogger`` — context manager that wraps node execution with
  automatic timing, state-change tracking, and error capture.

All logging goes through Python's built-in ``logging`` module.  No print
statements should be used anywhere in the pipeline.

Example
-------
::

    from src.utils.logger import get_logger, NodeExecutionLogger

    logger = get_logger(__name__)

    def my_node(state):
        with NodeExecutionLogger("my_node", state, logger) as ctx:
            # ... node logic ...
            ctx.mark_fields_updated(["ranked_candidates"])
        return ctx.build_timestamp_entry()
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from schemas.graph_schema import NodeError, NodeStatus, NodeTimestamp


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def _configure_root_once() -> None:
    """Configure the root logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — human-readable for development
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with structured formatting.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    _configure_root_once()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Node execution context manager
# ---------------------------------------------------------------------------

class NodeExecutionLogger:
    """Context manager that instruments a pipeline node execution.

    Handles:
    * Logging node entry and exit.
    * Timing the execution.
    * Tracking which state fields were updated.
    * Capturing exceptions and building ``NodeError`` dicts.

    Parameters
    ----------
    node_name : str
        Human-readable identifier for the node.
    state : dict
        The current ``PipelineState`` (read-only reference for diagnostics).
    logger : logging.Logger
        Logger instance to write to.
    """

    def __init__(
        self,
        node_name: str,
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.node_name = node_name
        self._state = state
        self._logger = logger

        self._start_time: float = 0.0
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._duration_ms: float = 0.0
        self._status: NodeStatus = NodeStatus.PENDING
        self._updated_fields: list[str] = []
        self._errors: list[NodeError] = [] # Changed to list to support multiple errors

    # -- Context manager protocol ------------------------------------------

    def __enter__(self) -> NodeExecutionLogger:
        self._start_time = time.perf_counter()
        self._started_at = datetime.now(tz=timezone.utc)
        self._status = NodeStatus.RUNNING

        self._logger.info(
            "[%s] START - Node execution started",
            self.node_name,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        elapsed = time.perf_counter() - self._start_time
        self._duration_ms = round(elapsed * 1000, 2)
        self._completed_at = datetime.now(tz=timezone.utc)

        if exc_val is not None:
            # If an unhandled exception occurs, capture it as a NodeError
            self._errors.append(NodeError(
                node_name=self.node_name,
                error_type=type(exc_val).__name__,
                error_message=str(exc_val),
                timestamp=self._completed_at,
                traceback=traceback.format_exc(),
            ))
            self._status = NodeStatus.FAILED
            self._logger.error(
                "[%s] ERROR - Node failed after %.1fms - %s: %s",
                self.node_name,
                self._duration_ms,
                type(exc_val).__name__,
                exc_val,
            )
            # Suppress the exception so the graph can continue its error handling
            return True
        
        if self._errors: # If errors were explicitly added within the node
            self._status = NodeStatus.FAILED
            self._logger.warning(
                "[%s] WARNING - Node completed with %d errors in %.1fms - updated: %s",
                self.node_name,
                len(self._errors),
                self._duration_ms,
                ", ".join(self._updated_fields) or "(none)",
            )
        else:
            self._status = NodeStatus.SUCCESS
            self._logger.info(
                "[%s] SUCCESS - Node completed in %.1fms - updated: %s",
                self.node_name,
                self._duration_ms,
                ", ".join(self._updated_fields) or "(none)",
            )
        return False

    # -- Public helpers ----------------------------------------------------

    def mark_fields_updated(self, fields: list[str]) -> None:
        """Record which state fields this node updated."""
        self._updated_fields = fields

    @property
    def failed(self) -> bool:
        """Whether the node execution failed."""
        return self._status == NodeStatus.FAILED or bool(self._errors)
    
    @property
    def errors(self) -> list[NodeError]:
        """Return the list of errors captured during node execution."""
        return self._errors

    def add_error(self, error: NodeError) -> None:
        """Add a NodeError to the list of captured errors."""
        if isinstance(error, list):
            self._errors.extend(error) # Extend if a list of errors is passed
        else:
            self._errors.append(error)
        self._status = NodeStatus.FAILED # Mark as failed if an error is added

    def build_timestamp_entry(self) -> dict[str, Any]:
        """Build a serialised ``NodeTimestamp`` dict for appending to state."""
        ts = NodeTimestamp(
            node_name=self.node_name,
            started_at=self._started_at or datetime.now(tz=timezone.utc),
            completed_at=self._completed_at or datetime.now(tz=timezone.utc),
            duration_ms=self._duration_ms,
            status=self._status,
        )
        return ts.model_dump(mode="json")

    def build_error_entry(self) -> dict[str, Any] | None:
        """Build a serialised ``NodeError`` dict, or ``None`` if no error."""
        # This method might need adjustment if multiple errors are to be returned
        if self._errors:
            return self._errors[0].model_dump(mode="json") # Return the first error for compatibility, or refactor caller
        return None



