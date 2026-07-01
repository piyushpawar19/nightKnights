"""Routing utilities for conditional graph branching.

This module provides the *architecture* for dynamic routing within the
LangGraph pipeline.  The actual routing logic is intentionally minimal
(stubs returning ``"continue"`` / ``"evaluate"`` etc.) so that it can be
extended later without restructuring the graph.

How to wire a conditional edge
------------------------------
In ``graph.py``, replace a plain ``add_edge`` with::

    graph.add_conditional_edges(
        "hybrid_ranking",
        route_after_ranking,
        {
            RouteDecision.CONTINUE: "reranking",
            RouteDecision.RETRY: "retrieve_candidates",
            RouteDecision.ERROR_EXIT: END,
        },
    )

Extension points
----------------
* Add new ``RouteDecision`` members for additional control-flow paths.
* Add new router functions following the same signature:
  ``(dict) -> str``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Route decision enum
# ---------------------------------------------------------------------------

class RouteDecision(str, Enum):
    """Named route outcomes used as edge targets in LangGraph.

    Each value must match a key in the ``add_conditional_edges`` mapping.
    """

    CONTINUE = "continue"
    RETRY = "retry"
    SKIP = "skip"
    EARLY_EXIT = "early_exit"
    ERROR_EXIT = "error_exit"


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def should_retry_retrieval(state: dict[str, Any]) -> str:
    """Decide whether retrieval should be retried.

    Future logic might check:
    * ``len(state["retrieved_candidates"]) < min_threshold``
    * retrieval-specific error flags

    Returns
    -------
    str
        A ``RouteDecision`` value.
    """
    candidates = state.get("retrieved_candidates", [])
    errors = state.get("errors", [])

    retrieval_errors = [e for e in errors if e.get("node_name") == "retrieve_candidates"]
    if retrieval_errors:
        logger.warning(
            "Retrieval had errors — triggering error exit."
        )
        return RouteDecision.ERROR_EXIT

    if not candidates: # If no candidates are retrieved, trigger an error exit
        logger.info("No candidates retrieved, triggering error exit.")
        return RouteDecision.ERROR_EXIT

    logger.debug(
        "should_retry_retrieval: %d candidates retrieved → %s",
        len(candidates),
        RouteDecision.CONTINUE,
    )
    return RouteDecision.CONTINUE


def should_skip_evaluation(state: dict[str, Any]) -> str:
    """Decide whether the evaluation node should be skipped.

    Future logic might check:
    * a config flag ``skip_evaluation: true``
    * presence of ground-truth labels

    Returns
    -------
    str
        ``RouteDecision.CONTINUE`` to evaluate, ``RouteDecision.SKIP`` to skip.
    """
    # Stub: always evaluate
    logger.debug("should_skip_evaluation: → %s", RouteDecision.CONTINUE)
    return RouteDecision.CONTINUE


def check_error_threshold(state: dict[str, Any]) -> str:
    """Decide whether to abort the pipeline due to accumulated errors.

    Future logic might check:
    * ``len(state["errors"]) > max_allowed_errors``
    * critical-node failure flags

    Returns
    -------
    str
        ``RouteDecision.CONTINUE`` or ``RouteDecision.ERROR_EXIT``.
    """
    errors = state.get("errors", [])
    max_errors = 5  # Configurable threshold

    if len(errors) >= max_errors:
        logger.error(
            "Error threshold reached (%d/%d) — would trigger early exit.",
            len(errors),
            max_errors,
        )
        return RouteDecision.ERROR_EXIT

    logger.debug(
        "check_error_threshold: %d errors (threshold %d) → %s",
        len(errors),
        max_errors,
        RouteDecision.CONTINUE,
    )
    return RouteDecision.CONTINUE


def route_after_ranking(state: dict[str, Any]) -> str:
    """Decide what to do after hybrid ranking completes.

    Future logic might:
    * route to re-ranking if enough candidates
    * retry retrieval if too few candidates
    * exit on critical error

    Returns
    -------
    str
        A ``RouteDecision`` value.
    """
    ranked = state.get("ranked_candidates", [])
    errors = state.get("errors", [])
    ranking_errors = [e for e in errors if e.get("node_name") in ["hybrid_ranking", "reranking"]]

    if ranking_errors:
        logger.warning("Ranking had errors — triggering error exit.")
        return RouteDecision.ERROR_EXIT

    if not ranked:
        logger.warning("No ranked candidates — triggering error exit.")
        return RouteDecision.ERROR_EXIT

    logger.debug(
        "route_after_ranking: %d candidates ranked → %s",
        len(ranked),
        RouteDecision.CONTINUE,
    )
    return RouteDecision.CONTINUE
