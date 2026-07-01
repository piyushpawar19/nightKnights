"""LangGraph StateGraph construction and execution entry points.

This module wires up the individual placeholder nodes and routing functions
into a compiled LangGraph pipeline that coordinates the recruitment workflow.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes import (
    csv_generation_node,
    evaluation_node,
    explanation_node,
    feature_engineering_node,
    hybrid_ranking_node,
    parse_jd_node,
    retrieve_candidates_node,
    reranking_node,
)
from src.graph.router import (
    RouteDecision,
    route_after_ranking,
    should_retry_retrieval,
    should_skip_evaluation,
    check_error_threshold,
)
from src.state.pipeline_state import PipelineState, create_initial_state
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph pipeline.

    The graph flow is structured as follows:
        START -> parse_jd -> retrieve_candidates
              -> (retry check) -> feature_engineering
              -> hybrid_ranking -> (re-rank check) -> reranking
              -> explanation -> (evaluation skip check) -> evaluation
              -> csv_generation -> END

    Returns
    -------
    CompiledGraph
        The executable LangGraph instance.
    """
    logger.info("Constructing candidate ranking graph workflow")

    workflow = StateGraph(PipelineState)

    # 1. Add all nodes
    workflow.add_node("parse_jd", parse_jd_node)
    workflow.add_node("retrieve_candidates", retrieve_candidates_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("hybrid_ranking", hybrid_ranking_node)
    workflow.add_node("reranking", reranking_node)
    workflow.add_node("explanation", explanation_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("csv_generation", csv_generation_node)

    # 2. Add edges and conditional routing
    workflow.add_edge(START, "parse_jd")
    workflow.add_edge("parse_jd", "retrieve_candidates")

    # Routing after retrieval (allows for retry if list is empty)
    workflow.add_conditional_edges(
        "retrieve_candidates",
        should_retry_retrieval,
        {
            RouteDecision.CONTINUE: "feature_engineering",
            RouteDecision.RETRY: "retrieve_candidates",
            RouteDecision.ERROR_EXIT: END,  # Added error exit
        },
    )

    workflow.add_edge("feature_engineering", "hybrid_ranking")

    # Routing after hybrid ranking (allows for retrying retrieval or proceeding)
    workflow.add_conditional_edges(
        "hybrid_ranking",
        route_after_ranking,
        {
            RouteDecision.CONTINUE: "reranking",
            RouteDecision.RETRY: "retrieve_candidates",
            RouteDecision.ERROR_EXIT: END,  # Added error exit
        },
    )

    workflow.add_edge("reranking", "explanation")

    # Routing after explanation (allows skipping evaluation if ground-truth is absent)
    workflow.add_conditional_edges(
        "explanation",
        should_skip_evaluation,
        {
            RouteDecision.CONTINUE: "evaluation",
            RouteDecision.SKIP: "csv_generation",
            RouteDecision.ERROR_EXIT: END,  # Added error exit
        },
    )

    # Check for overall error threshold after nodes that don\"t have specific routing
    workflow.add_conditional_edges(
        "evaluation",
        check_error_threshold,
        {
            RouteDecision.CONTINUE: "csv_generation",
            RouteDecision.ERROR_EXIT: END,
        },
    )

    workflow.add_edge("csv_generation", END)

    logger.info("Compiling candidate ranking graph workflow")
    return workflow.compile()


def run_pipeline(raw_jd: str) -> dict[str, Any]:
    """Execute the candidate ranking pipeline from end to end.

    Parameters
    ----------
    raw_jd : str
        The verbatim job-description text.

    Returns
    -------
    dict[str, Any]
        The final PipelineState dictionary after pipeline completion.
    """
    logger.info("Initializing pipeline run")
    initial_state = create_initial_state(raw_jd)
    graph = build_graph()
    
    logger.info("Invoking graph execution")
    final_state = graph.invoke(initial_state)
    return final_state



