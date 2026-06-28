#!/usr/bin/env python3
"""
===============================================================================
AI Hiring Intelligence System
-------------------------------------------------------------------------------

Main Entry Point

This file intentionally contains **minimal business logic**.

Responsibilities:
1. Initialize logging
2. Validate startup inputs (e.g., Job Description)
3. Invoke the LangGraph orchestration pipeline
4. Display execution summary
5. Handle top-level exceptions

All business logic (retrieval, ranking, explainability, evaluation, CSV export)
must remain inside their respective modules.

Execution Flow:

main.py
    │
    ▼
run_pipeline()
    │
    ▼
LangGraph Orchestrator
    │
    ├── JD Parser
    ├── Skill Extraction
    ├── Retrieval Engine
    │      ├── DatasetLoader
    │      ├── CandidateProfileBuilder
    │      ├── Dense Retrieval
    │      ├── BM25 Retrieval
    │      └── Vector Store
    ├── Hybrid Ranker
    ├── Recruiter Re-Ranker
    ├── Explanation Generator
    ├── CSV Generator
    └── Evaluation

===============================================================================
"""

import sys
from pathlib import Path

# LangGraph entry point
from src.graph import run_pipeline

# Project logger
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    Main application entry point.

    This function should remain lightweight.
    It only prepares execution and delegates all work to LangGraph.
    """

    # -------------------------------------------------------------------------
    # Input Job Description
    # -------------------------------------------------------------------------
    jd_path = Path("job_description.txt")

    if not jd_path.exists():
        logger.error("Job Description not found: %s", jd_path)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Starting AI Hiring Intelligence Pipeline")
    logger.info("=" * 70)

    try:
        # ---------------------------------------------------------------------
        # Execute the complete workflow
        # ---------------------------------------------------------------------
        final_state = run_pipeline(jd_path)

        metadata = final_state.get("execution_metadata", {})

        logger.info("Run ID : %s", metadata.get("run_id", "N/A"))

        # ---------------------------------------------------------------------
        # Pipeline Summary
        # ---------------------------------------------------------------------
        logger.info("Structured JD Parsed : %s",
                    final_state.get("structured_jd") is not None)

        logger.info("Skills Extracted : %s",
                    len(final_state.get("extracted_skills", [])))

        logger.info("Candidates Retrieved : %d",
                    len(final_state.get("retrieved_candidates", [])))

        logger.info("Candidates Ranked : %d",
                    len(final_state.get("ranked_candidates", [])))

        logger.info("Candidates Re-Ranked : %d",
                    len(final_state.get("reranked_candidates", [])))

        logger.info("Explanations Generated : %d",
                    len(final_state.get("explanations", [])))

        logger.info("Submission File : %s",
                    final_state.get("submission_path", "Not Generated"))

        # ---------------------------------------------------------------------
        # Execution Timing
        # ---------------------------------------------------------------------
        logger.info("-" * 70)
        logger.info("Execution Timeline")
        logger.info("-" * 70)

        for timestamp in final_state.get("timestamps", []):
            logger.info(
                "%-25s %.2f ms (%s)",
                timestamp.get("node_name"),
                timestamp.get("duration_ms"),
                timestamp.get("status"),
            )

        # ---------------------------------------------------------------------
        # Pipeline Errors
        # ---------------------------------------------------------------------
        errors = final_state.get("errors", [])

        if errors:
            logger.warning("Pipeline completed with %d warning(s).", len(errors))

            for error in errors:
                logger.warning(
                    "[%s] %s : %s",
                    error.get("node_name"),
                    error.get("error_type"),
                    error.get("error_message"),
                )
        else:
            logger.info("Pipeline completed successfully with no errors.")

    except Exception:
        logger.exception("Fatal error during pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()