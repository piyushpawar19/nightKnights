"""Entry point for the recruiter-style AI hiring system."""

import sys
from src.graph import run_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Main function to run the LangGraph pipeline with a sample job description."""
    sample_jd = (
        "We are looking for a Senior Machine Learning Engineer to join our team in Bangalore. "
        "The ideal candidate will have 5+ years of experience building ML pipelines, "
        "strong proficiency in Python and TensorFlow, and experience with LangChain and Kubernetes."
    )
    
    logger.info("Starting AI Hackathon Pipeline Execution")
    try:
        final_state = run_pipeline(sample_jd)
        
        # Log summary of execution
        metadata = final_state.get("execution_metadata", {})
        run_id = metadata.get("run_id", "N/A")
        logger.info("Pipeline completed. Run ID: %s", run_id)
        
        # Display key fields populated
        logger.info("Structured JD parsed: %s", final_state.get("structured_jd") is not None)
        logger.info("Skills extracted: %s", final_state.get("extracted_skills"))
        logger.info("Number of candidates retrieved: %d", len(final_state.get("retrieved_candidates", [])))
        logger.info("Number of candidates ranked: %d", len(final_state.get("ranked_candidates", [])))
        logger.info("Number of candidates reranked: %d", len(final_state.get("reranked_candidates", [])))
        logger.info("Number of explanations generated: %d", len(final_state.get("explanations", [])))
        logger.info("Submission CSV generated at: %s", final_state.get("submission_path"))
        
        # Log accumulated timestamps
        logger.info("Execution Timing Summary:")
        for ts in final_state.get("timestamps", []):
            logger.info("  - Node '%s': %.1f ms (Status: %s)", ts.get("node_name"), ts.get("duration_ms"), ts.get("status"))
            
        # Log any errors
        errors = final_state.get("errors", [])
        if errors:
            logger.warning("Pipeline finished with %d warnings/errors:", len(errors))
            for err in errors:
                logger.warning("  - Node '%s': %s - %s", err.get("node_name"), err.get("error_type"), err.get("error_message"))
        else:
            logger.info("Pipeline executed with zero errors.")
            
    except Exception as e:
        logger.exception("Unhandled exception during pipeline execution: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
