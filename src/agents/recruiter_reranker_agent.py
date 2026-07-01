
import logging
from typing import Dict, Any

from ..interfaces.reranking_interface import RerankingAgentInterface, LLMInterface, PromptBuilderInterface, ResponseParserInterface, RecruiterRerankerInterface
from ..ranking.recruiter_reranker import RecruiterReranker
from ..ranking.recruiter_prompt_builder import RecruiterPromptBuilder
from ..ranking.recruiter_parser import RecruiterParser
from ..schemas.reranking_schema import RerankedCandidates

logger = logging.getLogger(__name__)

class RecruiterRerankerAgent(RerankingAgentInterface):
    """Agent for re-ranking candidates using recruiter-style reasoning."""

    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.prompt_builder = RecruiterPromptBuilder()
        self.response_parser = RecruiterParser()
        self.recruiter_reranker = RecruiterReranker(
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the recruiter re-ranking process.

        Args:
            state (Dict[str, Any]): The current state dictionary containing:
                - "parsed_jd": Parsed Job Description.
                - "ranked_candidates": List of candidates ranked by the hybrid ranker.
                - "candidate_features": Features for each candidate.

        Returns:
            Dict[str, Any]: The updated state dictionary with "reranked_candidates".
        
        Raises:
            ValueError: If critical input keys are missing from the state.
        """
        logger.info("Starting RecruiterRerankerAgent run.")

        # 1. Validate inputs
        parsed_jd = state.get("parsed_jd")
        ranked_candidates = state.get("ranked_candidates")
        candidate_features = state.get("candidate_features")

        if not all([parsed_jd, ranked_candidates, candidate_features]):
            missing_keys = [key for key, value in {
                "parsed_jd": parsed_jd,
                "ranked_candidates": ranked_candidates,
                "candidate_features": candidate_features
            }.items() if value is None]
            raise ValueError(f"Missing critical input keys in state: {', '.join(missing_keys)}")
        
        # 2. Perform reranking using the RecruiterReranker module
            reranked_candidates_pydantic: RerankedCandidates = self.recruiter_reranker.rerank_candidates(
                parsed_jd=parsed_jd,
                ranked_candidates=ranked_candidates,
                candidate_features=candidate_features,
                llm_interface=self.llm_interface
            )

            # 3. Update the state with the reranked candidates
            state["reranked_candidates"] = reranked_candidates_pydantic.model_dump()
            
            logger.info("RecruiterRerankerAgent finished successfully.")
            return state
