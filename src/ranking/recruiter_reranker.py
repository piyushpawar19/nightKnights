import logging
from typing import Dict, Any, List
from pydantic import ValidationError
from functools import lru_cache
import os
from joblib import Memory

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

from ..interfaces.reranking_interface import RecruiterRerankerInterface, LLMInterface, PromptBuilderInterface, ResponseParserInterface
from ..schemas.reranking_schema import RerankedCandidates, RerankedCandidate, RecruiterAssessment
from .reranking_utils import normalize_score, stable_sort_candidates, load_ranking_weights

logger = logging.getLogger(__name__)

class RecruiterReranker(RecruiterRerankerInterface):
    """Refines candidate ranking using recruiter-style reasoning via an LLM."""

    def __init__(self, 
                 prompt_builder: PromptBuilderInterface,
                 response_parser: ResponseParserInterface,
                 config_path: str = "configs/ranking.yaml",
                 top_k: int = 10):
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.ranking_weights = load_ranking_weights(config_path)
        self.top_k = top_k

    @memory.cache # Cache reranking results for identical inputs
    def rerank_candidates(self, 
                           parsed_jd_json: str,
                           ranked_candidates_json: str,
                           candidate_features_json: str,
                           llm_interface: LLMInterface) -> RerankedCandidates:
        """Reranks candidates based on recruiter-style reasoning.

        Args:
            parsed_jd_json (str): JSON string of Parsed Job Description.
            ranked_candidates_json (str): JSON string of List of candidates ranked by the hybrid ranker.
            candidate_features_json (str): JSON string of Features for each candidate.
            llm_interface (LLMInterface): The LLM interface to use for generating assessments.

        Returns:
            RerankedCandidates: A Pydantic model containing the reranked candidates.
        """
        import json
        parsed_jd = json.loads(parsed_jd_json)
        ranked_candidates = json.loads(ranked_candidates_json)
        candidate_features = json.loads(candidate_features_json)

        if not ranked_candidates:
            logger.info("No candidates to rerank. Returning empty list.")
            return RerankedCandidates(candidates=[])

        reranked_results: List[Dict[str, Any]] = []

        for candidate in ranked_candidates:
            candidate_id = candidate.get("candidate_id")
            if not candidate_id:
                logger.warning(f"Candidate missing \"candidate_id\". Skipping: {candidate}")
                continue

            # Build prompt for individual candidate assessment
            candidate_prompt = self.prompt_builder.build_recruiter_prompt(
                parsed_jd=parsed_jd,
                candidate_features={candidate_id: candidate_features.get(candidate_id, {})},
                ranked_candidates=[candidate]
            )

            llm_response = llm_interface.invoke(candidate_prompt)
            
            try:
                assessment = self.response_parser.parse_recruiter_assessment(llm_response)
            except ValueError as e:
                logger.error(f"Failed to parse recruiter assessment for candidate {candidate_id}: {e}")
                # Fallback: use default assessment or skip candidate
                assessment = self._create_default_assessment()
            except Exception as e:
                logger.error(f"Unexpected error during assessment parsing for candidate {candidate_id}: {e}")
                assessment = self._create_default_assessment()

            # Combine scores
            previous_score = candidate.get("score", 0.0)
            recruiter_score = normalize_score(assessment.recruiter_score)
            
            hybrid_weight = self.ranking_weights.get("hybrid_score_weight", 0.5)
            recruiter_weight = self.ranking_weights.get("recruiter_score_weight", 0.5)

            # Ensure weights sum to 1 to maintain score range
            total_weight = hybrid_weight + recruiter_weight
            if total_weight == 0:
                logger.warning("Both hybrid and recruiter weights are zero. Using equal weights.")
                hybrid_weight = 0.5
                recruiter_weight = 0.5
                total_weight = 1.0
            
            final_score = (
                (previous_score * (hybrid_weight / total_weight)) +
                (recruiter_score * (recruiter_weight / total_weight))
            )

            reranked_candidate = RerankedCandidate(
                candidate_id=candidate_id,
                previous_rank=candidate.get("rank", 0),
                new_rank=0, # Placeholder, will be updated after stable sort
                previous_score=previous_score,
                recruiter_score=recruiter_score,
                final_score=final_score,
                assessment=assessment
            )
            reranked_results.append(reranked_candidate.model_dump())
        
        # Apply stable sorting and assign final ranks
        final_reranked_list = stable_sort_candidates(reranked_results)

        # Convert dictionaries back to Pydantic models and apply top_k
        return RerankedCandidates(candidates=[RerankedCandidate(**c) for c in final_reranked_list[:self.top_k]])

    def _create_default_assessment(self) -> RecruiterAssessment:
        """Creates a default RecruiterAssessment in case of LLM parsing failure."""
        logger.warning("Creating default recruiter assessment due to parsing failure.")
        return RecruiterAssessment(
            strengths=["N/A"],
            concerns=["LLM assessment failed or was malformed."],
            hiring_recommendation="No",
            confidence=0.1, # Low confidence
            recruiter_score=0.1, # Low score
            reasoning="Failed to obtain a valid assessment from the LLM. Defaulting to low scores."
        )
