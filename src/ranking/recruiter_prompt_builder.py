
import json
from typing import Dict, Any, List
from ..interfaces.reranking_interface import PromptBuilderInterface
from ..prompts.recruiter_prompt import REC_RERANKER_SYSTEM_PROMPT, REC_RERANKER_USER_PROMPT
from ..schemas.reranking_schema import RecruiterAssessment

class RecruiterPromptBuilder(PromptBuilderInterface):
    """Builds structured prompts for the recruiter re-ranking LLM."""

    def build_recruiter_prompt(self, 
                               parsed_jd: Dict[str, Any],
                               candidate_features: Dict[str, Any],
                               ranked_candidates: List[Dict[str, Any]]) -> str:
        """Builds a structured prompt for the recruiter re-ranking LLM.

        The prompt requests strict JSON output only, adhering to the RecruiterAssessment schema.

        Args:
            parsed_jd (Dict[str, Any]): Parsed Job Description.
            candidate_features (Dict[str, Any]): Features for all candidates.
            ranked_candidates (List[Dict[str, Any]]): Candidates from the hybrid ranker.

        Returns:
            str: The structured prompt combining system and user instructions.
        """
        # We need to iterate through each candidate in ranked_candidates and pull out their
        # features from candidate_features. Then format this for the prompt.

        candidate_assessments = []
        for candidate in ranked_candidates:
            candidate_id = candidate.get("candidate_id")
            if candidate_id and candidate_id in candidate_features:
                features = candidate_features[candidate_id]
                candidate_assessments.append({
                    "candidate_id": candidate_id,
                    "previous_rank": candidate.get("rank"),
                    "previous_score": candidate.get("score"),
                    "features": features
                })

        user_prompt_formatted = REC_RERANKER_USER_PROMPT.format(
            parsed_jd=json.dumps(parsed_jd, indent=2),
            candidate_features=json.dumps(candidate_assessments, indent=2),
            ranked_candidates=json.dumps(ranked_candidates, indent=2) # This might be redundant if candidate_assessments is comprehensive
        )

        # Combine system and user prompts
        full_prompt = f"{REC_RERANKER_SYSTEM_PROMPT}\n\n{user_prompt_formatted}"

        return full_prompt
