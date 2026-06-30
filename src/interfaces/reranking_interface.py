from abc import ABC, abstractmethod
from typing import Dict, Any
from ..schemas.reranking_schema import RerankedCandidates, RecruiterAssessment

class LLMInterface(ABC):
    """Abstract Base Class for LLM interactions."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invokes the LLM with a given prompt and returns the raw response."""
        pass

class RerankingAgentInterface(ABC):
    """Abstract Base Class for the Recruiter Reranking Agent."""

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the recruiter reranking process.

        Args:
            state (Dict[str, Any]): The current state dictionary containing parsed JD, ranked candidates, and candidate features.

        Returns:
            Dict[str, Any]: The updated state dictionary with 'reranked_candidates'.
        """
        pass

class RecruiterRerankerInterface(ABC):
    """Abstract Base Class for the Recruiter Reranker."""

    @abstractmethod
    def rerank_candidates(self, 
                          parsed_jd: Dict[str, Any],
                          ranked_candidates: List[Dict[str, Any]],
                          candidate_features: Dict[str, Any],
                          llm_interface: LLMInterface) -> RerankedCandidates:
        """Reranks candidates based on recruiter-style reasoning.

        Args:
            parsed_jd (Dict[str, Any]): Parsed Job Description.
            ranked_candidates (List[Dict[str, Any]]): List of candidates ranked by the hybrid ranker.
            candidate_features (Dict[str, Any]): Features for each candidate.
            llm_interface (LLMInterface): The LLM interface to use for generating assessments.

        Returns:
            RerankedCandidates: A Pydantic model containing the reranked candidates.
        """
        pass

class PromptBuilderInterface(ABC):
    """Abstract Base Class for building prompts."""

    @abstractmethod
    def build_recruiter_prompt(self, parsed_jd: Dict[str, Any], candidate_features: Dict[str, Any], ranked_candidates: List[Dict[str, Any]]) -> str:
        """Builds a structured prompt for the recruiter re-ranking LLM.

        Args:
            parsed_jd (Dict[str, Any]): Parsed Job Description.
            candidate_features (Dict[str, Any]): Features for all candidates.
            ranked_candidates (List[Dict[str, Any]]): Candidates from the hybrid ranker.

        Returns:
            str: The structured prompt.
        """
        pass

class ResponseParserInterface(ABC):
    """Abstract Base Class for parsing LLM responses."""

    @abstractmethod
    def parse_recruiter_assessment(self, response: str) -> RecruiterAssessment:
        """Parses and validates the LLM's recruiter assessment response.

        Args:
            response (str): The raw JSON response from the LLM.

        Returns:
            RecruiterAssessment: A validated Pydantic model of the assessment.
        """
        pass
