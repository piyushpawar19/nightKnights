from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.schemas.jd_schema import StructuredJD
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.ranking_schema import RankedCandidate
from src.schemas.explanation_schema import Explanation, RecruiterAssessment


class LLMGenerator(ABC):
    """Abstract interface for an LLM that generates explanations."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Explanation:
        """Generates a single explanation based on a prompt."""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Explanation]:
        """Generates explanations in a batch based on a list of prompts."""
        pass


class ExplanationAgentInterface(ABC):
    """Abstract interface for the Explanation Agent."""

    @abstractmethod
    def generate_explanation(
        self,
        structured_jd: StructuredJD,
        candidate_profile: CandidateProfile,
        ranked_candidate: RankedCandidate,
        recruiter_assessment: Optional[RecruiterAssessment] = None
    ) -> Explanation:
        """Generates a single recruiter-style explanation for a candidate."""
        pass

    @abstractmethod
    def generate_explanations_batch(
        self,
        structured_jd: StructuredJD,
        candidates_data: List[Dict[str, Any]]
    ) -> List[Explanation]:
        """Generates recruiter-style explanations for a batch of candidates."""
        pass


class EvidenceExtractorInterface(ABC):
    """Abstract interface for extracting evidence from various inputs."""
    @abstractmethod
    def extract_evidence(
        self,
        structured_jd: StructuredJD,
        candidate_profile: CandidateProfile,
        ranked_candidate: RankedCandidate,
        recruiter_assessment: Optional[RecruiterAssessment] = None
    ) -> Dict[str, Any]:
        """Extracts structured evidence relevant for explanation generation."""
        pass


class PromptManagerInterface(ABC):
    """Abstract interface for managing prompts."""
    @abstractmethod
    def load_prompt(self, prompt_name: str) -> str:
        """Loads a prompt template by name."""
        pass

    @abstractmethod
    def inject_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Injects variables into a prompt template."""
        pass


class ExplanationBuilderInterface(ABC):
    """Abstract interface for building explanations from evidence."""
    @abstractmethod
    def build_explanation(
        self,
        candidate_id: str,
        evidence: Dict[str, Any],
        recruiter_assessment: Optional[RecruiterAssessment] = None
    ) -> Explanation:
        """Builds a structured Explanation object from extracted evidence."""
        pass


class FallbackGeneratorInterface(ABC):
    """Abstract interface for generating fallback explanations."""
    @abstractmethod
    def generate_fallback_explanation(
        self,
        candidate_id: str,
        reason: str = "LLM explanation failed or timed out."
    ) -> Explanation:
        """Generates a deterministic fallback explanation."""
        pass