from typing import List, Dict, Any, Optional

from src.schemas.jd_schema import StructuredJD
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.ranking_schema import RankedCandidate
from src.schemas.explanation_schema import Explanation, RecruiterAssessment
from src.interfaces.explanation_interface import ExplanationAgentInterface, LLMGenerator
from src.explainability.evidence_extractor import EvidenceExtractor
from src.explainability.prompt_manager import PromptManager
from src.explainability.explanation_builder import ExplanationBuilder
from src.explainability.fallback_generator import FallbackGenerator
from src.explainability.explanation_service import ExplanationService
import logging

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager # Assuming this exists


class ExplanationAgent(ExplanationAgentInterface):
    """The main entry point for the Recruiter Explanation Agent.
    Orchestrates the explanation generation process.
    """

    def __init__(self, config_manager: ConfigManager, logger: logging.Logger, llm_generator: LLMGenerator):
        self.config_manager = config_manager
        self.logger = logger
        self.llm_generator = llm_generator

        self.prompts_dir = self.config_manager.get_config("paths.prompts_dir")
        self.fallback_template_path = self.config_manager.get_config("paths.fallback_template")

        # Initialize sub-components
        self.prompt_manager = PromptManager(self.prompts_dir, self.logger)
        self.fallback_template_content = self.prompt_manager._load_file_content(self.fallback_template_path)

        self.evidence_extractor = EvidenceExtractor()
        self.explanation_builder = ExplanationBuilder()
        self.fallback_generator = FallbackGenerator(self.fallback_template_content, self.logger)

        self.explanation_service = ExplanationService(
            config_manager=self.config_manager,
            logger=self.logger,
            evidence_extractor=self.evidence_extractor,
            prompt_manager=self.prompt_manager,
            explanation_builder=self.explanation_builder,
            fallback_generator=self.fallback_generator,
            llm_generator=self.llm_generator
        )

    def generate_explanation(
        self,
        structured_jd: StructuredJD,
        candidate_profile: CandidateProfile,
        ranked_candidate: RankedCandidate,
        recruiter_assessment: Optional[RecruiterAssessment] = None
    ) -> Explanation:
        """Generates a single recruiter-style explanation for a candidate.
        Delegates to the ExplanationService.
        """
        return self.explanation_service.generate_explanation(
            structured_jd,
            candidate_profile,
            ranked_candidate,
            recruiter_assessment
        )

    def generate_explanations_batch(
        self,
        structured_jd: StructuredJD,
        candidates_data: List[Dict[str, Any]]
    ) -> List[Explanation]:
        """Generates recruiter-style explanations for a batch of candidates.
        Delegates to the ExplanationService.
        """
        return self.explanation_service.generate_explanations_batch(
            structured_jd,
            candidates_data
        )
