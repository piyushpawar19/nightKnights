import json
from typing import List, Dict, Any, Optional

from src.schemas.jd_schema import StructuredJD
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.ranking_schema import RankedCandidate
from src.schemas.explanation_schema import Explanation, RecruiterAssessment
from src.interfaces.explanation_interface import (
    ExplanationAgentInterface,
    EvidenceExtractorInterface,
    PromptManagerInterface,
    ExplanationBuilderInterface,
    FallbackGeneratorInterface,
    LLMGenerator
)
import logging

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager # Assuming this exists


class ExplanationService(ExplanationAgentInterface):
    """Orchestrates the generation of recruiter explanations."""

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: logging.Logger,
        evidence_extractor: EvidenceExtractorInterface,
        prompt_manager: PromptManagerInterface,
        explanation_builder: ExplanationBuilderInterface,
        fallback_generator: FallbackGeneratorInterface,
        llm_generator: LLMGenerator # Dependency for LLM interaction
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.evidence_extractor = evidence_extractor
        self.prompt_manager = prompt_manager
        self.explanation_builder = explanation_builder
        self.fallback_generator = fallback_generator
        self.llm_generator = llm_generator

        self.explanation_config = (
            self.config_manager.get_config("explanation")
            if hasattr(self.config_manager, "get_config")
            else None
        )

    def generate_explanation(
        self,
        structured_jd: StructuredJD,
        candidate_profile: CandidateProfile,
        ranked_candidate: RankedCandidate,
        recruiter_assessment: Optional[RecruiterAssessment] = None
    ) -> Explanation:
        """Generates a single recruiter-style explanation for a candidate.

        Args:
            structured_jd: The structured Job Description.
            candidate_profile: The canonical candidate profile.
            ranked_candidate: The ranked candidate data.
            recruiter_assessment: Optional recruiter assessment data.

        Returns:
            An Explanation object.
        """
        candidate_id = candidate_profile.candidate_id
        self.logger.info(f"Generating explanation for candidate: {candidate_id}")

        try:
            # 1. Evidence Extraction
            evidence = self.evidence_extractor.extract_evidence(
                structured_jd, candidate_profile, ranked_candidate, recruiter_assessment
            )
            self.logger.debug(f"Evidence extracted for {candidate_id}: {evidence}")

            # 2. Prompt Building
            recruiter_prompt_template = self.prompt_manager.load_prompt("recruiter_prompt")
            prompt_variables = {
                "job_title": structured_jd.job_title,
                "company": structured_jd.company,
                "candidate_name": candidate_profile.name,
                "structured_jd": structured_jd.model_dump_json(),
                "candidate_profile": candidate_profile.model_dump_json(),
                "ranked_candidate": ranked_candidate.model_dump_json(),
                "recruiter_assessment": recruiter_assessment.model_dump_json() if recruiter_assessment else "None",
                "evidence_summary": json.dumps(evidence) # Pass evidence as JSON string
            }
            final_prompt = self.prompt_manager.inject_variables(recruiter_prompt_template, prompt_variables)
            self.logger.debug(f"Final prompt for {candidate_id}: {final_prompt[:500]}...")

            # 3. LLM Generation
            # Assuming LLMGenerator.generate returns an Explanation object directly or parses it
            llm_params: dict[str, Any] = {}
            if isinstance(self.explanation_config, dict):
                llm_params = self.explanation_config.get("llm_params", {})
            llm_explanation = self.llm_generator.generate(final_prompt, **llm_params)
            self.logger.info(f"LLM generated explanation for {candidate_id}.")

            # If LLM returns a raw string (e.g., JSON string), parse it into an Explanation object
            if isinstance(llm_explanation, str):
                parsed_data = json.loads(llm_explanation)
                # Update recruiter_assessment within parsed_data if it exists and is a dict
                if 'recruiter_assessment' in parsed_data and isinstance(parsed_data['recruiter_assessment'], dict):
                    parsed_data['recruiter_assessment'] = RecruiterAssessment(**parsed_data['recruiter_assessment'])
                llm_explanation = Explanation(**parsed_data)

            return llm_explanation

        except Exception as e:
            self.logger.error(f"Error generating explanation for {candidate_id} with LLM: {e}. Falling back.")
            # 4. Fallback Generation
            return self.fallback_generator.generate_fallback_explanation(candidate_id, reason=str(e))

    def generate_explanations_batch(
        self,
        structured_jd: StructuredJD,
        candidates_data: List[Dict[str, Any]] # Expecting list of dicts that can construct CandidateProfile and RankedCandidate
    ) -> List[Explanation]:
        """Generates recruiter-style explanations for a batch of candidates.

        This method expects a list of dictionaries, where each dictionary should contain
        enough data to construct a CandidateProfile and a RankedCandidate, and optionally
        a RecruiterAssessment. It will iterate through these, generate individual
        explanations, and return a list of Explanation objects.
        """
        self.logger.info(f"Generating explanations for a batch of {len(candidates_data)} candidates.")
        explanations: List[Explanation] = []
        for i, data in enumerate(candidates_data):
            try:
                candidate_profile = CandidateProfile(**data.get("candidate_profile", {}))
                ranked_candidate = RankedCandidate(**data.get("ranked_candidate", {}))
                recruiter_assessment = None
                if "recruiter_assessment" in data:
                    recruiter_assessment = RecruiterAssessment(**data["recruiter_assessment"])

                explanation = self.generate_explanation(
                    structured_jd,
                    candidate_profile,
                    ranked_candidate,
                    recruiter_assessment
                )
                explanations.append(explanation)
            except Exception as e:
                candidate_id = data.get("candidate_profile", {}).get("candidate_id", f"unknown_candidate_{i}")
                self.logger.error(f"Failed to process candidate {candidate_id} in batch: {e}")
                # Append fallback explanation for the failed candidate to ensure batch completion
                explanations.append(self.fallback_generator.generate_fallback_explanation(candidate_id, reason=f"Batch processing failed for candidate: {e}"))
        return explanations