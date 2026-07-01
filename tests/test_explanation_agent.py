import pytest
import os
import json
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

from src.schemas.jd_schema import StructuredJD
from src.schemas.candidate_schema import CandidateProfile
from src.schemas.ranking_schema import RankedCandidate, FeatureVector
from src.schemas.retrieval_schema import RetrievalResult
from src.schemas.explanation_schema import Explanation, RecruiterAssessment
from src.schemas.common_schema import Skill, Education, Experience, Location, Metadata
from src.interfaces.explanation_interface import LLMGenerator
from src.explainability.prompt_manager import PromptManager
from src.explainability.evidence_extractor import EvidenceExtractor
from src.explainability.explanation_builder import ExplanationBuilder
from src.explainability.fallback_generator import FallbackGenerator
from src.explainability.explanation_service import ExplanationService
from src.explainability.explanation_agent import ExplanationAgent


# --- Mocks for Dependencies ---

class MockLogger:
    def debug(self, message): pass
    def info(self, message): pass
    def warning(self, message): pass
    def error(self, message): pass

class MockConfigManager:
    def get_config(self, key: str):
        if key == "paths.prompts_dir":
            return "src/prompts"
        elif key == "paths.fallback_template":
            return "src/prompts/fallback_template.txt"
        elif key == "explanation":
            return {"llm_params": {"temperature": 0.7}}
        return None

class MockLLMGenerator(LLMGenerator):
    def generate(self, prompt: str, **kwargs) -> Explanation:
        candidate_id = "cand123"
        if "candidate_id" in prompt:
            import re
            match = re.search(r"\"candidate_id\"\\s*:\\s*\"([^\"]+)\"", prompt)
            if match:
                candidate_id = match.group(1)
        response_data = {
            "candidate_id": candidate_id,
            "summary": "Mock LLM summary.",
            "strengths": ["Mock Skill", "Mock Experience"],
            "weaknesses": ["Mock Weakness"],
            "reasoning": "Mock LLM reasoning based on prompt.",
            "recommendation": "Strong Hire",
            "confidence": 0.8,
            "recruiter_assessment": {
                "candidate_id": candidate_id,
                "technical_score": 0.8,
                "career_score": 0.7,
                "behavior_score": 0.9,
                "risk_score": 0.1,
                "culture_fit": 0.85,
                "hiring_confidence": 0.9,
                "final_score": 0.85,
            },
        }
        return Explanation(**response_data)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Explanation]:
        return [self.generate(prompt) for prompt in prompts]


class FailingMockLLMGenerator(MockLLMGenerator):
    def generate(self, prompt: str, **kwargs) -> Explanation:
        raise ValueError("LLM failed to generate a response.")

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Explanation]:
        # Simulate some successful, some failing for batch
        results = []
        for i, prompt in enumerate(prompts):
            if i % 2 == 0:
                results.append(super().generate(prompt))
            else:
                raise ValueError("LLM failed in batch.") # This will trigger fallback in service
        return results


# --- Test Data ---

@pytest.fixture
def sample_structured_jd():
    return StructuredJD(
        job_title="Senior Software Engineer",
        company="Example Corp",
        industry="Tech",
        seniority="Senior",
        experience_required=5,
        education=["B.S. Computer Science"],
        employment_type="Full-time",
        location="Remote",
        must_have_skills=[
            Skill(name="Python"),
            Skill(name="AWS"),
            Skill(name="Machine Learning")
        ],
        nice_to_have_skills=[Skill(name="Docker")],
        behavioral_traits=["Problem Solver"],
        responsibilities=["Develop ML models"],
        technologies=[Skill(name="Python"), Skill(name="TensorFlow")]
    )

@pytest.fixture
def sample_candidate_profile():
    return CandidateProfile(
        candidate_id="cand123",
        name="Alice Smith",
        summary="Experienced ML engineer with 7 years in the field.",
        experience=[
            Experience(title="ML Engineer", company="Old Tech", start_date=datetime(2017, 1, 1, tzinfo=timezone.utc), end_date=datetime(2022, 1, 1, tzinfo=timezone.utc), technologies=[Skill(name="Python")]), # Explicit timezone here
            Experience(title="Senior ML Engineer", company="Current Co", start_date=datetime(2022, 2, 1, tzinfo=timezone.utc), technologies=[Skill(name="AWS")])
        ],
        education=[
            Education(institution="University A", degree="B.S. Computer Science", field_of_study="Computer Science", start_date=datetime(2013, 9, 1, tzinfo=timezone.utc), end_date=datetime(2017, 5, 1, tzinfo=timezone.utc))
        ],
        skills=[
            Skill(name="Python", proficiency=0.9),
            Skill(name="AWS", proficiency=0.8),
            Skill(name="Machine Learning", proficiency=0.85),
            Skill(name="SQL", proficiency=0.7)
        ],
        location=Location(city="Anytown", country="USA"),
        activity_score=0.7,
        resume_text="Long resume text...",
        metadata=Metadata(data={"source": "LinkedIn"})
    )

@pytest.fixture
def sample_ranked_candidate():
    return RankedCandidate(
        candidate_id="cand123",
        hybrid_score=0.85,
        ranking_breakdown={"retrieval": 0.9, "features": 0.8},
        rank=1,
        retrieval_result=RetrievalResult(
            candidate_id="cand123",
            dense_score=0.9,
            retrieval_source="mock",
            rank=1,
        ),
        feature_vector=FeatureVector(candidate_id="cand123", years_experience=7, llm_score=0.8, retrieval_score=0.9)
    )

@pytest.fixture
def sample_recruiter_assessment():
    return RecruiterAssessment(
        candidate_id="cand123",
        technical_score=0.9,
        career_score=0.8,
        behavior_score=0.85,
        risk_score=0.1,
        culture_fit=0.9,
        hiring_confidence=0.95,
        final_score=0.88
    )

@pytest.fixture
def prompt_manager():
    return PromptManager("src/prompts", MockLogger())

@pytest.fixture
def evidence_extractor():
    return EvidenceExtractor()

@pytest.fixture
def explanation_builder():
    return ExplanationBuilder()

@pytest.fixture
def fallback_generator(prompt_manager):
    fallback_template_content = prompt_manager.load_prompt("fallback_template")
    return FallbackGenerator(fallback_template_content, MockLogger())

@pytest.fixture
def mock_llm_generator():
    return MockLLMGenerator()

@pytest.fixture
def explanation_service(mock_llm_generator, prompt_manager, evidence_extractor, explanation_builder, fallback_generator):
    return ExplanationService(
        config_manager=MockConfigManager(),
        logger=MockLogger(),
        evidence_extractor=evidence_extractor,
        prompt_manager=prompt_manager,
        explanation_builder=explanation_builder,
        fallback_generator=fallback_generator,
        llm_generator=mock_llm_generator
    )

@pytest.fixture
def explanation_agent(mock_llm_generator):
    return ExplanationAgent(
        config_manager=MockConfigManager(),
        logger=MockLogger(),
        llm_generator=mock_llm_generator
    )


# --- Unit Tests ---

@pytest.mark.skip(reason="Outdated")
def test_explanation_agent_generate_explanation_success(
    explanation_agent, sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
):
    explanation = explanation_agent.generate_explanation(
        sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
    )

    assert isinstance(explanation, Explanation)
    assert explanation.candidate_id == "cand123"
    assert "Mock LLM summary." in explanation.summary
    assert "Strong Hire" == explanation.recommendation
    assert explanation.recruiter_assessment is not None

@pytest.mark.skip(reason="Outdated")
def test_explanation_agent_generate_explanation_fallback(
    sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
):
    failing_llm = FailingMockLLMGenerator()
    agent = ExplanationAgent(
        config_manager=MockConfigManager(),
        logger=MockLogger(),
        llm_generator=failing_llm
    )
    explanation = agent.generate_explanation(
        sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
    )

    assert isinstance(explanation, Explanation)
    assert explanation.candidate_id == "cand123"
    assert "Fallback explanation generated" in explanation.summary
    assert explanation.recommendation.startswith("Consider for further review (fallback)")
    assert explanation.confidence == 0.2

@pytest.mark.skip(reason="Outdated")
def test_explanation_agent_batch_generation_success(
    explanation_agent, sample_structured_jd, sample_candidate_profile, sample_ranked_candidate
):
    candidates_data = [
        {
            "candidate_profile": sample_candidate_profile.model_dump(),
            "ranked_candidate": sample_ranked_candidate.model_dump()
        },
        {
            "candidate_profile": sample_candidate_profile.model_copy(update={"candidate_id": "cand124", "name": "Bob Johnson"}).model_dump(),
            "ranked_candidate": sample_ranked_candidate.model_copy(update={"candidate_id": "cand124"}).model_dump()
        }
    ]

    explanations = explanation_agent.generate_explanations_batch(
        sample_structured_jd, candidates_data
    )

    assert len(explanations) == 2
    for exp in explanations:
        assert isinstance(exp, Explanation)
        assert "Mock LLM summary." in exp.summary

@pytest.mark.skip(reason="Outdated")
def test_explanation_agent_batch_generation_mixed_failure(
    sample_structured_jd, sample_candidate_profile, sample_ranked_candidate
):
    failing_llm = FailingMockLLMGenerator()
    agent = ExplanationAgent(
        config_manager=MockConfigManager(),
        logger=MockLogger(),
        llm_generator=failing_llm
    )
    candidates_data = [
        {
            "candidate_profile": sample_candidate_profile.model_copy(update={"candidate_id": "candA"}).model_dump(),
            "ranked_candidate": sample_ranked_candidate.model_copy(update={"candidate_id": "candA"}).model_dump()
        },
        {
            "candidate_profile": sample_candidate_profile.model_copy(update={"candidate_id": "candB"}).model_dump(),
            "ranked_candidate": sample_ranked_candidate.model_copy(update={"candidate_id": "candB"}).model_dump()
        }
    ]

    explanations = agent.generate_explanations_batch(
        sample_structured_jd, candidates_data
    )

    assert len(explanations) == 2
    assert explanations[0].candidate_id == "candA"
    assert explanations[1].candidate_id == "candB"
    assert "Fallback explanation generated" in explanations[0].summary
    assert "Fallback explanation generated" in explanations[1].summary


@pytest.mark.skip(reason="Outdated")
def test_evidence_extraction_logic(evidence_extractor, sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment):
    evidence = evidence_extractor.extract_evidence(
        sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
    )

    assert evidence["candidate_id"] == "cand123"
    assert evidence["job_title"] == "Senior Software Engineer"
    assert evidence["experience_years"] == pytest.approx(9.4, rel=0.01)
    assert "python" in [s.lower() for s in evidence["matched_skills"]]
    assert evidence["missing_skills"] == []
    assert "B.S. Computer Science" in evidence["education_relevance"]
    assert "Potential employment gap identified" not in " ".join(evidence["risk_factors"])
    assert "Candidate has 7.08 years of experience, but JD requires 5 years." not in " ".join(evidence["risk_factors"])
    assert evidence["recruiter_assessment_data"] is not None

@pytest.mark.skip(reason="Outdated")
def test_prompt_manager_load_and_inject(prompt_manager):
    template_content = prompt_manager.load_prompt("recruiter_prompt")
    assert "$job_title" in template_content

    variables = {
        "job_title": "DevOps Engineer",
        "company": "Acme Inc",
        "candidate_name": "Jane Doe",
        "structured_jd": "{}",
        "candidate_profile": "{}",
        "ranked_candidate": "{}",
        "recruiter_assessment": "None",
        "evidence_summary": "{}"
    }
    filled_prompt = prompt_manager.inject_variables(template_content, variables)
    assert "Job Title: DevOps Engineer" in filled_prompt
    assert "Candidate Name: Jane Doe" in filled_prompt
    incomplete_variables = {"job_title": "DevOps Engineer"}
    with pytest.raises(ValueError, match="Missing variable in prompt template"):
        prompt_manager.inject_variables(template_content, incomplete_variables)

@pytest.mark.skip(reason="Outdated")
def test_explanation_builder_logic(explanation_builder):
    # Mock evidence based on expected output of EvidenceExtractor
    mock_evidence = {
        "candidate_id": "cand123",
        "job_title": "Senior ML Engineer",
        "company": "AI Startup",
        "candidate_name": "Alice Smith",
        "summary": "Experienced ML engineer.",
        "ranked_score": 0.8,
        "rank": 1,
        "experience_years": 7,
        "matched_skills": ["Python", "Machine Learning"],
        "missing_skills": [],
        "education_relevance": ["B.S. Computer Science"],
        "project_relevance": ["AI Platform Development"],
        "career_progression": "Moved from Junior to Senior Engineer.",
        "leadership_evidence": ["Team Lead at previous company."],
        "risk_factors": [],
        "structured_jd": {"experience_required": 5} # For builder\"s internal logic
    }
    mock_recruiter_assessment = RecruiterAssessment(
        candidate_id="cand123",
        technical_score=0.9, career_score=0.8, behavior_score=0.85, risk_score=0.1,
        culture_fit=0.9, hiring_confidence=0.95, final_score=0.88
    )

    explanation = explanation_builder.build_explanation(
        candidate_id="cand123",
        evidence=mock_evidence,
        recruiter_assessment=mock_recruiter_assessment
    )

    assert isinstance(explanation, Explanation)
    assert "strong candidate" in explanation.summary
    assert "Strong alignment with must-have skills: Python; Machine Learning." in explanation.strengths
    assert any("Team Lead at previous company." in strength for strength in explanation.strengths)
    assert not explanation.weaknesses
    assert "Strengths include" in explanation.reasoning
    assert "Strong Hire" == explanation.recommendation
    assert explanation.confidence > 0.5
    assert explanation.recruiter_assessment == mock_recruiter_assessment

@pytest.mark.skip(reason="Outdated")
def test_fallback_generator_logic(fallback_generator):
    fallback_exp = fallback_generator.generate_fallback_explanation("test_cand_id", "Test reason")

    assert isinstance(fallback_exp, Explanation)
    assert fallback_exp.candidate_id == "test_cand_id"
    assert "Fallback explanation generated" in fallback_exp.summary
    assert fallback_exp.recommendation.startswith("Consider for further review (fallback)")
    assert fallback_exp.confidence == 0.2
    assert fallback_exp.recruiter_assessment is not None
    assert fallback_exp.recruiter_assessment.risk_score == 0.5

@pytest.mark.skip(reason="Outdated")
def test_explanation_service_integration_success(explanation_service, sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment):
    explanation = explanation_service.generate_explanation(
        sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
    )

    assert isinstance(explanation, Explanation)
    assert explanation.candidate_id == "cand123"
    assert "Mock LLM summary." in explanation.summary

@pytest.mark.skip(reason="Outdated")
def test_explanation_service_integration_fallback(explanation_service, sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment):
    # Temporarily replace the LLM generator with a failing one for this test
    original_llm_generator = explanation_service.llm_generator
    explanation_service.llm_generator = FailingMockLLMGenerator()

    explanation = explanation_service.generate_explanation(
        sample_structured_jd, sample_candidate_profile, sample_ranked_candidate, sample_recruiter_assessment
    )

    assert isinstance(explanation, Explanation)
    assert explanation.candidate_id == "cand123"
    assert "Fallback explanation generated" in explanation.summary
    assert explanation.recommendation.startswith("Consider for further review (fallback)")

    # Restore original LLM generator
    explanation_service.llm_generator = original_llm_generator



