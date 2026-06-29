import pytest
from datetime import datetime
import uuid

from schemas.common_schema import (
    Skill, Education, Experience, Project, Certification, Location,
    Company, Timestamp, Score, Metadata, UUIDMixin
)
from schemas.jd_schema import StructuredJD, SkillTaxonomy
from schemas.candidate_schema import CandidateProfile
from schemas.retrieval_schema import RetrievalResult
from schemas.ranking_schema import FeatureVector, RankedCandidate
from schemas.explanation_schema import Explanation, RecruiterAssessment
from schemas.evaluation_schema import EvaluationMetrics, EvaluationReport
from schemas.submission_schema import SubmissionRecord


# --- Common Schema Tests ---

def test_metadata_schema():
    meta = Metadata(data={
        "source": "test",
        "version": 1
    })
    assert meta.model_dump() == {
        "data": {
            "source": "test",
            "version": 1
        }
    }
    assert Metadata.model_validate_json(meta.model_dump_json()).data["source"] == "test"

def test_skill_schema():
    skill = Skill(name="Python", proficiency=0.9)
    assert skill.name == "Python"
    assert skill.proficiency == 0.9
    with pytest.raises(ValueError):
        Skill(name="")
    with pytest.raises(ValueError):
        Skill(name="Python", proficiency=1.1)

def test_education_schema():
    edu = Education(institution="University X", degree="B.S. Computer Science", start_date=datetime(2010, 9, 1))
    assert edu.institution == "University X"
    assert edu.end_date is None

def test_experience_schema():
    exp = Experience(title="Software Engineer", company="Tech Corp", start_date=datetime(2015, 1, 1), end_date=datetime(2020, 1, 1))
    assert exp.title == "Software Engineer"
    with pytest.raises(ValueError):
        Experience(title="Dev", company="ABC", start_date=datetime(2020, 1, 1), end_date=datetime(2015, 1, 1))

def test_project_schema():
    proj = Project(name="AI Platform", technologies=[Skill(name="Python")])
    assert proj.name == "AI Platform"
    assert proj.technologies[0].name == "Python"

def test_certification_schema():
    cert = Certification(name="PMP", issuing_organization="PMI", issue_date=datetime(2022, 1, 1), expiration_date=datetime(2023, 1, 1))
    assert cert.name == "PMP"
    with pytest.raises(ValueError):
        Certification(name="Cert", issuing_organization="Org", issue_date=datetime(2023, 1, 1), expiration_date=datetime(2022, 1, 1))

def test_location_schema():
    loc = Location(country="USA", city="New York")
    assert loc.country == "USA"

def test_company_schema():
    comp = Company(name="Google", industry="Tech")
    assert comp.name == "Google"

def test_timestamp_schema():
    ts = Timestamp()
    assert ts.created_at is not None
    assert ts.updated_at is not None

def test_score_schema():
    score = Score(value=0.8, metric="relevance")
    assert score.value == 0.8
    with pytest.raises(ValueError):
        Score(value=1.1, metric="bad")

def test_uuid_mixin_schema():
    item = UUIDMixin()
    assert isinstance(item.id, uuid.UUID)
    with pytest.raises(ValueError):
        UUIDMixin(id="invalid-uuid")
    valid_uuid = str(uuid.uuid4())
    item_with_id = UUIDMixin(id=valid_uuid)
    assert str(item_with_id.id) == valid_uuid
    assert UUIDMixin.model_validate_json(item_with_id.model_dump_json()).id == item_with_id.id


# --- JD Schema Tests ---

def test_skill_taxonomy_schema():
    taxonomy = SkillTaxonomy(normalized_skills=["Python", "Java"], confidence_scores={"Python": 0.9})
    assert "Python" in taxonomy.normalized_skills
    assert taxonomy.confidence_scores["Python"] == 0.9
    with pytest.raises(ValueError):
        SkillTaxonomy(normalized_skills=["Python"], confidence_scores={"Python": 1.1})

def test_structured_jd_schema():
    jd = StructuredJD(
        job_title="ML Engineer",
        company="AI Corp",
        experience_required=5,
        must_have_skills=[Skill(name="Python")]
    )
    assert jd.job_title == "ML Engineer"
    assert jd.experience_required == 5
    assert jd.must_have_skills[0].name == "Python"
    with pytest.raises(ValueError):
        StructuredJD(job_title="", company="AI Corp")
    with pytest.raises(ValueError):
        StructuredJD(job_title="ML Engineer", company="AI Corp", experience_required=-1)


# --- Candidate Schema Tests ---

def test_candidate_profile_schema():
    candidate = CandidateProfile(
        candidate_id="C001",
        name="John Doe",
        experience=[
            Experience(title="SDE", company="Google", start_date=datetime(2018, 1, 1))
        ],
        skills=[Skill(name="Java", proficiency=0.8)]
    )
    assert candidate.candidate_id == "C001"
    assert candidate.name == "John Doe"
    assert candidate.experience[0].company == "Google"
    assert candidate.skills[0].name == "Java"
    assert CandidateProfile.model_validate_json(candidate.model_dump_json()).candidate_id == "C001"


# --- Retrieval Schema Tests ---

def test_retrieval_result_schema():
    result = RetrievalResult(
        candidate_id="C002",
        dense_score=0.75,
        retrieval_source="vector_db",
        rank=1
    )
    assert result.candidate_id == "C002"
    assert result.dense_score == 0.75
    with pytest.raises(ValueError):
        RetrievalResult(candidate_id="C002", retrieval_source="vector_db", rank=0)


# --- Ranking Schema Tests ---

def test_feature_vector_schema():
    fv = FeatureVector(
        candidate_id="C003",
        years_experience=7,
        llm_score=0.9,
        normalized_vector=[0.1, 0.2, 0.7]
    )
    assert fv.candidate_id == "C003"
    assert fv.years_experience == 7
    with pytest.raises(ValueError):
        FeatureVector(candidate_id="C003", years_experience=-1)
    with pytest.raises(ValueError):
        FeatureVector(candidate_id="C003", llm_score=1.1)

def test_ranked_candidate_schema():
    retrieval = RetrievalResult(candidate_id="C004", retrieval_source="db", rank=1)
    ranked = RankedCandidate(
        candidate_id="C004",
        hybrid_score=0.92,
        rank=1,
        retrieval_result=retrieval
    )
    assert ranked.candidate_id == "C004"
    assert ranked.hybrid_score == 0.92
    assert ranked.retrieval_result.candidate_id == "C004"
    with pytest.raises(ValueError):
        RankedCandidate(candidate_id="C004", hybrid_score=1.1, rank=1, retrieval_result=retrieval)
    with pytest.raises(ValueError):
        RankedCandidate(candidate_id="C004", hybrid_score=0.5, rank=0, retrieval_result=retrieval)


# --- Explanation Schema Tests ---

def test_recruiter_assessment_schema():
    assessment = RecruiterAssessment(
        candidate_id="C004",
        technical_score=0.8,
        career_score=0.7,
        behavior_score=0.9,
        risk_score=0.1,
        culture_fit=0.85,
        hiring_confidence=0.95,
        final_score=0.85
    )
    assert assessment.technical_score == 0.8
    with pytest.raises(ValueError):
        RecruiterAssessment(
            candidate_id="C004",
            technical_score=1.1, career_score=0.5, behavior_score=0.5, risk_score=0.5,
            culture_fit=0.5, hiring_confidence=0.5, final_score=0.5
        )

def test_explanation_schema():
    explanation = Explanation(
        candidate_id="C005",
        summary="Strong candidate, good technical skills.",
        strengths=["Python", "Leadership"],
        weaknesses=["Limited experience in X"],
        reasoning="Candidate's profile aligns well with job requirements.",
        recommendation="Strong Hire",
        confidence=0.9
    )
    assert explanation.candidate_id == "C005"
    assert "Python" in explanation.strengths
    with pytest.raises(ValueError):
        Explanation(
            candidate_id="", summary="Test", strengths=[], weaknesses=[],
            reasoning="Test", recommendation="Test", confidence=0.5
        )


# --- Evaluation Schema Tests ---

def test_evaluation_metrics_schema():
    metrics = EvaluationMetrics(
        metric_name="Recall@5",
        value=0.85,
        unit="percentage"
    )
    assert metrics.metric_name == "Recall@5"
    assert metrics.value == 0.85
    with pytest.raises(ValueError):
        EvaluationMetrics(metric_name="", value=0.5)

def test_evaluation_report_schema():
    report = EvaluationReport(
        report_name="Q1 Performance",
        start_time="2023-01-01T00:00:00Z",
        end_time="2023-03-31T23:59:59Z",
        metrics=[
            EvaluationMetrics(metric_name="Precision@1", value=0.7),
            EvaluationMetrics(metric_name="Latency", value=150, unit="ms")
        ]
    )
    assert report.report_name == "Q1 Performance"
    assert len(report.metrics) == 2


# --- Submission Schema Tests ---

def test_submission_record_schema():
    record = SubmissionRecord(
        candidate_id="C006",
        rank=1,
        score=0.95,
        explanation_reference="/path/to/explanation.pdf"
    )
    assert record.candidate_id == "C006"
    assert record.rank == 1
    assert record.score == 0.95
    with pytest.raises(ValueError):
        SubmissionRecord(candidate_id="C006", rank=0, score=0.95)
    with pytest.raises(ValueError):
        SubmissionRecord(candidate_id="C006", rank=1, score=1.1)