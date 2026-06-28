import logging
import time
from typing import Any, Type

from src.interfaces.adapter_interfaces import (
    BM25Adapter,
    DatasetLoaderAdapter,
    DenseRetrievalAdapter,
    FeatureEngineeringAdapter,
    HybridRankerAdapter,
    JDParserAdapter,
    ProfileBuilderAdapter,
    RecruiterRerankerAdapter,
    SkillExtractorAdapter,
    VectorStoreAdapter,
)
from src.models.domain_models import Candidate, JobDescription, RankedCandidate, SearchResult
from src.schemas.graph_schema import StructuredJD
from src.state.pipeline_state import PipelineState

logger = logging.getLogger(__name__)


class BaseNodeAdapter:
    """Base class for concrete node adapters with shared validation helpers."""

    def __init__(self, module: Any):
        self._module = module

    def _validate_inputs(self, state: PipelineState, expected_inputs: list[str]) -> None:
        for input_name in expected_inputs:
            if state.get(input_name) is None:
                raise ValueError(
                    f"Missing required input '{input_name}' in PipelineState for {self.__class__.__name__}."
                )
        logger.debug("Inputs validated successfully for %s.", self.__class__.__name__)

    def _validate_outputs(self, outputs: Any, expected_type: Type) -> None:
        if not isinstance(outputs, expected_type):
            raise TypeError(
                f"Output type mismatch for {self.__class__.__name__}. "
                f"Expected {expected_type}, got {type(outputs)}."
            )
        logger.debug("Outputs validated successfully for %s.", self.__class__.__name__)

    def _log_execution_time(self, start_time: float, end_time: float, module_name: str) -> None:
        execution_time = (end_time - start_time) * 1000
        logger.info("%s adapter executed in %.2f ms.", module_name, execution_time)


class ConcreteJDParserAdapter(JDParserAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["raw_jd"])

        parsed_jd_dict = self._module.parse_jd(state["raw_jd"])
        self._validate_outputs(parsed_jd_dict, dict)

        structured = JobDescription(
            title=parsed_jd_dict.get("title", "Unknown"),
            raw_text=state["raw_jd"],
            key_responsibilities=parsed_jd_dict.get("requirements", []),
        )
        updated: PipelineState = {
            **state,
            "structured_jd": structured.model_dump(mode="json"),
            "extracted_skills": parsed_jd_dict.get("requirements", []),
        }

        self._log_execution_time(start_time, time.perf_counter(), "JDParserAgent")
        return updated


class ConcreteSkillExtractorAdapter(SkillExtractorAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["structured_jd", "candidate_profiles"])

        structured_jd = StructuredJD.model_validate(state["structured_jd"])
        jd_skills = self._module.extract_skills_from_jd(structured_jd.raw_text)
        candidate_profiles: dict[str, Candidate] = state["candidate_profiles"]
        candidate_skills = {
            cid: self._module.extract_skills_from_profile(profile.resume_text or "")
            for cid, profile in candidate_profiles.items()
        }

        self._validate_outputs(jd_skills, list)
        self._validate_outputs(candidate_skills, dict)

        updated_profiles = dict(candidate_profiles)
        for cid, skills in candidate_skills.items():
            if cid in updated_profiles:
                profile = updated_profiles[cid]
                updated_profiles[cid] = profile.model_copy(update={"skills": skills})

        updated: PipelineState = {
            **state,
            "extracted_skills": jd_skills,
            "candidate_profiles": updated_profiles,
        }
        self._log_execution_time(start_time, time.perf_counter(), "SkillExtractorAgent")
        return updated


class ConcreteProfileBuilderAdapter(ProfileBuilderAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["raw_candidate_data"])

        candidate_profiles: dict[str, Candidate] = self._module.build_profiles(state["raw_candidate_data"])
        self._validate_outputs(candidate_profiles, dict)
        for profile in candidate_profiles.values():
            self._validate_outputs(profile, Candidate)

        updated: PipelineState = {**state, "candidate_profiles": candidate_profiles}
        self._log_execution_time(start_time, time.perf_counter(), "ProfileBuilderAgent")
        return updated


class ConcreteDenseRetrievalAdapter(DenseRetrievalAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["structured_jd", "candidate_profiles"])

        structured_jd = StructuredJD.model_validate(state["structured_jd"])
        candidates_to_search = list(state["candidate_profiles"].values())
        search_results: list[SearchResult] = self._module.dense_retrieve(
            structured_jd.raw_text, candidates_to_search
        )

        self._validate_outputs(search_results, list)
        for result in search_results:
            self._validate_outputs(result, SearchResult)

        existing = list(state.get("retrieved_candidates", []))
        updated: PipelineState = {
            **state,
            "retrieved_candidates": existing + [r.model_dump(mode="json") for r in search_results],
        }
        self._log_execution_time(start_time, time.perf_counter(), "DenseRetrievalAgent")
        return updated


class ConcreteBM25Adapter(BM25Adapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["structured_jd", "candidate_profiles"])

        structured_jd = StructuredJD.model_validate(state["structured_jd"])
        candidates_to_search = list(state["candidate_profiles"].values())
        search_results: list[SearchResult] = self._module.bm25_retrieve(
            structured_jd.raw_text, candidates_to_search
        )

        self._validate_outputs(search_results, list)
        existing = list(state.get("retrieved_candidates", []))
        updated: PipelineState = {
            **state,
            "retrieved_candidates": existing + [r.model_dump(mode="json") for r in search_results],
        }
        self._log_execution_time(start_time, time.perf_counter(), "BM25Agent")
        return updated


class ConcreteDatasetLoaderAdapter(DatasetLoaderAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        raw_data = self._module.load_dataset()
        self._validate_outputs(raw_data, list)
        updated: PipelineState = {**state, "raw_candidate_data": raw_data}
        self._log_execution_time(start_time, time.perf_counter(), "DatasetLoader")
        return updated


class ConcreteVectorStoreAdapter(VectorStoreAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["candidate_profiles"])
        self._module.index_candidates(list(state["candidate_profiles"].values()))
        self._log_execution_time(start_time, time.perf_counter(), "VectorStore")
        return state


class ConcreteFeatureEngineeringAdapter(FeatureEngineeringAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["retrieved_candidates", "structured_jd"])

        structured_jd = StructuredJD.model_validate(state["structured_jd"])
        candidates_with_features = self._module.engineer_features(
            state["retrieved_candidates"], structured_jd
        )
        self._validate_outputs(candidates_with_features, list)

        updated: PipelineState = {**state, "candidates_with_features": candidates_with_features}
        self._log_execution_time(start_time, time.perf_counter(), "FeatureEngineeringAgent")
        return updated


class ConcreteHybridRankerAdapter(HybridRankerAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["candidates_with_features"])

        structured_jd = StructuredJD.model_validate(state["structured_jd"])
        ranked_candidates: list[RankedCandidate] = self._module.rank_candidates(
            state["candidates_with_features"], structured_jd
        )

        self._validate_outputs(ranked_candidates, list)
        updated: PipelineState = {
            **state,
            "ranked_candidates": [rc.model_dump(mode="json") for rc in ranked_candidates],
        }
        self._log_execution_time(start_time, time.perf_counter(), "HybridRankerAgent")
        return updated


class ConcreteRecruiterRerankerAdapter(RecruiterRerankerAdapter, BaseNodeAdapter):
    def execute(self, state: PipelineState) -> PipelineState:
        start_time = time.perf_counter()
        self._validate_inputs(state, ["ranked_candidates"])

        final_ranked: list[RankedCandidate] = self._module.rerank_candidates(
            state["ranked_candidates"],
            state.get("user_preferences", {}),
        )

        self._validate_outputs(final_ranked, list)
        updated: PipelineState = {
            **state,
            "reranked_candidates": [rc.model_dump(mode="json") for rc in final_ranked],
        }
        self._log_execution_time(start_time, time.perf_counter(), "RecruiterRerankerAgent")
        return updated
