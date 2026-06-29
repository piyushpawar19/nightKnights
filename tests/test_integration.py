import logging
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

from graph.dependency_registry import get_dependency_registry
from graph.integration_manager import IntegrationManager
from graph.node_adapters import ConcreteJDParserAdapter
from interfaces.adapter_interfaces import (
    DenseRetrievalAdapter,
    HybridRankerAdapter,
    JDParserAdapter,
    ModuleAdapter,
)
from models.domain_models import Candidate, JobDescription, RankedCandidate, SearchResult
from schemas.retrieval_schema import RetrievalResult
from state.pipeline_state import PipelineState, create_initial_state
from utils.module_loader import ModuleLoader


class MockJDParserAgent:
    def parse_jd(self, raw_jd: str) -> dict:
        logging.info("MockJDParserAgent: Parsing JD.")
        return {"title": "Software Engineer", "description": raw_jd, "requirements": ["Python", "ML"]}


class MockDenseRetrievalAgent:
    def dense_retrieve(self, query: str, candidates: List[Candidate]) -> List[SearchResult]:
        logging.info("MockDenseRetrievalAgent: Performing dense retrieval.")
        return [
            RetrievalResult(
                candidate_id=c.candidate_id,
                dense_score=0.8,
                retrieval_source="mock_dense",
                rank=1,
            )
            for c in candidates[:1]
        ]


class MockHybridRankerAgent:
    def rank_candidates(self, candidates_with_features: List[Any], jd: JobDescription) -> List[RankedCandidate]:
        logging.info("MockHybridRankerAgent: Ranking candidates.")
        results = []
        for i, candidate in enumerate(candidates_with_features[:1]):
            candidate_id = getattr(candidate, "candidate_id", candidate.get("candidate_id"))
            retrieval = RetrievalResult(
                candidate_id=candidate_id,
                dense_score=0.8,
                retrieval_source="mock_dense",
                rank=1,
            )
            results.append(
                RankedCandidate(
                    candidate_id=candidate_id,
                    rank=i + 1,
                    hybrid_score=0.9 - i * 0.1,
                    retrieval_result=retrieval,
                )
            )
        return results


class IntegrationJDParserAdapter(JDParserAdapter):
    def __init__(self, module: Any):
        self._adapter = ConcreteJDParserAdapter(module)

    def execute(self, state: PipelineState) -> PipelineState:
        return self._adapter.execute(state)


class IntegrationDenseRetrievalAdapter(DenseRetrievalAdapter):
    def __init__(self, module: Any):
        self._module = module

    def execute(self, state: PipelineState) -> PipelineState:
        from graph.node_adapters import ConcreteDenseRetrievalAdapter

        return ConcreteDenseRetrievalAdapter(self._module).execute(state)


class IntegrationHybridRankerAdapter(HybridRankerAdapter):
    def __init__(self, module: Any):
        self._module = module

    def execute(self, state: PipelineState) -> PipelineState:
        from graph.node_adapters import ConcreteHybridRankerAdapter

        return ConcreteHybridRankerAdapter(self._module).execute(state)


def _sample_candidate(candidate_id: str, resume_text: str) -> Candidate:
    return Candidate(candidate_id=candidate_id, name=f"Candidate {candidate_id}", resume_text=resume_text)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        if hasattr(get_dependency_registry, "_instance"):
            delattr(get_dependency_registry, "_instance")
        self.registry = get_dependency_registry()
        self.integration_manager = IntegrationManager()
        self.module_loader = ModuleLoader()

        self.module_loader.set_mock_implementation("jd_parser_agent", MockJDParserAgent())
        self.module_loader.set_mock_implementation("dense_retrieval_agent", MockDenseRetrievalAgent())
        self.module_loader.set_mock_implementation("hybrid_ranker_agent", MockHybridRankerAgent())

        self.integration_manager.register_adapter("jd_parser_agent", IntegrationJDParserAdapter)
        self.registry.register_service("JDParser", IntegrationJDParserAdapter)
        self.integration_manager.register_adapter("dense_retrieval_agent", IntegrationDenseRetrievalAdapter)
        self.registry.register_service("DenseRetriever", IntegrationDenseRetrievalAdapter)
        self.integration_manager.register_adapter("hybrid_ranker_agent", IntegrationHybridRankerAdapter)
        self.registry.register_service("HybridRanker", IntegrationHybridRankerAdapter)

        mock_jd_parser = self.module_loader.load_module("jd_parser_agent", "src.agents", use_mock=True)
        self.integration_manager.register_module("jd_parser_agent", mock_jd_parser, is_mock=True)

        mock_dense_retrieval = self.module_loader.load_module("dense_retrieval_agent", "src.retrieval", use_mock=True)
        self.integration_manager.register_module("dense_retrieval_agent", mock_dense_retrieval, is_mock=True)

        mock_hybrid_ranker = self.module_loader.load_module("hybrid_ranker_agent", "src.ranking", use_mock=True)
        self.integration_manager.register_module("hybrid_ranker_agent", mock_hybrid_ranker, is_mock=True)

    def _build_initial_state(self) -> PipelineState:
        state = create_initial_state("We need a skilled Python ML Engineer.")
        state["candidate_profiles"] = {
            "1": _sample_candidate("1", "Python developer with ML experience."),
            "2": _sample_candidate("2", "Java developer."),
        }
        state["candidates_with_features"] = [
            {"candidate_id": "1", "features": {"score": 0.9}},
            {"candidate_id": "2", "features": {"score": 0.5}},
        ]
        state["user_preferences"] = {"rerank_bias": "python_skills"}
        return state

    def test_module_registration_and_fallback(self):
        jd_parser_module = self.integration_manager.get_module("jd_parser_agent")
        self.assertIsNotNone(jd_parser_module)
        self.assertIsInstance(jd_parser_module, MockJDParserAgent)
        self.assertIsNone(self.integration_manager.get_module("non_existent_module"))

    def test_adapter_lookup_and_execution(self):
        state = self._build_initial_state()

        jd_parser_adapter = self.integration_manager.get_adapter("jd_parser_agent")
        self.assertIsNotNone(jd_parser_adapter)
        state = jd_parser_adapter.execute(state)
        self.assertIsNotNone(state.get("structured_jd"))
        self.assertEqual(state["structured_jd"]["title"], "Software Engineer")

        dense_retrieval_adapter = self.integration_manager.get_adapter("dense_retrieval_agent")
        self.assertIsNotNone(dense_retrieval_adapter)
        state = dense_retrieval_adapter.execute(state)
        self.assertGreater(len(state["retrieved_candidates"]), 0)
        self.assertEqual(state["retrieved_candidates"][0]["candidate_id"], "1")

        hybrid_ranker_adapter = self.integration_manager.get_adapter("hybrid_ranker_agent")
        self.assertIsNotNone(hybrid_ranker_adapter)
        state = hybrid_ranker_adapter.execute(state)
        self.assertGreater(len(state["ranked_candidates"]), 0)
        self.assertEqual(state["ranked_candidates"][0]["candidate_id"], "1")

    def test_registry_lookup(self):
        jd_parser_adapter_class = self.registry.get_service_adapter("JDParser")
        self.assertIsNotNone(jd_parser_adapter_class)
        self.assertEqual(jd_parser_adapter_class, IntegrationJDParserAdapter)

        mock_jd_parser = self.integration_manager.get_module("jd_parser_agent")
        jd_parser_adapter_instance = jd_parser_adapter_class(mock_jd_parser)
        self.assertIsInstance(jd_parser_adapter_instance, JDParserAdapter)

    def test_dynamic_loading_and_mock_fallback(self):
        mock_module = self.module_loader.load_module("jd_parser_agent", "non.existent.path", use_mock=True)
        self.assertIsInstance(mock_module, MockJDParserAgent)
        self.assertIsNone(self.module_loader.load_module("non_existent_module", "non.existent.path"))

    @patch("src.graph.integration_manager.logger")
    def test_compatibility_validation_logging(self, mock_logger):
        manager = IntegrationManager()

        class IncompatibleModule:
            def incompatible_method(self):
                pass

        manager.register_module("incompatible_mod", IncompatibleModule())
        is_compatible = manager.validate_module_interface("incompatible_mod", {"parse_jd": None})
        self.assertFalse(is_compatible)
        mock_logger.error.assert_called_with(
            "Module \'incompatible_mod\' is missing expected method \'parse_jd\' or it\'s not callable."
        )

        manager.register_module("compatible_mod", MockJDParserAgent())
        self.assertTrue(manager.validate_module_interface("compatible_mod", {"parse_jd": None}))

    def test_error_handling_in_adapter_inputs(self):
        jd_parser_adapter = self.integration_manager.get_adapter("jd_parser_agent")
        with self.assertRaisesRegex(ValueError, "Missing required input \'raw_jd\'"):
            jd_parser_adapter.execute({})  # type: ignore[arg-type]

    @patch.object(MockJDParserAgent, "parse_jd", side_effect=TypeError("Mock parsing error"))
    def test_error_handling_in_module_execution(self, mock_parse_jd):
        jd_parser_adapter = self.integration_manager.get_adapter("jd_parser_agent")
        with self.assertRaisesRegex(TypeError, "Mock parsing error"):
            jd_parser_adapter.execute(create_initial_state("some raw text"))

    def test_error_handling_in_adapter_outputs(self):
        jd_parser_adapter = self.integration_manager.get_adapter("jd_parser_agent")
        original_parse_jd = MockJDParserAgent.parse_jd
        MockJDParserAgent.parse_jd = lambda self, raw_jd: "wrong type string"
        try:
            with self.assertRaisesRegex(TypeError, "Output type mismatch for ConcreteJDParserAdapter"):
                jd_parser_adapter.execute(create_initial_state("some raw text"))
        finally:
            MockJDParserAgent.parse_jd = original_parse_jd

    def test_lifecycle_management_placeholders(self):
        with self.assertLogs("src.graph.integration_manager", level="INFO") as cm:
            self.integration_manager.lifecycle_init()
            self.assertTrue(any("Initializing modules" in line for line in cm.output))

        with self.assertLogs("src.graph.integration_manager", level="INFO") as cm:
            self.integration_manager.lifecycle_shutdown()
            self.assertTrue(any("Shutting down modules" in line for line in cm.output))

    @patch("src.utils.module_loader.logger")
    def test_module_loader_logging(self, mock_logger):
        loader = ModuleLoader()
        loader.set_mock_implementation("test_mod_exists", MagicMock())
        loader.load_module("test_mod_exists", "any.path", use_mock=True)
        mock_logger.warning.assert_called_with("Loading mock implementation for \'test_mod_exists\'.")
        loader.load_module("non_existent", "any.path")
        mock_logger.error.assert_called_with("No mock implementation available for \'non_existent\'.")


if __name__ == "__main__":
    unittest.main()
