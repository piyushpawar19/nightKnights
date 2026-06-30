
import unittest
import json
from unittest.mock import Mock, patch
from pydantic import ValidationError

from nightKnights.src.schemas.reranking_schema import RecruiterAssessment, RerankedCandidate, RerankedCandidates
from nightKnights.src.interfaces.reranking_interface import LLMInterface, PromptBuilderInterface, ResponseParserInterface
from nightKnights.src.ranking.reranking_utils import normalize_score, stable_sort_candidates, load_ranking_weights
from nightKnights.src.ranking.recruiter_prompt_builder import RecruiterPromptBuilder
from nightKnights.src.ranking.recruiter_parser import RecruiterParser
from nightKnights.src.ranking.recruiter_reranker import RecruiterReranker
from nightKnights.src.agents.recruiter_reranker_agent import RecruiterRerankerAgent

class TestRecruiterReranker(unittest.TestCase):

    def setUp(self):
        self.mock_llm_interface = Mock(spec=LLMInterface)
        self.prompt_builder = RecruiterPromptBuilder()
        self.response_parser = RecruiterParser()
        self.recruiter_reranker = RecruiterReranker(
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
            config_path="tests/test_configs/ranking.yaml" # Use a test config
        )
        self.agent = RecruiterRerankerAgent(llm_interface=self.mock_llm_interface)

        # Mock data
        self.parsed_jd = {"title": "Software Engineer", "required_skills": ["Python", "SQL"]}
        self.candidate_features = {
            "cand1": {"skills": ["Python", "Java"], "experience": 5},
            "cand2": {"skills": ["SQL", "C++"], "experience": 3}
        }
        self.ranked_candidates = [
            {"candidate_id": "cand1", "rank": 1, "score": 0.8},
            {"candidate_id": "cand2", "rank": 2, "score": 0.6}
        ]
        self.valid_llm_response = json.dumps({
            "strengths": ["Strong Python skills"],
            "concerns": ["Less SQL experience"],
            "hiring_recommendation": "Yes",
            "confidence": 0.9,
            "recruiter_score": 0.85,
            "reasoning": "Good overall match but needs more SQL."
        })
        self.malformed_llm_response = "```json" + self.valid_llm_response + "```"
        self_invalid_llm_response_schema = json.dumps({
            "strengths": ["Strong Python skills"],
            "hiring_recommendation": "Yes",
            "confidence": 0.9,
            "recruiter_score": 0.85,
            "reasoning": "Good overall match but needs more SQL."
            # Missing 'concerns' field
        })
        self.invalid_score_llm_response = json.dumps({
            "strengths": ["Strong Python skills"],
            "concerns": ["Less SQL experience"],
            "hiring_recommendation": "Yes",
            "confidence": 1.1, # Invalid confidence
            "recruiter_score": 0.85,
            "reasoning": "Good overall match but needs more SQL."
        })
        self.empty_ranked_candidates = []

        # Create a dummy config file for testing
        self.test_config_dir = "nightKnights/tests/test_configs"
        self.test_config_path = f"{self.test_config_dir}/ranking.yaml"
        import os
        os.makedirs(self.test_config_dir, exist_ok=True)
        with open(self.test_config_path, "w") as f:
            f.write("reranker_weights:\n")
            f.write("  hybrid_score_weight: 0.6\n")
            f.write("  recruiter_score_weight: 0.4\n")

    def tearDown(self):
        # Clean up the dummy config file
        import os
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.test_config_dir):
            os.rmdir(self.test_config_dir)

    def test_normalize_score(self):
        self.assertAlmostEqual(normalize_score(0.5), 0.5)
        self.assertAlmostEqual(normalize_score(1.5, max_val=1.0), 1.0)
        self.assertAlmostEqual(normalize_score(-0.5), 0.0)

    def test_stable_sort_candidates(self):
        candidates = [
            {"candidate_id": "c1", "final_score": 0.8, "previous_rank": 1},
            {"candidate_id": "c2", "final_score": 0.7, "previous_rank": 2},
            {"candidate_id": "c3", "final_score": 0.8, "previous_rank": 3}, # Tie with c1, c3 should be after c1
        ]
        sorted_cands = stable_sort_candidates(candidates)
        self.assertEqual(sorted_cands[0]["candidate_id"], "c1")
        self.assertEqual(sorted_cands[0]["new_rank"], 1)
        self.assertEqual(sorted_cands[1]["candidate_id"], "c3")
        self.assertEqual(sorted_cands[1]["new_rank"], 2)
        self.assertEqual(sorted_cands[2]["candidate_id"], "c2")
        self.assertEqual(sorted_cands[2]["new_rank"], 3)

    def test_load_ranking_weights(self):
        weights = load_ranking_weights(self.test_config_path)
        self.assertEqual(weights["hybrid_score_weight"], 0.6)
        self.assertEqual(weights["recruiter_score_weight"], 0.4)

    def test_prompt_generation(self):
        prompt = self.prompt_builder.build_recruiter_prompt(
            self.parsed_jd, self.candidate_features, self.ranked_candidates
        )
        self.assertIsInstance(prompt, str)
        self.assertIn("Software Engineer", prompt)
        self.assertIn("Python", prompt)
        self.assertIn("cand1", prompt)

    def test_valid_assessment_parsing(self):
        assessment = self.response_parser.parse_recruiter_assessment(self.valid_llm_response)
        self.assertIsInstance(assessment, RecruiterAssessment)
        self.assertEqual(assessment.hiring_recommendation, "Yes")
        self.assertAlmostEqual(assessment.confidence, 0.9)

    def test_malformed_json_recovery(self):
        assessment = self.response_parser.parse_recruiter_assessment(self.malformed_llm_response)
        self.assertIsInstance(assessment, RecruiterAssessment)
        self.assertEqual(assessment.hiring_recommendation, "Yes")

    def test_invalid_schema_parsing(self):
        with self.assertRaises(ValueError):
            self.response_parser.parse_recruiter_assessment(self_invalid_llm_response_schema)

    def test_invalid_score_validation(self):
        with self.assertRaises(ValueError):
            self.response_parser.parse_recruiter_assessment(self.invalid_score_llm_response)

    def test_reranker_empty_shortlist(self):
        reranked = self.recruiter_reranker.rerank_candidates(
            self.parsed_jd, self.empty_ranked_candidates, self.candidate_features, self.mock_llm_interface
        )
        self.assertIsInstance(reranked, RerankedCandidates)
        self.assertEqual(len(reranked.candidates), 0)

    @patch("nightKnights.src.ranking.reranking_utils.load_ranking_weights", return_value={
        "hybrid_score_weight": 0.6,
        "recruiter_score_weight": 0.4
    })
    def test_reranking_logic(self, mock_load_weights):
        self.mock_llm_interface.invoke.return_value = self.valid_llm_response

        reranked_candidates_pydantic = self.recruiter_reranker.rerank_candidates(
            self.parsed_jd, self.ranked_candidates, self.candidate_features, self.mock_llm_interface
        )
        
        self.assertIsInstance(reranked_candidates_pydantic, RerankedCandidates)
        self.assertEqual(len(reranked_candidates_pydantic.candidates), 2)
        self.assertEqual(reranked_candidates_pydantic.candidates[0].candidate_id, "cand1")
        self.assertAlmostEqual(reranked_candidates_pydantic.candidates[0].recruiter_score, 0.85)
        
        # Verify final score calculation (0.8 * 0.6 + 0.85 * 0.4 = 0.48 + 0.34 = 0.82)
        self.assertAlmostEqual(reranked_candidates_pydantic.candidates[0].final_score, 0.82)
        self.assertEqual(reranked_candidates_pydantic.candidates[0].new_rank, 1)

        self.assertEqual(reranked_candidates_pydantic.candidates[1].candidate_id, "cand2")
        # For cand2, let's assume LLM returns same assessment for simplicity, though in reality it would be different
        # Expected final score for cand2: (0.6 * 0.6 + 0.85 * 0.4) = 0.36 + 0.34 = 0.70
        self.assertAlmostEqual(reranked_candidates_pydantic.candidates[1].final_score, 0.70)
        self.assertEqual(reranked_candidates_pydantic.candidates[1].new_rank, 2)

    def test_agent_run_success(self):
        self.mock_llm_interface.invoke.return_value = self.valid_llm_response
        initial_state = {
            "parsed_jd": self.parsed_jd,
            "ranked_candidates": self.ranked_candidates,
            "candidate_features": self.candidate_features
        }
        updated_state = self.agent.run(initial_state)

        self.assertIn("reranked_candidates", updated_state)
        reranked_cands_dict = updated_state["reranked_candidates"]
        self.assertIsInstance(reranked_cands_dict, dict)
        self.assertIn("candidates", reranked_cands_dict)
        self.assertEqual(len(reranked_cands_dict["candidates"]), 2)
        self.assertEqual(reranked_cands_dict["candidates"][0]["candidate_id"], "cand1")
        self.assertEqual(reranked_cands_dict["candidates"][0]["new_rank"], 1)

    def test_agent_run_missing_inputs(self):
        initial_state = {
            "parsed_jd": self.parsed_jd,
            "ranked_candidates": self.ranked_candidates,
            # "candidate_features": self.candidate_features # Missing this key
        }
        with self.assertRaises(ValueError) as cm:
            self.agent.run(initial_state)
        self.assertIn("Missing critical input keys in state: candidate_features", str(cm.exception))

    def test_json_serialization(self):
        assessment_data = json.loads(self.valid_llm_response)
        assessment = RecruiterAssessment(**assessment_data)
        serialized_assessment = assessment.json()
        self.assertIsInstance(serialized_assessment, str)
        deserialized_assessment = RecruiterAssessment.parse_raw(serialized_assessment)
        self.assertEqual(assessment, deserialized_assessment)

        reranked_cand_data = {
            "candidate_id": "cand1",
            "previous_rank": 1,
            "new_rank": 1,
            "previous_score": 0.8,
            "recruiter_score": 0.85,
            "final_score": 0.82,
            "assessment": assessment.dict()
        }
        reranked_candidate = RerankedCandidate(**reranked_cand_data)
        serialized_reranked_candidate = reranked_candidate.json()
        self.assertIsInstance(serialized_reranked_candidate, str)
        deserialized_reranked_candidate = RerankedCandidate.parse_raw(serialized_reranked_candidate)
        self.assertEqual(reranked_candidate, deserialized_reranked_candidate)

        reranked_candidates_data = {"candidates": [reranked_cand_data]}
        reranked_candidates = RerankedCandidates(**reranked_candidates_data)
        serialized_reranked_candidates = reranked_candidates.json()
        self.assertIsInstance(serialized_reranked_candidates, str)
        deserialized_reranked_candidates = RerankedCandidates.parse_raw(serialized_reranked_candidates)
        self.assertEqual(reranked_candidates, deserialized_reranked_candidates)

if __name__ == "__main__":
    unittest.main()
