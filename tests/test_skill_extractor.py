import pytest
import unittest
import datetime
from unittest.mock import MagicMock, patch
from src.schemas.skill_schema import ExtractedSkills
from src.preprocessing.normalization import SkillNormalization
from src.preprocessing.synonym_mapper import SynonymMapper
from src.preprocessing.taxonomy import SkillTaxonomy
from src.preprocessing.skill_extractor import SkillExtractor
from src.agents.skill_extractor_agent import SkillExtractorAgent

class TestSkillExtractorAgent(unittest.TestCase):

    def setUp(self):
        self.normalization_mock = MagicMock(spec=SkillNormalization)
        self.synonym_mapper_mock = MagicMock(spec=SynonymMapper)
        self.taxonomy_mock = MagicMock(spec=SkillTaxonomy)
        self.extractor_mock = MagicMock(spec=SkillExtractor)

        with patch("src.agents.skill_extractor_agent.SkillNormalization") as MockNormalization, \
             patch("src.agents.skill_extractor_agent.SynonymMapper") as MockSynonymMapper, \
             patch("src.agents.skill_extractor_agent.SkillTaxonomy") as MockTaxonomy, \
             patch("src.agents.skill_extractor_agent.SkillExtractor") as MockExtractor:

            MockNormalization.return_value = self.normalization_mock
            MockSynonymMapper.return_value = self.synonym_mapper_mock
            MockTaxonomy.return_value = self.taxonomy_mock
            MockExtractor.return_value = self.extractor_mock

            self.agent = SkillExtractorAgent()

    @pytest.mark.skip(reason="Outdated")
    def test_empty_jd(self):
        state = {"parsed_jd": {}}
        result_state = self.agent.run(state)
        self.assertIn("extracted_skills", result_state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        self.assertEqual(extracted_skills.calculate_total_skills(), 0)
        self.assertFalse(extracted_skills.metadata["normalization_applied"])
        self.assertEqual(extracted_skills.metadata["duplicate_skills_removed"], 0)

    @pytest.mark.skip(reason="Outdated")
    def test_duplicate_skills(self):
        self.extractor_mock.extract_from_parsed_jd.return_value = ["Python", "python", "PYTHON", "Java"]
        self.normalization_mock.normalize_skill.side_effect = lambda s: {
            "Python": "Python", "python": "Python", "PYTHON": "Python", "Java": "Java"
        }[s]
        self.synonym_mapper_mock.map_synonyms.side_effect = lambda s: s
        self.taxonomy_mock.categorize_skill.side_effect = lambda s: {"Python": "programming_languages", "Java": "programming_languages"}[s]

        state = {"parsed_jd": {"job_description": "Proficient in Python and Java. Strong Python skills required."}}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        self.assertIn("Python", extracted_skills.programming_languages)
        self.assertIn("Java", extracted_skills.programming_languages)
        self.assertEqual(len(extracted_skills.programming_languages), 2)
        self.assertEqual(extracted_skills.metadata["total_skills"], 2)
        self.assertEqual(extracted_skills.metadata["duplicate_skills_removed"], 2)
        self.assertTrue(extracted_skills.metadata["normalization_applied"])

    @pytest.mark.skip(reason="Outdated")
    def test_mixed_formatting_and_normalization(self):
        self.extractor_mock.extract_from_parsed_jd.return_value = ["js", "Node.JS", "Tensor flow", "k8s"]
        self.normalization_mock.normalize_skill.side_effect = lambda s: {
            "js": "JavaScript", "Node.JS": "Node.js", "Tensor flow": "TensorFlow", "k8s": "Kubernetes"
        }[s]
        self.synonym_mapper_mock.map_synonyms.side_effect = lambda s: s
        self.taxonomy_mock.categorize_skill.side_effect = lambda s: {
            "JavaScript": "programming_languages", "Node.js": "frameworks",
            "TensorFlow": "libraries", "Kubernetes": "devops_tools"
        }[s]

        state = {"parsed_jd": {"requirements": "Experience with js, Node.JS, Tensor flow, and k8s."}}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        self.assertIn("JavaScript", extracted_skills.programming_languages)
        self.assertIn("Node.js", extracted_skills.frameworks)
        self.assertIn("TensorFlow", extracted_skills.libraries)
        self.assertIn("Kubernetes", extracted_skills.devops_tools)
        self.assertEqual(extracted_skills.metadata["total_skills"], 4)
        self.assertEqual(extracted_skills.metadata["duplicate_skills_removed"], 0)
        self.assertTrue(extracted_skills.metadata["normalization_applied"])

    @pytest.mark.skip(reason="Outdated")
    def test_synonym_mapping(self):
        self.extractor_mock.extract_from_parsed_jd.return_value = ["ReactJS", "TF 2", "aws ec2"]
        self.normalization_mock.normalize_skill.side_effect = lambda s: {
            "ReactJS": "ReactJS", "TF 2": "TF 2", "aws ec2": "AWS EC2"
        }[s]
        self.synonym_mapper_mock.map_synonyms.side_effect = lambda s: {
            "ReactJS": "React", "TF 2": "TensorFlow", "AWS EC2": "AWS"
        }.get(s, s)
        self.taxonomy_mock.categorize_skill.side_effect = lambda s: {
            "React": "frameworks", "TensorFlow": "libraries", "AWS": "cloud_platforms"
        }[s]

        state = {"parsed_jd": {"job_description": "Skilled in ReactJS, TF 2 and AWS EC2."}}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        self.assertIn("React", extracted_skills.frameworks)
        self.assertIn("TensorFlow", extracted_skills.libraries)
        self.assertIn("AWS", extracted_skills.cloud_platforms)
        self.assertEqual(extracted_skills.metadata["total_skills"], 3)
        self.assertEqual(extracted_skills.metadata["duplicate_skills_removed"], 0)
        self.assertTrue(extracted_skills.metadata["normalization_applied"])

    @pytest.mark.skip(reason="Outdated")
    def test_taxonomy_classification(self):
        self.extractor_mock.extract_from_parsed_jd.return_value = ["Python", "Docker", "Agile", "Communication"]
        self.normalization_mock.normalize_skill.side_effect = lambda s: s
        self.synonym_mapper_mock.map_synonyms.side_effect = lambda s: s
        self.taxonomy_mock.categorize_skill.side_effect = lambda s: {
            "Python": "programming_languages",
            "Docker": "devops_tools",
            "Agile": "methodologies",
            "Communication": "soft_skills"
        }[s]

        state = {"parsed_jd": {"requirements": "Must have Python, Docker, Agile, Communication skills."}}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        self.assertIn("Python", extracted_skills.programming_languages)
        self.assertIn("Docker", extracted_skills.devops_tools)
        self.assertIn("Agile", extracted_skills.methodologies)
        self.assertIn("Communication", extracted_skills.soft_skills)
        self.assertEqual(extracted_skills.metadata["total_skills"], 4)

    @pytest.mark.skip(reason="Outdated")
    def test_missing_sections(self):
        self.extractor_mock.extract_from_parsed_jd.return_value = []
        state = {"parsed_jd": {"non_existent_section": "Some content"}}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        self.assertEqual(extracted_skills.calculate_total_skills(), 0)

    @pytest.mark.skip(reason="Outdated")
    def test_malformed_input_parsed_jd_not_dict(self):
        state = {"parsed_jd": "This is not a dict"}
        with self.assertRaises(AttributeError):
            self.agent.run(state)

    @pytest.mark.skip(reason="Outdated")
    def test_json_serialization(self):
        self.extractor_mock.extract_from_parsed_jd.return_value = ["Python"]
        self.normalization_mock.normalize_skill.return_value = "Python"
        self.synonym_mapper_mock.map_synonyms.return_value = "Python"
        self.taxonomy_mock.categorize_skill.return_value = "programming_languages"

        state = {"parsed_jd": {"job_description": "Python developer."}}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])
        json_output = extracted_skills.to_json()
        self.assertIsInstance(json_output, str)
        self.assertIn("Python", json_output)
        self.assertIn("programming_languages", json_output)
        self.assertIn("extraction_timestamp", json_output)

    @pytest.mark.skip(reason="Outdated")
    def test_schema_validation(self):
        valid_data = {
            "programming_languages": ["Python"],
            "frameworks": [],
            "libraries": [],
            "databases": [],
            "cloud_platforms": [],
            "devops_tools": [],
            "ai_ml": [],
            "data_engineering": [],
            "analytics_bi": [],
            "operating_systems": [],
            "version_control": [],
            "methodologies": [],
            "certifications": [],
            "soft_skills": [],
            "technical_skills": [],
            "other_skills": [],
            "metadata": {
                "extraction_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "schema_version": "1.0",
                "total_skills": 1,
                "duplicate_skills_removed": 0,
                "normalization_applied": True,
            }
        }
        extracted_skills = ExtractedSkills(**valid_data)
        self.assertEqual(extracted_skills.programming_languages, ["Python"])
        self.assertEqual(extracted_skills.metadata["total_skills"], 1)

        invalid_data = valid_data.copy()
        invalid_data["programming_languages"] = "Python"
        with self.assertRaises(ValueError):
            ExtractedSkills(**invalid_data)

    @pytest.mark.skip(reason="Outdated")
    def test_skill_extractor_logic_integration(self):
        self.agent = SkillExtractorAgent()

        parsed_jd = {
            "job_description": "We need a strong Python developer with React.js experience. Knowledge of AWS EC2 and Docker is a plus. Agile methodologies are used. Excellent communication skills.",
            "requirements": "Familiarity with k8s, tf, and PostgreSQL is preferred. Expertise in Machine learning."
        }

        state = {"parsed_jd": parsed_jd}
        result_state = self.agent.run(state)
        extracted_skills = ExtractedSkills(**result_state["extracted_skills"])

        self.assertIn("Python", extracted_skills.programming_languages)
        self.assertIn("React", extracted_skills.frameworks)
        self.assertIn("AWS", extracted_skills.cloud_platforms)
        self.assertIn("Docker", extracted_skills.devops_tools)
        self.assertIn("Kubernetes", extracted_skills.devops_tools)
        self.assertIn("TensorFlow", extracted_skills.libraries)
        self.assertIn("PostgreSQL", extracted_skills.databases)
        self.assertIn("Agile", extracted_skills.methodologies)
        self.assertIn("Communication", extracted_skills.soft_skills)
        self.assertIn("Machine Learning", extracted_skills.ai_ml)
        
        self.assertEqual(extracted_skills.metadata["total_skills"], 10)
        self.assertGreaterEqual(extracted_skills.metadata["duplicate_skills_removed"], 0)
        self.assertTrue(extracted_skills.metadata["normalization_applied"])

if __name__ == '__main__':
    unittest.main()
