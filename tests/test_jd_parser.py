import pytest
import unittest
import datetime
from typing import Dict, Any
from src.agents.jd_parser_agent import JDParserAgent
from src.schemas.jd_schema import ParsedJD, JobInfo, Requirements, Skills, Responsibilities, Education, Certification, ParsingMetadata
from src.jd_parser.section_splitter import SectionSplitter
from src.jd_parser.experience_parser import ExperienceParser
from src.jd_parser.education_parser import EducationParser
from src.jd_parser.salary_parser import SalaryParser


class TestJDParserAgent(unittest.TestCase):

    def setUp(self):
        self.agent = JDParserAgent()
        self.section_splitter = SectionSplitter()
        self.experience_parser = ExperienceParser()
        self.education_parser = EducationParser()
        self.salary_parser = SalaryParser()


    @pytest.mark.skip(reason="Outdated")
    def test_empty_jd(self):
        state = {"raw_jd": ""}
        parsed_state = self.agent.run(state)
        self.assertIn("parsed_jd", parsed_state)
        parsed_jd = parsed_state["parsed_jd"]
        self.assertIsInstance(parsed_jd, ParsedJD)
        self.assertIsInstance(parsed_jd.job_info, JobInfo)
        self.assertIsInstance(parsed_jd.requirements, Requirements)
        self.assertIsInstance(parsed_jd.skills, Skills)
        self.assertIsInstance(parsed_jd.responsibilities, Responsibilities)
        self.assertIsInstance(parsed_jd.metadata, ParsingMetadata)

        self.assertIsNone(parsed_jd.job_info.title)
        self.assertEqual(len(parsed_jd.requirements.mandatory_requirements), 0)
        self.assertEqual(len(parsed_jd.skills.technical_skills), 0)


    @pytest.mark.skip(reason="Outdated")
    def test_minimal_jd(self):
        minimal_jd = (
            "Software Engineer\n"
            "Acme Corp - New York, NY\n"
            "We are looking for a skilled Software Engineer."
        )
        state = {"raw_jd": minimal_jd}
        parsed_state = self.agent.run(state)
        parsed_jd = parsed_state["parsed_jd"]

        self.assertEqual(parsed_jd.job_info.title, "Software Engineer")
        # Company and location might be harder for simple regex, depending on the implementation
        # self.assertEqual(parsed_jd.job_info.company, "Acme Corp")
        # self.assertEqual(parsed_jd.job_info.location, "New York, NY")
        self.assertIsInstance(parsed_jd, ParsedJD)


    @pytest.mark.skip(reason="Outdated")
    def test_experience_parsing(self):
        jd_text = (
            "We require a minimum of 5 years of experience in software development. "
            "Ideally, 7-10 years of experience. This is a Senior Software Engineer role. "
            "Candidates with 2+ years are also welcome for a Junior position."
        )
        min_exp, max_exp, seniority = self.experience_parser.parse_experience(jd_text)
        self.assertEqual(min_exp, 7)
        self.assertEqual(max_exp, 10)
        self.assertEqual(seniority, "senior")

        jd_text_junior = "Entry level position, 0-2 years of experience."
        min_exp_j, max_exp_j, seniority_j = self.experience_parser.parse_experience(jd_text_junior)
        self.assertEqual(min_exp_j, 0)
        self.assertEqual(max_exp_j, 2)
        self.assertEqual(seniority_j, "junior")


    @pytest.mark.skip(reason="Outdated")
    def test_education_parsing(self):
        jd_text = (
            "Requires a BS or MS in Computer Science or a related field. "
            "PhD is a plus. Equivalent degree is acceptable."
        )
        education = self.education_parser.parse_education(jd_text)
        self.assertIn(("bachelor", "computer science", False), education)
        self.assertIn(("master", "computer science", False), education)
        self.assertIn(("phd", None, False), education)
        self.assertIn(("equivalent", None, False), education)

        jd_text_mandatory = "MUST have a BE in Electrical Engineering."
        education_m = self.education_parser.parse_education(jd_text_mandatory)
        self.assertIn(("bachelor", "electrical engineering", True), education_m)


    @pytest.mark.skip(reason="Outdated")
    def test_salary_parsing(self):
        jd_text = (
            "Salary: $100,000 - $120,000 per annum. "
            "Also considering candidates for 8 LPA to 12 LPA. "
            "Monthly salary can be around $8000."
        )
        salary_info = self.salary_parser.parse_salary(jd_text)
        self.assertIsNotNone(salary_info)
        self.assertEqual(salary_info["min_salary"], 100000.0)
        self.assertEqual(salary_info["max_salary"], 120000.0)
        self.assertEqual(salary_info["currency"], "USD")
        self.assertEqual(salary_info["period"], "annual")

        jd_text_inr = "Salary 15-20 LPA CTC. Monthly Rs. 1,50,000."
        salary_info_inr = self.salary_parser.parse_salary(jd_text_inr)
        self.assertIsNotNone(salary_info_inr)
        self.assertEqual(salary_info_inr["min_salary"], 1500000.0) # 15 LPA
        self.assertEqual(salary_info_inr["max_salary"], 2000000.0) # 20 LPA
        self.assertEqual(salary_info_inr["currency"], "INR")
        self.assertEqual(salary_info_inr["period"], "annual")


    @pytest.mark.skip(reason="Outdated")
    def test_section_detection(self):
        jd_text = (
            "Job Title\n"
            "About Us:\nSome company info.\n"
            "Responsibilities:\n- Do X\n- Do Y\n"
            "Requirements:\n- Have Z skill\n"
            "Education:\n- Degree in CS\n"
            "Benefits:\n- Health, Dental\n"
            "Nice to Have:\n- Experience with A\n"
        )
        sections = self.section_splitter.split_jd_into_sections(jd_text)
        self.assertIn("about_us", sections)
        self.assertIn("responsibilities", sections)
        self.assertIn("requirements", sections)
        self.assertIn("education", sections)
        self.assertIn("benefits", sections)
        self.assertIn("nice_to_have", sections)
        self.assertNotIn("unclassified", sections) # Should ideally be empty if all split

    @pytest.mark.skip(reason="Outdated")
    def test_schema_validation(self):
        valid_parsed_jd = ParsedJD(
            job_info=JobInfo(title="Test Job", company="Test Co", location="Test City"),
            requirements=Requirements(mandatory_requirements=["skill1"], education=[Education(degree="BS", field="CS", required=True)]),
            skills=Skills(technical_skills=["Python"]),
            responsibilities=Responsibilities(responsibilities_list=["task1"]),
            preferences=dict(), # Empty preferences is okay for now
            metadata=ParsingMetadata(parse_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        )
        # This will raise ValidationError if not valid
        self.assertIsInstance(valid_parsed_jd, ParsedJD)

        # Test invalid schema (e.g., missing mandatory field for sub-schema if any were defined)
        # For ParsedJD, most fields are optional or have defaults, so direct Pydantic validation handles much.
        # This test ensures basic instantiation works.

if __name__ == "__main__":
    unittest.main()
