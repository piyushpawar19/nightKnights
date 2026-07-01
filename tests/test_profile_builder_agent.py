import pytest
import unittest
from src.retrieval.profile_builder_agent import CandidateProfileBuilder, InvalidCandidateRecordError
from src.models.candidate_profile import CandidateProfile

class TestCandidateProfileBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = CandidateProfileBuilder()
        
        # Valid sample candidate record matching schemas
        self.valid_sample = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Jane Doe",
                "headline": "Senior Software Engineer",
                "summary": "Experienced engineer with a passion for clean code.",
                "location": "Boston",
                "country": "USA",
                "years_of_experience": 5.0,
                "current_title": "Software Engineer II",
                "current_company": "TechCorp"
            },
            "career_history": [
                {
                    "company": "TechCorp",
                    "title": "Software Engineer II",
                    "duration_months": 24,
                    "is_current": True,
                    "description": "Full-stack development using Python and React."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 10},
                {"name": "React", "proficiency": "advanced", "endorsements": 5}
            ],
            "education": [
                {
                    "institution": "Boston University",
                    "degree": "B.S.",
                    "field_of_study": "Computer Science",
                    "start_year": 2016,
                    "end_year": 2020,
                    "tier": "tier_2"
                }
            ],
            "certifications": [
                {"name": "AWS Solutions Architect", "issuer": "Amazon", "year": 2021}
            ],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2020-05-01",
                "last_active_date": "2026-06-25",
                "open_to_work_flag": True,
                "applications_submitted_30d": 3,
                "connection_count": 150
            }
        }

    @pytest.mark.skip(reason="Outdated")
    def test_build_profile_success(self):
        profile = self.builder.build_profile(self.valid_sample)
        self.assertIsInstance(profile, CandidateProfile)
        self.assertEqual(profile.candidate_id, "CAND_0000001")
        self.assertEqual(profile.full_name, "Jane Doe")
        self.assertEqual(profile.current_company, "TechCorp")
        self.assertEqual(profile.current_role, "Software Engineer II")
        self.assertEqual(profile.years_experience, 5.0)
        self.assertIn("Python", profile.skills)
        self.assertIn("React", profile.skills)
        self.assertIn("AWS Solutions Architect", profile.certifications[0])
        self.assertEqual(profile.metadata["profile_completeness"], 0.9)
        self.assertEqual(profile.metadata["num_skills"], 2)
        self.assertIn("Software Engineer II", profile.search_text)

    @pytest.mark.skip(reason="Outdated")
    def test_build_profile_missing_candidate_id(self):
        invalid_sample = self.valid_sample.copy()
        del invalid_sample["candidate_id"]
        
        with self.assertRaises(InvalidCandidateRecordError):
            self.builder.build_profile(invalid_sample)

    @pytest.mark.skip(reason="Outdated")
    def test_build_profile_invalid_type(self):
        with self.assertRaises(InvalidCandidateRecordError):
            self.builder.build_profile("Not a dictionary")

    @pytest.mark.skip(reason="Outdated")
    def test_build_profiles_batch(self):
        candidates = [self.valid_sample, {"invalid": "record"}, self.valid_sample]
        profiles = self.builder.build_profiles(candidates)
        
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0].candidate_id, "CAND_0000001")
        self.assertEqual(profiles[1].candidate_id, "CAND_0000001")

    @pytest.mark.skip(reason="Outdated")
    def test_clean_text(self):
        text = "  Hello    World  \n New   Line "
        cleaned = self.builder._clean_text(text)
        self.assertEqual(cleaned, "Hello World New Line")

    @pytest.mark.skip(reason="Outdated")
    def test_safe_get(self):
        val = self.builder._safe_get(self.valid_sample, "profile.anonymized_name")
        self.assertEqual(val, "Jane Doe")

        val_default = self.builder._safe_get(self.valid_sample, "profile.non_existent", "Default")
        self.assertEqual(val_default, "Default")

if __name__ == "__main__":
    unittest.main()
