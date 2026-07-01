import logging
import datetime
import re
from typing import Dict, Any, Optional

from src.schemas.jd_schema import (
    ParsedJD, JobInfo, Requirements, Skills, Responsibilities, Preferences, ParsingMetadata, Education, Certification
)
from src.jd_parser.section_splitter import SectionSplitter
from src.jd_parser.requirement_extractor import RequirementExtractor
from src.jd_parser.experience_parser import ExperienceParser
from src.jd_parser.education_parser import EducationParser
from src.jd_parser.salary_parser import SalaryParser

logger = logging.getLogger(__name__)

class JDParser:
    """Orchestrates the parsing of a raw Job Description into a structured ParsedJD object."""

    def __init__(self):
        self.section_splitter = SectionSplitter()
        self.requirement_extractor = RequirementExtractor()
        self.experience_parser = ExperienceParser()
        self.education_parser = EducationParser()
        self.salary_parser = SalaryParser()

    def parse_jd(self, raw_jd: str) -> ParsedJD:
        """
        Main method to parse the raw job description.

        Args:
            raw_jd (str): The unstructured job description text.

        Returns:
            ParsedJD: A strongly typed representation of the parsed job description.
        """
        sections = self.section_splitter.split_jd_into_sections(raw_jd)
        logger.info(f"JD split into sections: {sections.keys()}")

        job_info = self._parse_job_info(raw_jd, sections)
        requirements = self._parse_requirements(sections)
        skills = self._parse_skills(sections)
        responsibilities = self._parse_responsibilities(sections)
        preferences = self._parse_preferences(sections) # Added call to parse preferences
        metadata = self._create_parsing_metadata()

        return ParsedJD(
            job_info=job_info,
            requirements=requirements,
            skills=skills,
            responsibilities=responsibilities,
            preferences=preferences,
            metadata=metadata
        )

    def _parse_preferences(self, sections: Dict[str, str]) -> Preferences:
        """
        Parses job preferences or \"nice-to-haves\".
        This is a placeholder and would involve extracting specific preference details.
        """
        preferences_text = sections.get("nice_to_have", "")
        # For now, simply return a default empty Preferences object.
        # In a real scenario, you might extract keywords or phrases related to preferences.
        return Preferences() # Consider populating this with actual preferences if detectable











    def _parse_job_info(self, raw_jd: str, sections: Dict[str, str]) -> JobInfo:
        """
        Parses general job information.
        This is a placeholder; more sophisticated parsing would be needed here.
        """
        title = self._extract_title(raw_jd)
        company = self._extract_company(raw_jd) # Example: simplified
        location = self._extract_location(raw_jd)

        min_exp, max_exp, seniority = self.experience_parser.parse_experience(
            sections.get("experience", "") + "\n" + sections.get("requirements", "") + "\n" + sections.get("unclassified", "")
        )
        salary_info = self.salary_parser.parse_salary(
            sections.get("salary", "") + "\n" + sections.get("unclassified", "")
        )

        return JobInfo(
            title=title,
            company=company,
            location=location,
            minimum_experience=min_exp,
            maximum_experience=max_exp,
            seniority=seniority,
            salary=salary_info
            # Add other fields as they are parsed
        )

    def _extract_title(self, text: str) -> Optional[str]:
        """
        A very basic title extraction heuristic (e.g., first bold line, or first line).
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            # Heuristic: Often the first non-empty line, or a prominent one.
            return lines[0]
        return None

    def _extract_company(self, text: str) -> Optional[str]:
        """
        Placeholder for company extraction. Usually requires more context or named entity recognition.
        """
        # This is highly dependent on JD format. A simple regex might target "Company Name Jobs"
        match = re.search(r"at\s+([A-Z][a-zA-Z0-9\s&,-]+(?:Ltd|LLC|Inc|Corp|Group))", text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Placeholder for location extraction. Can use a list of cities/countries or NER.
        """
        # Example: looking for City, State or City, Country pattern
        match = re.search(r"(remote|hybrid|on-site)|([A-Z][a-zA-Z\s-]+,\s*[A-Z]{2,})|([A-Z][a-zA-Z\s-]+,\s*[A-Z][a-zA-Z\s-]+(?:, [A-Z]{2})?)", text)
        if match:
            for i in range(1, match.lastindex + 1):
                if match.group(i):
                    return match.group(i).strip()
        return None

    def _parse_requirements(self, sections: Dict[str, str]) -> Requirements:
        """
        Parses job requirements.
        """
        requirements_text = sections.get("requirements", "") + "\n" + sections.get("unclassified", "")
        extracted = self.requirement_extractor.extract_requirements(requirements_text)

        # Convert extracted education/certifications to schema objects
        education_list = []
        for degree, field, is_mandatory in self.education_parser.parse_education(requirements_text):
            education_list.append(Education(degree=degree, field=field, required=is_mandatory))

        certifications_list = [Certification(**cert.dict()) for cert in extracted["certifications"]]

        return Requirements(
            mandatory_requirements=extracted["mandatory_requirements"],
            preferred_requirements=extracted["preferred_requirements"],
            certifications=certifications_list,
            education=education_list,
        )

    def _parse_skills(self, sections: Dict[str, str]) -> Skills:
        """
        Parses job skills.
        """
        skills_text = sections.get("skills", "") + "\n" + sections.get("requirements", "") + "\n" + sections.get("unclassified", "")
        technical_skills = self.requirement_extractor._extract_technical_skills(skills_text)

        # This is a simplification; a real skill extractor would categorize more granularly.
        # For now, put all detected technical skills into the \"technical_skills\" field.
        return Skills(
            technical_skills=technical_skills,
            programming_languages=[], # To be filled by a more advanced extractor
            frameworks=[],
            libraries=[],
            databases=[],
            cloud=[],
            devops=[],
            ai_ml=[],
            soft_skills=[]
        )

    def _parse_responsibilities(self, sections: Dict[str, str]) -> Responsibilities:
        """
        Parses job responsibilities.
        """
        responsibilities_text = sections.get("responsibilities", "")
        lines = [line.strip() for line in responsibilities_text.splitlines() if line.strip() and len(line.strip().split()) > 3]
        return Responsibilities(responsibilities_list=lines)

    def _create_parsing_metadata(self) -> ParsingMetadata:
        """
        Creates metadata for the parsing process.
        """
        return ParsingMetadata(
            parse_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            parser_version="1.0"
        )
