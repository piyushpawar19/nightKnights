import re
from typing import List, Dict, Any
from src.schemas.jd_schema import Certification, Education
import logging

logger = logging.getLogger(__name__)

class RequirementExtractor:
    """Extracts various types of requirements from job description text."""

    def __init__(self):
        self.mandatory_patterns = [
            re.compile(r"must have|required|mandatory|essential|minimum of", re.IGNORECASE),
            re.compile(r"need to have|demonstrated experience in", re.IGNORECASE),
        ]
        self.preferred_patterns = [
            re.compile(r"preferred|nice to have|bonus points|ideally", re.IGNORECASE),
            re.compile(r"a plus|strong advantage", re.IGNORECASE),
        ]
        self.certification_patterns = [
            re.compile(r"(certified|certification) in (\w+\s?\w+)", re.IGNORECASE),
            re.compile(r"(aws|azure|gcp|pmp|cisco|scrum master) certification", re.IGNORECASE),
        ]
        self.education_patterns = [
            re.compile(r"(bachelor|master|phd|be|b.tech|bs|me|m.tech|ms|mba) in (\w+\s?\w+)", re.IGNORECASE),
            re.compile(r"(computer science|engineering|it|information technology) degree", re.IGNORECASE),
            re.compile(r"equivalent degree|relevant field", re.IGNORECASE),
        ]
        self.technical_keywords = [
            "python", "java", "c++", "javascript", "react", "angular", "vue", "spring", "django",
            "flask", "node.js", "aws", "azure", "gcp", "docker", "kubernetes", "sql", "nosql",
            "tensorflow", "pytorch", "scikit-learn", "hadoop", "spark", "kafka", "terraform",
            "ansible", "jenkins", "git", "jira", "agile", "scrum", "linux", "unix", "windows",
            "api", "rest", "graphql", "microservices", "frontend", "backend", "fullstack",
            "data structures", "algorithms", "object-oriented programming", "oop", "databases",
            "cloud computing", "machine learning", "deep learning", "artificial intelligence",
            "natural language processing", "nlp", "computer vision", "cv", "data science", "big data",
            "web development", "mobile development", "android", "ios", "devops", "ci/cd",
            "system design", "software architecture", "testing", "unit testing", "integration testing",
            "security", "cybersecurity", "networking", "scripting", "bash", "shell", "powershell",
            "typescript", "golang", "ruby", "php", "swift", "kotlin", "r", "scala", "html", "css",
            "mongodb", "postgresql", "mysql", "redis", "cassandra", "dynamodb", "s3", "ec2", "lambda",
            "azure functions", "google cloud functions", "kubernetes", "ecs", "eks", "gke",
            "airflow", "mlflow", "databricks", "snowflake", "tableau", "power bi", "excel",
            "confluence", "slack", "microsoft teams", "communication", "problem-solving", "leadership",
            "teamwork", "critical thinking", "adaptability", "creativity", "time management",
            "project management", "stakeholder management", "presentation skills", "mentoring",
            "documentation", "writing skills", "research", "analytical skills", "troubleshooting",
            "performance tuning", "optimization", "scalability", "reliability", "resilience",
            "distributed systems", "high availability", "fault tolerance", "disaster recovery",
            "message queues", "event-driven architecture", "serverless", "blockchain", "iot",
            "augmented reality", "virtual reality", "game development", "unity", "unreal engine",
            "graphic design", "ux/ui", "figma", "sketch", "adobe xd", "photoshop", "illustrator",
            "seo", "sem", "marketing", "sales", "business development", "customer service",
            "financial analysis", "budgeting", "forecasting", "compliance", "risk management",
            "auditing", "legal", "hr", "recruitment", "training", "onboarding", "payroll",
            "supply chain", "logistics", "operations", "manufacturing", "quality assurance", "qa",
            "manual testing", "automation testing", "performance testing", "security testing",
            "functional testing", "non-functional testing", "user acceptance testing", "uat",
            "test plan", "test strategy", "test case", "bug tracking", "test management tools"
        ]

    def extract_requirements(self, text: str) -> Dict[str, Any]:
        """
        Extracts mandatory, preferred, certifications, and education from a given text section.

        Args:
            text (str): The text section (e.g., from 'requirements' or 'qualifications').

        Returns:
            Dict[str, Any]: A dictionary containing lists of extracted items.
        """
        mandatory_reqs = self._extract_list_items(text, self.mandatory_patterns)
        preferred_reqs = self._extract_list_items(text, self.preferred_patterns)
        certifications = self._extract_certifications(text)
        education = self._extract_education(text)
        technical_skills = self._extract_technical_skills(text)

        return {
            "mandatory_requirements": mandatory_reqs,
            "preferred_requirements": preferred_reqs,
            "certifications": certifications,
            "education": education,
            "technical_skills": technical_skills,
        }

    def _extract_list_items(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """
        Extracts general list items that match any of the given patterns.
        """
        extracted_items = set()
        lines = text.splitlines()

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if the line itself contains a strong indicator for mandatory/preferred
            is_mandatory_line = any(p.search(line_stripped) for p in self.mandatory_patterns)
            is_preferred_line = any(p.search(line_stripped) for p in self.preferred_patterns)

            # Simple bullet point or numbered list item detection
            match = re.match(r"^\s*[-*•\d]+\s*(.*)", line_stripped)
            item_content = match.group(1).strip() if match else line_stripped

            # Heuristic: if a line is short and looks like an item, add it.
            # Avoid adding section headers themselves.
            if len(item_content.split()) > 3 and len(item_content.split()) < 30 and not any(p.search(item_content) for p in self.certification_patterns + self.education_patterns):
                if is_mandatory_line or is_preferred_line or (match and len(item_content) > 10):
                    extracted_items.add(item_content)

        return sorted(list(extracted_items))

    def _extract_certifications(self, text: str) -> List[Certification]:
        """
        Extracts certifications from the text.
        """
        certifications = []
        for pattern in self.certification_patterns:
            for match in pattern.finditer(text):
                cert_name = match.group(0).strip()  # Get the full matched string
                is_required = any(p.search(match.group(0)) for p in self.mandatory_patterns)
                certifications.append(Certification(name=cert_name, required=is_required))
        return certifications

    def _extract_education(self, text: str) -> List[Education]:
        """
        Extracts education requirements from the text.
        """
        education_list = []
        for pattern in self.education_patterns:
            for match in pattern.finditer(text):
                degree_info = match.group(0).strip()
                # Attempt to extract degree and field more specifically
                degree_match = re.search(r"(bachelor|master|phd|be|b.tech|bs|me|m.tech|ms|mba)", degree_info, re.IGNORECASE)
                field_match = re.search(r"in (\w+\s?\w+|computer science|engineering|it|information technology)", degree_info, re.IGNORECASE)

                degree = degree_match.group(0) if degree_match else None
                field = field_match.group(1) if field_match else None

                is_required = any(p.search(match.group(0)) for p in self.mandatory_patterns)
                education_list.append(Education(degree=degree, field=field, required=is_required))
        return education_list

    def _extract_technical_skills(self, text: str) -> List[str]:
        """
        Extracts technical skills based on a predefined list of keywords.
        """
        found_skills = set()
        text_lower = text.lower()

        for keyword in self.technical_keywords:
            if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", text_lower):
                found_skills.add(keyword)

        return sorted(list(found_skills))
