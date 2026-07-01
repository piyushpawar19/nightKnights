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
        # Consolidate technical skills for a single, efficient regex search
        self.technical_keywords = [
            "python", "java", "javascript", "c++", "c#", "go", "ruby", "php",
            "typescript", "swift", "kotlin", "rust", "scala", "perl", "html", "css",
            "sql", "r", "bash", "shell", "react", "angular", "vue", "node.js", "express",
            "django", "flask", "spring", "rails", "asp.net", "laravel", "symfony",
            "next.js", "nestjs", "nuxtjs", "gatsby", "flutter", "xamarin", "react native",
            "jquery", "bootstrap", "tailwind", "material-ui", "ant design", "tensorflow",
            "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
            "d3.js", "redux", "axios", "mongoose", "sqlalchemy", "hibernate", "nltk",
            "spacy", "opencv", "pillow", "requests", "beautiful soup", "selenium",
            "cypress", "jest", "mocha", "chai", "junit", "nunit", "xunit", "mockito",
            "pytest", "postgresql", "mongodb", "mysql", "sql server", "oracle",
            "dynamodb", "cassandra", "redis", "elasticsearch", "firebase", "mariadb",
            "sqlite", "cosmos db", "neo4j", "hbase", "teradata", "vertica", "snowflake",
            "databricks", "aws", "azure", "gcp", "ibm cloud", "oracle cloud", "alibaba cloud",
            "heroku", "vercel", "netlify", "digitalocean", "docker", "kubernetes",
            "jenkins", "git", "github", "gitlab", "bitbucket", "travis ci", "circleci",
            "ansible", "puppet", "chef", "terraform", "cloudformation", "prometheus",
            "grafana", "nagios", "splunk", "elk stack", "jira", "confluence", "trello",
            "asana", "argo cd", "tekton", "ci/cd", "machine learning", "deep learning", "ai",
            "nlp", "computer vision", "reinforcement learning", "generative ai", "llm",
            "neural networks", "data science", "predictive modeling", "statistical modeling",
            "mlops", "apache spark", "apache hadoop", "apache kafka", "etl",
            "data warehousing", "data lakes", "apache flink", "apache airflow", "delta lake",
            "presto", "trino", "hive", "pig", "kafka streams", "beam", "tableau", "power bi",
            "looker", "qlikview", "google analytics", "mixpanel", "amplitude", "metabase",
            "redash", "superset", "business intelligence", "data visualization", "reporting",
            "dashboards", "linux", "windows", "macos", "ubuntu", "centos", "red hat",
            "debian", "android", "ios", "svn", "perforce", "mercurial", "agile", "scrum",
            "kanban", "waterfall", "lean", "xp", "tdd", "bdd", "ddd", "devops", "pmp",
            "csm", "aws certified", "azure certified", "gcp certified", "comptia", "ccna",
            "cisco certified", "oracle certified", "microsoft certified", "cka", "ceh",
            "api design", "microservices", "system design", "oop", "functional programming",
            "data structures", "algorithms", "networking", "security", "scalability",
            "performance tuning", "database design", "web development", "mobile development",
            "backend development", "frontend development", "fullstack development",
            "cloud architecture", "containerization", "virtualization", "scripting",
            "automation", "troubleshooting", "debugging", "technical writing",
            "unit testing", "integration testing", "end-to-end testing", "ci/cd pipelines",
            "infrastructure as code", "sre", "blockchain", "iot", "augmented reality",
            "virtual reality", "game development", "unity", "unreal engine", "graphic design",
            "ux/ui", "figma", "sketch", "adobe xd", "photoshop", "illustrator", "seo",
            "sem", "marketing", "sales", "business development", "customer service",
            "financial analysis", "budgeting", "forecasting", "compliance", "risk management",
            "auditing", "legal", "hr", "recruitment", "training", "onboarding", "payroll",
            "supply chain", "logistics", "operations", "manufacturing", "quality assurance",
            "qa", "manual testing", "automation testing", "performance testing",
            "security testing", "functional testing", "non-functional testing",
            "user acceptance testing", "uat", "test plan", "test strategy", "test case",
            "bug tracking", "test management tools",
            "communication", "teamwork", "problem-solving", "critical thinking",
            "adaptability", "leadership", "time management", "collaboration",
            "creativity", "interpersonal skills", "attention to detail"
        ]
        self.technical_skill_pattern = re.compile(
            r"\b(?:" + "|".join(map(re.escape, self.technical_keywords)) + r")\b",
            re.IGNORECASE
        )

    def extract_requirements(self, text: str) -> Dict[str, Any]:
        """
        Extracts mandatory, preferred, certifications, and education from a given text section.

        Args:
            text (str): The text section (e.g., from \"requirements\" or \"qualifications\").

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
        Extracts technical skills based on a predefined list of keywords using a compiled regex pattern.
        """
        found_skills = set()
        for match in self.technical_skill_pattern.finditer(text):
            found_skills.add(match.group(0))
        return sorted(list(found_skills))
