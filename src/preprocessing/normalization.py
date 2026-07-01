import re
import unicodedata
from typing import List
from src.interfaces.skill_interface import SkillNormalizationInterface

class SkillNormalization(SkillNormalizationInterface):
    def __init__(self):
        self.normalization_rules = {
            "js": "JavaScript",
            "py": "Python",
            "nodejs": "Node.js",
            "postgres": "PostgreSQL",
            "tf": "TensorFlow",
            "k8s": "Kubernetes",
            "aws": "Amazon Web Services",
            "azure": "Microsoft Azure",
            "gcp": "Google Cloud Platform",
            "ml": "Machine Learning",
            "ai": "Artificial Intelligence",
            "bi": "Business Intelligence",
            "ci/cd": "CI/CD",
            "devops": "DevOps",
            "api": "API",
            "rdbms": "Relational Database Management System",
            "nosql": "NoSQL",
            "golang": "Go",
            "c#": "C#",
            "c++": "C++",
            "css": "CSS",
            "html": "HTML",
            "sql": "SQL",
            "git": "Git",
            "jira": "Jira",
            "agile": "Agile",
            "scrum": "Scrum",
            "linux": "Linux",
            "windows": "Windows",
            "macos": "macOS",
        }

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s.\-\/#+ ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        return text

    def _apply_rules(self, skill: str) -> str:
        for abbreviation, full_form in self.normalization_rules.items():
            if skill == abbreviation.lower():
                return full_form
        return skill

    def _handle_singular_plural(self, skill: str) -> str:
        if skill.endswith("s") and skill[:-1] in self.normalization_rules.values():
            return skill[:-1]
        return skill

    def normalize_skill(self, skill: str) -> str:
        cleaned_skill = self._clean_text(skill)
        normalized_skill = self._apply_rules(cleaned_skill)
        normalized_skill = self._handle_singular_plural(normalized_skill)

        if normalized_skill.lower() == "javascript":
            normalized_skill = "JavaScript"
        elif normalized_skill.lower() == "python":
            normalized_skill = "Python"
        elif normalized_skill.lower() == "node.js":
            normalized_skill = "Node.js"
        elif normalized_skill.lower() == "postgresql":
            normalized_skill = "PostgreSQL"
        elif normalized_skill.lower() == "tensorflow":
            normalized_skill = "TensorFlow"
        elif normalized_skill.lower() == "kubernetes":
            normalized_skill = "Kubernetes"
        elif normalized_skill.lower() == "amazon web services":
            normalized_skill = "AWS"
        elif normalized_skill.lower() == "microsoft azure":
            normalized_skill = "Azure"
        elif normalized_skill.lower() == "google cloud platform":
            normalized_skill = "GCP"
        elif normalized_skill.lower() == "machine learning":
            normalized_skill = "Machine Learning"
        elif normalized_skill.lower() == "artificial intelligence":
            normalized_skill = "Artificial Intelligence"
        elif normalized_skill.lower() == "business intelligence":
            normalized_skill = "Business Intelligence"
        elif normalized_skill.lower() == "ci/cd":
            normalized_skill = "CI/CD"
        elif normalized_skill.lower() == "devops":
            normalized_skill = "DevOps"
        elif normalized_skill.lower() == "api":
            normalized_skill = "API"
        elif normalized_skill.lower() == "relational database management system":
            normalized_skill = "RDBMS"
        elif normalized_skill.lower() == "nosql":
            normalized_skill = "NoSQL"
        elif normalized_skill.lower() == "go":
            normalized_skill = "Go"
        elif normalized_skill.lower() == "c#":
            normalized_skill = "C#"
        elif normalized_skill.lower() == "c++":
            normalized_skill = "C++"
        elif normalized_skill.lower() == "css":
            normalized_skill = "CSS"
        elif normalized_skill.lower() == "html":
            normalized_skill = "HTML"
        elif normalized_skill.lower() == "sql":
            normalized_skill = "SQL"
        elif normalized_skill.lower() == "git":
            normalized_skill = "Git"
        elif normalized_skill.lower() == "jira":
            normalized_skill = "Jira"
        elif normalized_skill.lower() == "agile":
            normalized_skill = "Agile"
        elif normalized_skill.lower() == "scrum":
            normalized_skill = "Scrum"
        elif normalized_skill.lower() == "linux":
            normalized_skill = "Linux"
        elif normalized_skill.lower() == "windows":
            normalized_skill = "Windows"
        elif normalized_skill.lower() == "macos":
            normalized_skill = "macOS"

        return normalized_skill

    def normalize_skills(self, skills: List[str]) -> List[str]:
        normalized_list = []
        seen_skills = set()
        for skill in skills:
            normalized = self.normalize_skill(skill)
            if normalized and normalized not in seen_skills:
                normalized_list.append(normalized)
                seen_skills.add(normalized)
        return normalized_list
