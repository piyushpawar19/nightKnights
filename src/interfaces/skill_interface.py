from typing import Protocol, runtime_checkable
from src.schemas.skill_schema import ExtractedSkills


@runtime_checkable
class SkillExtractorInterface(Protocol):
    def run(self, state: dict) -> dict:
        """
        Extracts, normalizes, and categorizes skills from parsed job description data.
        The input state dictionary must contain \'parsed_jd\'.
        The output state dictionary will contain \'extracted_skills\'.
        """
        ...


@runtime_checkable
class SkillNormalizationInterface(Protocol):
    def normalize_skill(self, skill: str) -> str:
        """
        Normalizes a single skill string (casing, whitespace, punctuation, spelling).
        """
        ...

    def normalize_skills(self, skills: list[str]) -> list[str]:
        """
        Normalizes a list of skill strings.
        """
        ...


@runtime_checkable
class SynonymMappingInterface(Protocol):
    def map_synonyms(self, skill: str) -> str:
        """
        Maps a skill to its preferred synonym.
        """
        ...


@runtime_checkable
class SkillTaxonomyInterface(Protocol):
    def categorize_skill(self, skill: str) -> str:
        """
        Categorizes a single skill into a predefined category (e.g., \'programming_languages\').
        """
        ...


@runtime_checkable
class SkillExtractionEngineInterface(Protocol):
    def extract_skills(self, text: str) -> list[str]:
        """
        Extracts raw skill strings from a given text using regex and keyword dictionaries.
        """
        ...
