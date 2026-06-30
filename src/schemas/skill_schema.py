import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class ExtractedSkills(BaseModel):
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    libraries: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    cloud_platforms: List[str] = Field(default_factory=list)
    devops_tools: List[str] = Field(default_factory=list)
    ai_ml: List[str] = Field(default_factory=list)
    data_engineering: List[str] = Field(default_factory=list)
    analytics_bi: List[str] = Field(default_factory=list)
    operating_systems: List[str] = Field(default_factory=list)
    version_control: List[str] = Field(default_factory=list)
    methodologies: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    other_skills: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.metadata:
            self.metadata = {
                "extraction_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "schema_version": "1.0",
                "total_skills": self.calculate_total_skills(),
                "duplicate_skills_removed": 0,
                "normalization_applied": False,
            }

    def calculate_total_skills(self) -> int:
        total = 0
        for field_name, value in self.model_dump().items():
            if isinstance(value, list):
                total += len(value)
        return total

    def add_skill(self, category: str, skill: str):
        if category not in self.model_fields:
            raise ValueError(f"Invalid skill category: {category}")
        getattr(self, category).append(skill)
        self.metadata["total_skills"] = self.calculate_total_skills()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
