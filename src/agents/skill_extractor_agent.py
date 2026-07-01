import logging
import datetime
from typing import Dict, Any

from src.schemas.skill_schema import ExtractedSkills
from src.interfaces.skill_interface import SkillExtractorInterface
from src.preprocessing.normalization import SkillNormalization
from src.preprocessing.synonym_mapper import SynonymMapper
from src.preprocessing.taxonomy import SkillTaxonomy
from src.preprocessing.skill_extractor import SkillExtractor

class SkillExtractorAgent(SkillExtractorInterface):
    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.skill_normalizer = SkillNormalization()
        self.synonym_mapper = SynonymMapper()
        self.skill_taxonomy = SkillTaxonomy()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logging.info("SkillExtractorAgent started.")
        if "parsed_jd" not in state or not state["parsed_jd"]:
            logging.warning("No \'parsed_jd\' found in state. Returning empty extracted skills.")
            state["extracted_skills"] = ExtractedSkills().model_dump()
            return state

        parsed_jd = state["parsed_jd"]
        all_raw_skills = self.skill_extractor.extract_from_parsed_jd(parsed_jd)
        logging.info(f"Raw skills extracted: {len(all_raw_skills)}")

        normalized_and_mapped_skills_set = set()
        duplicate_skills_removed = 0

        for skill in all_raw_skills:
            normalized_skill = self.skill_normalizer.normalize_skill(skill)
            mapped_skill = self.synonym_mapper.map_synonyms(normalized_skill)
            if mapped_skill not in normalized_and_mapped_skills_set:
                normalized_and_mapped_skills_set.add(mapped_skill)
            else:
                duplicate_skills_removed += 1

        final_extracted_skills = ExtractedSkills()
        for skill in normalized_and_mapped_skills_set:
            category = self.skill_taxonomy.categorize_skill(skill)
            try:
                final_extracted_skills.add_skill(category, skill)
            except ValueError as e:
                logging.error(f"Error categorizing skill \\'{skill}\\': {e}")
                final_extracted_skills.add_skill("other_skills", skill)

        final_extracted_skills.metadata["duplicate_skills_removed"] = duplicate_skills_removed
        final_extracted_skills.metadata["normalization_applied"] = True
        final_extracted_skills.metadata["total_skills"] = final_extracted_skills.calculate_total_skills()
        final_extracted_skills.metadata["extraction_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        state["extracted_skills"] = final_extracted_skills.model_dump()
        logging.info(f"SkillExtractorAgent finished. Total unique skills: {final_extracted_skills.metadata['total_skills']}")
        return state
