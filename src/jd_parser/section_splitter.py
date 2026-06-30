import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SectionSplitter:
    """Splits a raw job description into logical sections based on keywords and headings."""

    def __init__(self):
        self.section_patterns = {
            "responsibilities": re.compile(r"responsibilities|duties|what you will do", re.IGNORECASE),
            "requirements": re.compile(r"requirements|qualifications|skills and qualifications|what you bring", re.IGNORECASE),
            "skills": re.compile(r"technical skills|skills required|key skills", re.IGNORECASE),
            "education": re.compile(r"education|educational background", re.IGNORECASE),
            "experience": re.compile(r"experience|professional experience", re.IGNORECASE),
            "benefits": re.compile(r"benefits|perks|what we offer", re.IGNORECASE),
            "nice_to_have": re.compile(r"nice to have|bonus points|preferred qualifications|good to have", re.IGNORECASE),
            "about_us": re.compile(r"about us|company overview", re.IGNORECASE),
            "job_info": re.compile(r"job title|company|location|about the role", re.IGNORECASE) # Added job_info pattern
        }

    def split_jd_into_sections(self, jd_text: str) -> Dict[str, str]:
        """
        Splits the job description into sections based on predefined patterns.

        Args:
            jd_text (str):
                The raw job description text.

        Returns:
            Dict[str, str]: A dictionary where keys are section names and values are their content.
        """
        sections: Dict[str, str] = {"unclassified": jd_text}
        lines = jd_text.splitlines()
        current_section = "unclassified"
        section_content: List[str] = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if section_content and current_section != "unclassified":
                    sections[current_section] = "\n".join(section_content).strip()
                    section_content = []
                    current_section = "unclassified"
                continue

            detected_new_section = False
            for section_name, pattern in self.section_patterns.items():
                if pattern.search(line_stripped) and len(line_stripped.split()) < 10:  # Likely a heading
                    if current_section != "unclassified" and section_content:
                        sections[current_section] = "\n".join(section_content).strip()

                    current_section = section_name
                    section_content = []
                    logger.info(f"Detected section: {current_section} with heading ")
                    detected_new_section = True
                    break

            if not detected_new_section:
                section_content.append(line)

        if section_content and current_section != "unclassified":
            sections[current_section] = "\n".join(section_content).strip()

        # If no specific sections were found, the whole JD might be in \"unclassified\"
        # Attempt to re-split \"unclassified\" if it contains a lot of content and no other sections were found
        if "unclassified" in sections and len(sections) == 1:
            logger.debug("No specific sections detected, attempting a more generic split.")
            return self._generic_split(jd_text)

        return sections

    def _generic_split(self, jd_text: str) -> Dict[str, str]:
        """
        A fallback generic splitter if specific section patterns don\"t yield results.
        Attempts to split by common paragraph breaks or bullet points if no clear headings.
        """
        sections: Dict[str, str] = {}
        paragraphs = re.split(r"\n\s*\n", jd_text.strip())
        if paragraphs:
            sections["main_content"] = "\n\n".join(paragraphs)
        else:
            sections["main_content"] = jd_text
        return sections
