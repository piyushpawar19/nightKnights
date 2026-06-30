import re
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EducationParser:
    """Parses education requirements from job description text."""

    def __init__(self):
        self.degree_patterns = {
            "bachelor": re.compile(r"b(?:\.?e|\.?s|\.?tech)|bachelor(?:\'s)? degree", re.IGNORECASE),
            "master": re.compile(r"m(?:\.?e|\.?tech|\.?s)|master(?:\'s)? degree", re.IGNORECASE),
            "phd": re.compile(r"ph(?:\.?d|d\.?)|doctorate(?:\'s)? degree", re.IGNORECASE),
            "mba": re.compile(r"mba|master of business administration", re.IGNORECASE),
        }
        self.field_of_study_keywords = [
            "computer science", "engineering", "electrical engineering", "software engineering",
            "information technology", "it", "data science", "mathematics", "statistics",
            "physics", "computer engineering", "related technical field", "relevant field"
        ]
        self.equivalent_degree_patterns = [
            re.compile(r"equivalent degree|equivalent practical experience", re.IGNORECASE),
            re.compile(r"or equivalent combination of education and experience", re.IGNORECASE),
        ]

    def parse_education(self, text: str) -> List[Tuple[Optional[str], Optional[str], bool]]:
        """
        Extracts education requirements (degree, field, mandatory) from the text.

        Args:
            text (str): The text section containing education information.

        Returns:
            List[Tuple[Optional[str], Optional[str], bool]]:
                A list of tuples, each containing (degree, field_of_study, is_mandatory).
        """
        found_education = []
        text_lower = text.lower()

        for degree_type, pattern in self.degree_patterns.items():
            for match in pattern.finditer(text_lower):
                sentence_start = text_lower.rfind(".", 0, match.start()) + 1
                sentence_end = text_lower.find(".", match.end())
                if sentence_end == -1:
                    sentence_end = len(text_lower)
                # Ensure sentence is not None before processing
                sentence_content = text_lower[sentence_start:sentence_end] if sentence_start < sentence_end else ""

                is_mandatory = False
                if sentence_content:
                    is_mandatory = any(re.search(r"(required|mandatory|must have)", sentence_content, re.IGNORECASE))

                field = None
                if sentence_content:
                    for fos_keyword in self.field_of_study_keywords:
                        if re.search(r"\b" + re.escape(fos_keyword) + r"\b", sentence_content):
                            field = fos_keyword
                            break

                found_education.append((degree_type, field, is_mandatory))

        for pattern in self.equivalent_degree_patterns:
            if pattern.search(text_lower):
                # Assuming equivalent is not strictly mandatory unless explicitly specified in a surrounding phrase not captured here.
                # For now, default to False.
                found_education.append(("equivalent", None, False))

        # Remove duplicates and return
        return list(set(found_education))
