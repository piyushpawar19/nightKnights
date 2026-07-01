import re
from typing import Optional, Tuple
import logging
from functools import lru_cache
import os
from joblib import Memory

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

logger = logging.getLogger(__name__)

class ExperienceParser:
    """Parses experience requirements from job description text."""

    def __init__(self):
        # Patterns for X+ years, X-Y years, and minimum years
        self.experience_patterns = [
            re.compile(r"(\d+)\s*\+\s*years?", re.IGNORECASE),  # e.g., 5+ years
            re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*years?", re.IGNORECASE),  # e.g., 2-5 years
            re.compile(r"minimum of (?:at least)?\s*(\d+)\s*years?", re.IGNORECASE),  # e.g., minimum of 3 years
            re.compile(r"(\d+)\s*years? (?:of|s(?: of)?|experience)", re.IGNORECASE),  # e.g., 3 years of experience
        ]
        self.seniority_keywords = {
            "junior": ["junior", "entry level"],
            "mid": ["mid-level", "intermediate"],
            "senior": ["senior", "lead", "staff", "principal", "architect"],
            "manager": ["manager", "head", "director"],
        }

    @memory.cache # Cache results for common experience descriptions
    def parse_experience(self, text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Extracts minimum, maximum experience and seniority level from the text.

        Args:
            text (str): The text section containing experience information.

        Returns:
            Tuple[Optional[int], Optional[int], Optional[str]]:
                A tuple containing (min_experience, max_experience, seniority_level).
        """
        min_exp: Optional[int] = None
        max_exp: Optional[int] = None
        seniority: Optional[str] = self._detect_seniority(text)

        for pattern in self.experience_patterns:
            for match in pattern.finditer(text):
                if len(match.groups()) == 1:
                    # X+ years or minimum X years
                    current_min = int(match.group(1))
                    if min_exp is None or current_min > min_exp:
                        min_exp = current_min
                elif len(match.groups()) == 2:
                    # X-Y years
                    current_min = int(match.group(1))
                    current_max = int(match.group(2))
                    # Always take the higher minimum and lower maximum for a range
                    if min_exp is None or current_min > min_exp:
                        min_exp = current_min
                    if max_exp is None or current_max < max_exp:
                        max_exp = current_max
                logger.debug(f"Intermediate experience: min={min_exp}, max={max_exp}")

        # Heuristic to refine seniority based on parsed years if not explicitly detected from keywords
        if seniority is None and (min_exp is not None or max_exp is not None):
            if min_exp is not None:
                if min_exp >= 7:
                    seniority = "senior"
                elif 3 <= min_exp <= 6:
                    seniority = "mid"
                elif min_exp <= 2:
                    seniority = "junior"
            elif max_exp is not None:
                if max_exp <= 2:
                    seniority = "junior"

        logger.debug(f"Final parsed experience: min={min_exp}, max={max_exp}, seniority={seniority}")
        return min_exp, max_exp, seniority

    def _detect_seniority(self, text: str) -> Optional[str]:
        """
        Detects seniority level based on predefined keywords.
        """
        text_lower = text.lower()
        for level, keywords in self.seniority_keywords.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                    return level
        return None
