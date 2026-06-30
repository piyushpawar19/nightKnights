import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SalaryParser:
    """Parses salary information from job description text."""

    def __init__(self):
        self.salary_patterns = [
            # Matches formats like $X - $Y per annum, X-Y LPA, X-Y USD annual
            re.compile(r"(?:(\$|€|£|₹|rs\.?\s*))?(\d+(?:[\.,]\d{1,3})*)\s*(?:k|thousand)?\s*[-–]?\s*(?:(\$|€|£|₹|rs\.?\s*))?(\d+(?:[\.,]\d{1,3})*)\s*(?:k|thousand)?\s*(?:per annum|annual|yearly|p\.a\.|lpa|ctc)?", re.IGNORECASE),
            # Matches formats like $X,XXX - $Y,YYY per year
            re.compile(r"(?:(\$|€|£|₹|rs\.?\s*))?(\d{1,3}(?:[ ,]\d{3})*)\s*[-–]?\s*(?:(\$|€|£|₹|rs\.?\s*))?(\d{1,3}(?:[ ,]\d{3})*)\s*(?:per year|annually)?", re.IGNORECASE),
            # Matches single salary figures like $X,XXX per month, X LPA
            re.compile(r"(?:(\$|€|£|₹|rs\.?\s*))?(\d+(?:[\.,]\d{1,3})*)\s*(?:k|thousand)?\s*(?:per (?:month|year)|monthly|annually|p\.m\.|p\.a\.|lpa|ctc)?", re.IGNORECASE),
        ]
        self.currency_map = {
            "$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "rs": "INR", "inr": "INR"
        }
        self.period_keywords = {
            "per annum": "annual", "annual": "annual", "yearly": "annual", "p.a.": "annual", "lpa": "annual", "ctc": "annual",
            "per month": "monthly", "monthly": "monthly", "p.m.": "monthly"
        }

    def parse_salary(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts salary range, currency, and period from the text.

        Args:
            text (str): The text section containing salary information.

        Returns:
            Optional[Dict[str, Any]]:
                A dictionary with parsed salary details or None if not found.
        """
        text_lower = text.lower().replace(",", "") # Remove commas for easier parsing

        for pattern in self.salary_patterns:
            for match in pattern.finditer(text_lower):
                try:
                    groups = match.groups()
                    currency_symbol1 = groups[0]
                    min_salary_str = groups[1]
                    currency_symbol2 = groups[2]
                    max_salary_str = groups[3]
                    period_keyword = groups[4] if len(groups) > 4 else None # This might need adjustment based on patterns

                    currency = self._determine_currency(currency_symbol1 or currency_symbol2)
                    period = self._determine_period(match.group(0)) # Use full match to determine period

                    min_salary = self._parse_salary_value(min_salary_str, period)
                    max_salary = self._parse_salary_value(max_salary_str, period) if max_salary_str else min_salary

                    if min_salary is not None:
                        return {
                            "min_salary": min_salary,
                            "max_salary": max_salary,
                            "currency": currency,
                            "period": period,
                            "raw_text": match.group(0)
                        }
                except Exception as e:
                    logger.warning(f"Error parsing salary match \"{match.group(0)}\" - {e}")
                    continue

        return None

    def _parse_salary_value(self, value_str: str, period: Optional[str]) -> Optional[float]:
        """
        Converts a salary string to a float, handling 'k' for thousands.
        Adjusts value if LPA/CTC is used and implies a different scale.
        """
        try:
            value = float(value_str)
            # Heuristic: If period is annual and value is small, it might be in Lakhs (INR LPA)
            # This is a simplification and might need more robust currency/region detection.
            # We need to ensure we don't multiply already large numbers.
            if period == "annual" and value < 5000: # Assuming typical annual salaries are > 5000 without 'k' or 'LPA'
                if "lpa" in value_str.lower() or "ctc" in value_str.lower() or "lakh" in value_str.lower():
                    return value * 100000 # Convert Lakhs to actual value (e.g., 15 -> 1,500,000)

            # If 'k' or 'K' is explicitly present, multiply by 1000
            if re.search(r"\d+[kK]", value_str):
                return value * 1000

            return value
        except ValueError:
            return None

    def _determine_currency(self, symbol: Optional[str]) -> Optional[str]:
        """
        Determines the currency based on the symbol.
        """
        if symbol:
            return self.currency_map.get(symbol.lower().strip(), None)
        return None

    def _determine_period(self, text: str) -> Optional[str]:
        """
        Determines the salary period (annual/monthly) based on keywords in the text.
        """
        text_lower = text.lower()
        for keyword, period in self.period_keywords.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                return period
        # Default to annual if not specified, common for job descriptions
        return "annual"
