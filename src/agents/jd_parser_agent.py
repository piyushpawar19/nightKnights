import logging
from typing import Dict, Any
from src.interfaces.jd_interface import JDParserInterface
from src.jd_parser.parser import JDParser
from src.schemas.jd_schema import ParsedJD

logger = logging.getLogger(__name__)

class JDParserAgent(JDParserInterface):
    """Agent that orchestrates the JD parsing process."""

    def __init__(self):
        self.parser = JDParser()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates input, parses the raw JD, validates the output schema,
        and updates the state with the parsed JD.

        Args:
            state (Dict[str, Any]): The current state dictionary containing 'raw_jd'.

        Returns:
            Dict[str, Any]: The updated state dictionary with 'parsed_jd' added.
        """
        self._validate_input(state)
        raw_jd = state["raw_jd"]

        logger.info("Starting JD parsing process...")
        parsed_jd = self.parser.parse_jd(raw_jd)
        logger.info("JD parsing completed.")

        self._validate_output(parsed_jd)

        state["parsed_jd"] = parsed_jd
        return state


