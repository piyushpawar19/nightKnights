from typing import Dict, Any
from src.schemas.jd_schema import ParsedJD


class JDParserInterface:
    """Interface for the JD Parser Agent."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses a raw job description and updates the state with a strongly typed parsed_jd.

        Args:
            state (Dict[str, Any]): The current state dictionary containing 'raw_jd'.

        Returns:
            Dict[str, Any]: The updated state dictionary with 'parsed_jd' added.
        """
        raise NotImplementedError("The 'run' method must be implemented by subclasses.")

    def _validate_input(self, state: Dict[str, Any]):
        """
        Validates the input state for the presence of 'raw_jd'.
        """
        if "raw_jd" not in state or not isinstance(state["raw_jd"], str):
            raise ValueError("Input state must contain a 'raw_jd' string.")

    def _validate_output(self, parsed_jd: ParsedJD):
        """
        Validates the parsed JD against the ParsedJD schema.
        """
        if not isinstance(parsed_jd, ParsedJD):
            raise TypeError("Parsed JD must be an instance of ParsedJD schema.")
        # Pydantic's validation happens on instantiation, so this mostly checks type.
