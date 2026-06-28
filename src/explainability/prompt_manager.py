import os
from typing import Dict, Any
from string import Template

from src.interfaces.explanation_interface import PromptManagerInterface
import logging

from src.utils.logger import get_logger


class PromptManager(PromptManagerInterface):
    """Manages loading, versioning, and injecting variables into prompt templates."""

    def __init__(self, prompts_dir: str, logger: logging.Logger):
        self.prompts_dir = prompts_dir
        self.logger = logger
        self._prompt_cache: Dict[str, str] = {}

    def _load_file_content(self, file_path: str) -> str:
        """Loads content from a specified file path."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Prompt file not found: {file_path}")
            raise ValueError(f"Prompt file not found: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading prompt file {file_path}: {e}")
            raise RuntimeError(f"Error loading prompt file {file_path}: {e}")

    def load_prompt(self, prompt_name: str) -> str:
        """Loads a prompt template by name from the configured prompts directory.
        Caches the prompt content to avoid redundant file reads.
        """
        if prompt_name in self._prompt_cache:
            self.logger.debug(f"Loading prompt \'{prompt_name}\' from cache.")
            return self._prompt_cache[prompt_name]

        prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        self.logger.info(f"Loading prompt \'{prompt_name}\' from file: {prompt_path}")
        content = self._load_file_content(prompt_path)
        self._prompt_cache[prompt_name] = content
        return content

    def inject_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Injects variables into a prompt template using string.Template.

        Args:
            template: The raw template string with placeholders (e.g., $variable_name).
            variables: A dictionary where keys are variable names and values are their replacements.

        Returns:
            The formatted prompt string.

        Raises:
            KeyError: If a placeholder in the template is not provided in variables.
        """
        try:
            self.logger.debug(f"Injecting variables into template. Variables: {list(variables.keys())}")
            return Template(template).substitute(**variables)
        except KeyError as e:
            self.logger.error(f"Missing variable in prompt template: {e}")
            raise ValueError(f"Missing variable in prompt template: {e}")
        except Exception as e:
            self.logger.error(f"Error injecting variables into prompt: {e}")
            raise RuntimeError(f"Error injecting variables into prompt: {e}")