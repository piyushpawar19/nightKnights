"""Centralized configuration manager for the AI hiring system.

This module implements the ``ConfigManager``, a singleton responsible for
loading, validating, and providing access to application configuration.
It uses Pydantic models for strong typing and validation, and PyYAML for
parsing YAML configuration files. The manager supports lazy loading, caching,
and graceful error handling for missing or malformed configurations.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from src.interfaces.config_interface import ConfigProviderInterface
from src.models.config_models import (
    AppConfig,
    EvaluationConfig,
    ExportConfig,
    LLMConfig,
    RankingConfig,
    RetrievalConfig,
    BenchmarkConfig,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class YamlConfigProvider(ConfigProviderInterface):
    """Provides configuration by loading and validating YAML files.

    Parameters
    ----------
    config_dir : Path
        The directory where configuration YAML files are located.
    """

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self._config: AppConfig | None = None

    def _load_yaml_file(self, filename: str) -> dict[str, Any]:
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise ConfigError(f"Configuration file not found: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Malformed YAML in {filepath}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Error reading {filepath}: {e}") from e

    def load_config(self) -> AppConfig:
        """Load and validate all application configurations from YAML files."""
        logger.info(f"Loading configurations from {self.config_dir}")

        retrieval_data = self._load_yaml_file("retrieval.yaml")
        ranking_data = self._load_yaml_file("ranking.yaml")
        llm_data = self._load_yaml_file("llm.yaml")
        evaluation_data = self._load_yaml_file("evaluation.yaml")
        export_data = self._load_yaml_file("export.yaml")
        benchmark_data = self._load_yaml_file("benchmark.yaml")

        try:
            retrieval_config = RetrievalConfig(**retrieval_data)
            ranking_config = RankingConfig(**ranking_data)
            llm_config = LLMConfig(**llm_data)
            evaluation_config = EvaluationConfig(**evaluation_data)
            export_config = ExportConfig(**export_data)
            benchmark_config = BenchmarkConfig(**benchmark_data)

            self._config = AppConfig(
                retrieval=retrieval_config,
                ranking=ranking_config,
                llm=llm_config,
                evaluation=evaluation_config,
                export=export_config,
                benchmark=benchmark_config,
            )
            logger.info("Configuration loaded and validated successfully.")
            return self._config
        except ValidationError as e:
            raise ConfigError(f"Configuration validation failed: {e}") from e
        except Exception as e:
            raise ConfigError(f"Unknown error during config loading: {e}") from e

    def reload_config(self) -> AppConfig:
        """Reload configuration from YAML files."""
        self._config = None  # Clear cache
        logger.info("Reloading configuration.")
        return self.load_config()

    def get_config(self) -> AppConfig:
        """Get the currently loaded configuration, or load it if not cached."""
        if self._config is None:
            self._config = self.load_config()
        return self._config


class ConfigManager:
    """Singleton-style manager for accessing application configurations.

    This class ensures that configuration is loaded and validated only once,
    and provides typed accessors for various configuration sections.
    """

    _instance: ConfigManager | None = None
    _config_provider: ConfigProviderInterface | None = None

    def __new__(cls, config_dir: Path | None = None) -> ConfigManager:
        if cls._instance is None:
            if config_dir is None:
                raise ConfigError("config_dir must be provided for initial ConfigManager instantiation.")
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._config_provider = YamlConfigProvider(config_dir)
            cls._instance._app_config = cls._config_provider.load_config()
            logger.info("ConfigManager initialized and configuration loaded.")
        elif config_dir is not None and cls._config_provider is not None and cls._config_provider.config_dir != config_dir:
            # If config_dir changes, reload the configuration
            logger.warning(f"ConfigManager already initialized with {cls._config_provider.config_dir}, but new config_dir {config_dir} provided. Reloading configuration.")
            cls._config_provider = YamlConfigProvider(config_dir)
            cls._instance._app_config = cls._config_provider.load_config()
        return cls._instance

    def __init__(self, config_dir: Path | None = None) -> None:
        # The __init__ is called every time ConfigManager() is called,
        # but __new__ ensures it\'s a singleton. We can use __init__ for 
        # non-initialization logic if needed, but for a strict singleton
        # with cached config, most logic belongs in __new__.
        pass

    @classmethod
    def get_instance(cls) -> ConfigManager:
        if cls._instance is None:
            raise ConfigError("ConfigManager has not been initialized. Call ConfigManager(config_dir) first.")
        return cls._instance

    def reload_config(self) -> None:
        """Force a reload of all configurations from the source."""
        if self._config_provider is None:
            raise ConfigError("ConfigManager not initialized with a provider.")
        self._app_config = self._config_provider.reload_config()
        logger.info("Configuration reloaded successfully.")

    def get_app_config(self) -> AppConfig:
        """Get the full application configuration."""
        if self._app_config is None:
            raise ConfigError("AppConfig not loaded. Call load_config or ensure ConfigManager is initialized.")
        return self._app_config

    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval-specific configuration."""
        return self.get_app_config().retrieval

    def get_ranking_config(self) -> RankingConfig:
        """Get ranking-specific configuration."""
        return self.get_app_config().ranking

    def get_llm_config(self) -> LLMConfig:
        """Get LLM-specific configuration."""
        return self.get_app_config().llm

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation-specific configuration."""
        return self.get_app_config().evaluation

    def get_export_config(self) -> ExportConfig:
        """Get export-specific configuration."""
        return self.get_app_config().export

    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark-specific configuration."""
        return self.get_app_config().benchmark
