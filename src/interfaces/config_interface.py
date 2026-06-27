"""Interface contracts (Protocols) for configuration providers.

This module defines the ``ConfigProviderInterface``, which specifies the
contract for any class responsible for loading and managing application
configuration. This promotes dependency inversion, allowing the
``ConfigManager`` to depend on an abstraction rather than concrete
implementations (e.g., YAML files, environment variables, cloud services).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from src.models.config_models import AppConfig


@runtime_checkable
class ConfigProviderInterface(Protocol):
    """Abstract interface for configuration providers.

    Any class implementing this protocol can be used by the ``ConfigManager``
    to load and manage application settings.
    """

    @abstractmethod
    def load_config(self) -> AppConfig:
        """Load configuration data and return a validated ``AppConfig`` object.

        Implementations should handle reading from their specific source
        (e.g., YAML files, environment variables) and perform any necessary
        initialization or parsing before Pydantic validation.

        Returns
        -------
        AppConfig
            A fully validated and typed application configuration object.

        Raises
        ------
        ConfigError
            If configuration cannot be loaded or is invalid.
        """
        ...

    @abstractmethod
    def reload_config(self) -> AppConfig:
        """Reload configuration data from its source.

        This method should re-read the configuration from the source,
        re-validate it, and return the updated ``AppConfig`` object.

        Returns
        -------
        AppConfig
            A freshly loaded and validated application configuration object.

        Raises
        ------
        ConfigError
            If configuration cannot be reloaded or is invalid.
        """
        ...

    @abstractmethod
    def get_config(self) -> AppConfig:
        """Get the currently loaded configuration.

        This method should return the last successfully loaded and validated
        ``AppConfig`` object. It should *not* attempt to reload the config
        from the source.

        Returns
        -------
        AppConfig
            The current application configuration object.

        Raises
        ------
        ConfigError
            If no configuration has been loaded yet.
        """
        ...
