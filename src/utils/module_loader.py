import importlib.util
import logging
from types import ModuleType
from typing import Any, Dict, Optional, Type

from src.interfaces.adapter_interfaces import ModuleAdapter

logger = logging.getLogger(__name__)


class ModuleLoader:
    """
    Dynamically loads modules, handles optional module imports,
    version compatibility (placeholder), and provides mock implementations.
    """

    def __init__(self):
        self.loaded_modules: Dict[str, ModuleType] = {}
        self.mock_implementations: Dict[str, Any] = {}

    def set_mock_implementation(self, module_name: str, mock_module: Any):
        """
        Sets a mock implementation for a given module name.
        :param module_name: The name of the module to mock.
        :param mock_module: The mock object.
        """
        self.mock_implementations[module_name] = mock_module
        logger.info(f"Mock implementation set for \'{module_name}\'.")

    def load_module(self, module_name: str, package_path: str, use_mock: bool = False) -> Optional[Any]:
        """
        Dynamically loads a module, or returns a mock implementation.
        :param module_name: The name of the module to load (e.g., "profile_builder_agent").
        :param package_path: The Python package path to the module (e.g., "src.retrieval").
        :param use_mock: If True, explicitly tries to load the mock implementation.
        :return: The loaded module or its mock, or None if unavailable.
        """
        if use_mock and module_name in self.mock_implementations:
            logger.warning(f"Loading mock implementation for \'{module_name}\'.")
            return self.mock_implementations[module_name]

        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]

        full_module_path = f"{package_path}.{module_name}"
        try:
            spec = importlib.util.find_spec(full_module_path)
            if spec is None:
                raise ModuleNotFoundError(f"Module spec not found for {full_module_path}")

            module = importlib.util.module_from_spec(spec)
            if spec.loader:
                spec.loader.exec_module(module)
            else:
                raise ImportError(f"Module loader not found for {full_module_path}")

            self.loaded_modules[module_name] = module
            logger.info(f"Successfully loaded module: \'{full_module_path}\'.")
            return module
        except ModuleNotFoundError as e:
            logger.warning(f"Module \'{full_module_path}\' not found. {e}")
            if module_name in self.mock_implementations:
                logger.warning(f"Falling back to mock implementation for \'{module_name}\'.")
                return self.mock_implementations[module_name]
            else:
                logger.error(f"No mock implementation available for \'{module_name}\'.")
                return None
        except Exception as e:
            logger.error(f"Error loading module \'{full_module_path}\': {e}")
            return None

    def check_version_compatibility(self, module: Any, expected_version: str) -> bool:
        """
        Placeholder for checking module version compatibility.
        :param module: The loaded module.
        :param expected_version: The expected version string.
        :return: True if compatible, False otherwise.
        """
        logger.warning("Version compatibility check is a placeholder.")
        # In a real scenario, you'd check `module.__version__` or similar.
        return True

    def report_missing_modules(self):
        """
        Logs all modules that were attempted to load but could not be found
        and didn't have a mock fallback.
        """
        # This would be more effective if we tracked requested modules vs loaded/mocked.
        # For now, it's implicitly handled by the load_module logging.
        logger.info("Reporting missing modules (currently relies on load_module logging).")
