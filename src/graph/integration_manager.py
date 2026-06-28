import importlib.util
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from src.state.pipeline_state import PipelineState
from src.interfaces.adapter_interfaces import ModuleAdapter

logger = logging.getLogger(__name__)


class IntegrationManager:
    """
    Manages the registration, validation, and invocation of external modules.
    Handles dependency injection, graceful degradation, and mock fallbacks.
    """

    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.mock_modules: Dict[str, Any] = {}
        self.adapters: Dict[str, Type[ModuleAdapter]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}

    def register_module(self, name: str, module: Any, is_mock: bool = False):
        """
        Registers an external module or a mock implementation.
        :param name: The unique name of the module.
        :param module: The module object.
        :param is_mock: True if this is a mock implementation.
        """
        if is_mock:
            self.mock_modules[name] = module
            logger.info(f"Mock module '{name}' registered.")
        else:
            self.modules[name] = module
            logger.info(f"Module '{name}' registered.")

    def register_adapter(self, name: str, adapter_class: Type[ModuleAdapter]):
        """
        Registers an adapter class for a given module.
        :param name: The name of the module the adapter belongs to.
        :param adapter_class: The class of the adapter.
        """
        self.adapters[name] = adapter_class
        logger.info(f"Adapter '{name}' registered.")

    def get_module(self, name: str) -> Optional[Any]:
        """
        Retrieves a module, falling back to a mock if the real module is unavailable.
        :param name: The name of the module to retrieve.
        :return: The module object or its mock, or None if neither is available.
        """
        if name in self.modules:
            logger.debug(f"Retrieving real module '{name}'.")
            return self.modules[name]
        elif name in self.mock_modules:
            logger.warning(f"Module '{name}' not available, falling back to mock implementation.")
            return self.mock_modules[name]
        else:
            logger.error(f"Module '{name}' and its mock implementation are not registered.")
            return None

    def get_adapter(self, name: str) -> Optional[ModuleAdapter]:
        """
        Retrieves and instantiates an adapter for a given module.
        :param name: The name of the module whose adapter is to be retrieved.
        :return: An instance of the adapter, or None if not found.
        """
        adapter_class = self.adapters.get(name)
        if adapter_class:
            module_instance = self.get_module(name)
            if module_instance:
                return adapter_class(module_instance)  # Adapters need the module instance
            else:
                logger.error(f"Cannot instantiate adapter '{name}': Module instance not found.")
                return None
        else:
            logger.error(f"Adapter for '{name}' not registered.")
            return None

    def validate_module_interface(self, name: str, expected_signature: Dict[str, Any]) -> bool:
        """
        Validates if a module conforms to an expected interface (e.g., function signatures).
        This is a placeholder and would require more sophisticated reflection.
        :param name: The name of the module to validate.
        :param expected_signature: A dictionary defining the expected interface.
        :return: True if the module's interface is compatible, False otherwise.
        """
        module = self.get_module(name)
        if not module:
            return False

        # Example: Check for a specific method and its callable nature
        for method_name, _ in expected_signature.items():  # Signature details can be used for deeper validation
            if not hasattr(module, method_name) or not callable(getattr(module, method_name)):
                logger.error(f"Module '{name}' is missing expected method '{method_name}' or it's not callable.")
                return False
        logger.info(f"Module '{name}' interface validated successfully.")
        return True

    def add_dependency(self, dependent_module: str, dependency: str):
        """
        Adds a dependency between modules for lifecycle management or ordering.
        :param dependent_module: The module that depends on another.
        :param dependency: The module that is a dependency.
        """
        if dependent_module not in self.dependency_graph:
            self.dependency_graph[dependent_module] = []
        self.dependency_graph[dependent_module].append(dependency)
        logger.info(f"Dependency added: '{dependent_module}' depends on '{dependency}'.")

    def resolve_dependencies(self) -> List[str]:
        """
        Resolves the order of modules based on their dependencies (placeholder).
        For a real system, this would implement a topological sort.
        :return: A list of module names in resolved order.
        """
        logger.warning("Dependency resolution is a placeholder. Implement topological sort for production.")
        return list(self.modules.keys())

    def lifecycle_init(self):
        """
        Initializes all registered modules (placeholder).
        """
        logger.info("Initializing modules (lifecycle_init placeholder).")
        for name, module in self.modules.items():
            if hasattr(module, 'initialize'):
                module.initialize()
                logger.debug(f"Module '{name}' initialized.")

    def lifecycle_shutdown(self):
        """
        Shuts down all registered modules (placeholder).
        """
        logger.info("Shutting down modules (lifecycle_shutdown placeholder).")
        for name, module in self.modules.items():
            if hasattr(module, 'shutdown'):
                module.shutdown()
                logger.debug(f"Module '{name}' shutdown.")
