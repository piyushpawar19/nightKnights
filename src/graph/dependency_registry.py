from typing import Any, Dict, Optional, Type

from src.interfaces.adapter_interfaces import (
    JDParserAdapter, DenseRetrievalAdapter, BM25Adapter, HybridRankerAdapter,
    RecruiterRerankerAdapter, SkillExtractorAdapter, ProfileBuilderAdapter,
    DatasetLoaderAdapter, VectorStoreAdapter, FeatureEngineeringAdapter, ModuleAdapter
)


class DependencyRegistry:
    """
    A registry mapping logical service names to their concrete or mock implementations
    of adapters.
    """

    def __init__(self):
        self._registry: Dict[str, Type[ModuleAdapter]] = {}

    def register_service(self, service_name: str, adapter_class: Type[ModuleAdapter]):
        """
        Registers an adapter class for a given logical service name.
        :param service_name: The logical name of the service (e.g., "JDParser").
        :param adapter_class: The concrete adapter class that implements the service.
        """
        self._registry[service_name] = adapter_class

    def get_service_adapter(self, service_name: str) -> Optional[Type[ModuleAdapter]]:
        """
        Retrieves the adapter class for a given logical service name.
        :param service_name: The logical name of the service.
        :return: The adapter class, or None if not found.
        """
        return self._registry.get(service_name)


# Global instance of the DependencyRegistry
def get_dependency_registry() -> DependencyRegistry:
    """
    Provides a singleton instance of the DependencyRegistry.
    """
    if not hasattr(get_dependency_registry, "_instance"):
        get_dependency_registry._instance = DependencyRegistry()
    return get_dependency_registry._instance
