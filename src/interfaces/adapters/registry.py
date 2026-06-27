"""Re-export graph adapter registry for interface consumers."""

from src.graph.dependency_registry import DependencyRegistry, get_dependency_registry

__all__ = ["DependencyRegistry", "get_dependency_registry"]
