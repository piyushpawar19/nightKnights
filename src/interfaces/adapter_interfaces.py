from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.state.pipeline_state import PipelineState
from src.models.domain_models import JobDescription, Candidate, RankedCandidate, SearchResult


class ModuleAdapter(ABC):
    """
    Abstract base class for all module adapters.
    Defines the interface for converting PipelineState to module inputs,
    invoking the module, and converting outputs back to PipelineState.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Executes the external module through the adapter.
        :param state: The current state of the pipeline.
        :return: The updated pipeline state.
        """
        pass


class JDParserAdapter(ModuleAdapter):
    """
    Adapter for the JDParserAgent module.
    Converts PipelineState to JobDescription, invokes the agent,
    and updates PipelineState with the parsed JD.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Parses the job description.
        """
        pass


class SkillExtractorAdapter(ModuleAdapter):
    """
    Adapter for the SkillExtractorAgent module.
    Extracts skills from JD and candidates.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Extracts skills.
        """
        pass


class ProfileBuilderAdapter(ModuleAdapter):
    """
    Adapter for the ProfileBuilderAgent module.
    Builds comprehensive candidate profiles.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Builds candidate profiles.
        """
        pass


class DenseRetrievalAdapter(ModuleAdapter):
    """
    Adapter for the DenseRetrievalAgent module.
    Performs dense vector-based retrieval.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Performs dense retrieval.
        """
        pass


class BM25Adapter(ModuleAdapter):
    """
    Adapter for the BM25Agent module.
    Performs keyword-based BM25 retrieval.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Performs BM25 retrieval.
        """
        pass


class DatasetLoaderAdapter(ModuleAdapter):
    """
    Adapter for the DatasetLoader module.
    Loads the candidate dataset.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Loads the dataset.
        """
        pass


class VectorStoreAdapter(ModuleAdapter):
    """
    Adapter for the VectorStore module.
    Manages vector storage and retrieval.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Interacts with the vector store.
        """
        pass


class FeatureEngineeringAdapter(ModuleAdapter):
    """
    Adapter for the FeatureEngineeringAgent module.
    Generates features for ranking.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Performs feature engineering.
        """
        pass


class HybridRankerAdapter(ModuleAdapter):
    """
    Adapter for the HybridRankerAgent module.
    Combines multiple ranking signals.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Performs hybrid ranking.
        """
        pass


class RecruiterRerankerAdapter(ModuleAdapter):
    """
    Adapter for the RecruiterRerankerAgent module.
    Applies recruiter-specific reranking logic.
    """

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """
        Performs recruiter reranking.
        """
        pass
