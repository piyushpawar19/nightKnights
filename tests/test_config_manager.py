import pytest
from pathlib import Path
import os
import yaml

from src.utils.config_manager import ConfigManager, ConfigError, YamlConfigProvider
from src.models.config_models import AppConfig, RetrievalConfig, RankingConfig, LLMConfig, EvaluationConfig

# Define a temporary config directory for testing
TEST_CONFIG_DIR = Path("tests/temp_configs")

@pytest.fixture(scope="module")
def setup_temp_configs():
    """Set up temporary config files for testing and clean up afterwards."""
    TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Create dummy YAML files
    (TEST_CONFIG_DIR / "retrieval.yaml").write_text(
        """
        embedding_model: test_model
        embedding_dimension: 128
        faiss_index_type: HNSW
        dense_top_k: 10
        bm25_top_k: 10
        hybrid_top_k: 20
        cache_embeddings: false
        batch_size: 16
        """
    )
    (TEST_CONFIG_DIR / "ranking.yaml").write_text(
        """
        dense_weight: 0.3
        bm25_weight: 0.7
        feature_weight: 0.5
        rerank_top_k: 5
        feature_thresholds:
            skill_match: 0.6
        normalization_method: z_score
        """
    )
    (TEST_CONFIG_DIR / "llm.yaml").write_text(
        """
        provider: test_llm_provider
        model_name: test_llm_model
        temperature: 0.5
        max_tokens: 512
        timeout: 30.0
        retry_attempts: 2
        prompt_directory: /tmp/prompts
        """
    )
    (TEST_CONFIG_DIR / "evaluation.yaml").write_text(
        """
        output_dir: outputs/reports
        enabled_metrics:
          - accuracy
          - f1
        k_values:
          recall_k: 5
          ndcg_k: 5
          precision_k: 3
        report_formats:
          - json
        pipeline_version: "1.0.0"
        """
    )
    (TEST_CONFIG_DIR / "export.yaml").write_text(
        """
        output_dir: outputs/submissions
        filename: submission.csv
        delimiter: ","
        quotechar: '"'
        quoting: 0
        encoding: utf-8
        overwrite: false
        export_schema:
          - candidate_id
          - rank
        """
    )
    (TEST_CONFIG_DIR / "benchmark.yaml").write_text(
        """
        base_output_dir: outputs/benchmarks
        benchmark_runs: {}
        default_k_values:
          recall_k: 5
          precision_k: 5
          ndcg_k: 5
          top_k_accuracy_k: 3
        report_format: json
        """
    )
    yield
    # Teardown: remove temporary config files and directory
    for f in TEST_CONFIG_DIR.iterdir():
        f.unlink()
    TEST_CONFIG_DIR.rmdir()

@pytest.fixture(scope="function", autouse=True)
def reset_config_manager_singleton():
    """Reset the ConfigManager singleton before each test to ensure isolation."""
    ConfigManager._instance = None
    ConfigManager._config_provider = None
    yield

def test_config_manager_initialization(setup_temp_configs):
    """Test that ConfigManager initializes and loads config correctly."""
    config_manager = ConfigManager(TEST_CONFIG_DIR)
    assert config_manager is not None
    assert isinstance(config_manager.get_app_config(), AppConfig)

def test_config_manager_singleton_behavior(setup_temp_configs):
    """Test that ConfigManager is a singleton."""
    config1 = ConfigManager(TEST_CONFIG_DIR)
    config2 = ConfigManager(TEST_CONFIG_DIR)
    assert config1 is config2

def test_get_sub_configs(setup_temp_configs):
    """Test retrieving specific sub-configurations."""
    config_manager = ConfigManager(TEST_CONFIG_DIR)

    retrieval_config = config_manager.get_retrieval_config()
    assert isinstance(retrieval_config, RetrievalConfig)
    assert retrieval_config.embedding_model == "test_model"
    assert retrieval_config.embedding_dimension == 128

    ranking_config = config_manager.get_ranking_config()
    assert isinstance(ranking_config, RankingConfig)
    assert ranking_config.dense_weight == 0.3

    llm_config = config_manager.get_llm_config()
    assert isinstance(llm_config, LLMConfig)
    assert llm_config.provider == "test_llm_provider"

    evaluation_config = config_manager.get_evaluation_config()
    assert isinstance(evaluation_config, EvaluationConfig)
    assert "accuracy" in evaluation_config.enabled_metrics

def test_reload_config(setup_temp_configs):
    """Test reloading configuration and verifying changes."""
    config_manager = ConfigManager(TEST_CONFIG_DIR)
    initial_llm_model = config_manager.get_llm_config().model_name

    # Modify a config file directly
    (TEST_CONFIG_DIR / "llm.yaml").write_text(
        """
        provider: new_provider
        model_name: new_llm_model
        temperature: 0.8
        max_tokens: 2048
        timeout: 90.0
        retry_attempts: 5
        prompt_directory: /tmp/new_prompts
        """
    )

    config_manager.reload_config()
    reloaded_llm_model = config_manager.get_llm_config().model_name
    assert initial_llm_model != reloaded_llm_model
    assert reloaded_llm_model == "new_llm_model"

def test_missing_config_file_error():
    """Test error handling for a missing configuration file."""
    # Create a temporary directory but no config files inside
    missing_config_dir = Path("tests/missing_configs")
    missing_config_dir.mkdir(parents=True, exist_ok=True)

    try:
        with pytest.raises(ConfigError, match="Configuration file not found:.*"): # type: ignore
            ConfigManager(missing_config_dir)
    finally:
        missing_config_dir.rmdir()

def test_malformed_yaml_error(setup_temp_configs):
    """Test error handling for malformed YAML syntax."""
    retrieval_path = TEST_CONFIG_DIR / "retrieval.yaml"
    original = retrieval_path.read_text(encoding="utf-8")
    retrieval_path.write_text("embedding_model: [unclosed\n")
    try:
        with pytest.raises(ConfigError, match="Malformed YAML in.*|Configuration validation failed.*"):  # type: ignore
            ConfigManager(TEST_CONFIG_DIR).reload_config()
    finally:
        retrieval_path.write_text(original, encoding="utf-8")

def test_invalid_schema_error(setup_temp_configs):
    """Test error handling for configuration not matching Pydantic schema."""
    retrieval_path = TEST_CONFIG_DIR / "retrieval.yaml"
    original = retrieval_path.read_text(encoding="utf-8")
    retrieval_path.write_text(
        """
        embedding_model: test_model
        embedding_dimension: invalid_int
        faiss_index_type: HNSW
        dense_top_k: 10
        bm25_top_k: 10
        hybrid_top_k: 20
        cache_embeddings: false
        batch_size: 16
        """
    )
    try:
        with pytest.raises(ConfigError, match="Configuration validation failed.*"):  # type: ignore
            ConfigManager(TEST_CONFIG_DIR).reload_config()
    finally:
        retrieval_path.write_text(original, encoding="utf-8")

def test_config_manager_not_initialized_error():
    """Test accessing ConfigManager before initialization."""
    # Ensure singleton is reset
    ConfigManager._instance = None
    ConfigManager._config_provider = None

    with pytest.raises(ConfigError, match="ConfigManager has not been initialized.*"): # type: ignore
        ConfigManager.get_instance()

def test_config_dir_change_reinitializes(setup_temp_configs):
    """Test that ConfigManager reinitializes if config_dir changes."""
    config_manager_initial = ConfigManager(TEST_CONFIG_DIR)
    initial_retrieval_model = config_manager_initial.get_retrieval_config().embedding_model

    # Create a new temporary config directory and config file
    NEW_TEST_CONFIG_DIR = Path("tests/new_temp_configs")
    NEW_TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_TEST_CONFIG_DIR / "retrieval.yaml").write_text(
        """
        embedding_model: new_test_model
        embedding_dimension: 256
        faiss_index_type: Flat
        dense_top_k: 5
        bm25_top_k: 5
        hybrid_top_k: 10
        cache_embeddings: true
        batch_size: 8
        """
    )
    (NEW_TEST_CONFIG_DIR / "ranking.yaml").write_text(
        """
        dense_weight: 0.6
        bm25_weight: 0.4
        feature_weight: 0.3
        rerank_top_k: 10
        normalization_method: log_norm
        """
    )
    (NEW_TEST_CONFIG_DIR / "llm.yaml").write_text(
        """
        provider: new_llm_provider
        model_name: new_gemini
        temperature: 0.9
        max_tokens: 512
        timeout: 45.0
        retry_attempts: 1
        prompt_directory: /new_prompts
        """
    )
    (NEW_TEST_CONFIG_DIR / "evaluation.yaml").write_text(
        """
        output_dir: outputs/reports
        enabled_metrics:
          - precision
        k_values:
          recall_k: 3
          ndcg_k: 3
          precision_k: 1
        report_formats:
          - json
        pipeline_version: "1.0.0"
        """
    )
    (NEW_TEST_CONFIG_DIR / "export.yaml").write_text(
        """
        output_dir: outputs/submissions
        filename: submission.csv
        delimiter: ","
        quotechar: '"'
        quoting: 0
        encoding: utf-8
        overwrite: false
        export_schema:
          - candidate_id
        """
    )
    (NEW_TEST_CONFIG_DIR / "benchmark.yaml").write_text(
        """
        base_output_dir: outputs/benchmarks
        benchmark_runs: {}
        default_k_values:
          recall_k: 3
          precision_k: 3
          ndcg_k: 3
          top_k_accuracy_k: 1
        report_format: json
        """
    )

    config_manager_new = ConfigManager(NEW_TEST_CONFIG_DIR)
    new_retrieval_model = config_manager_new.get_retrieval_config().embedding_model

    assert initial_retrieval_model != new_retrieval_model
    assert new_retrieval_model == "new_test_model"
    assert config_manager_initial is config_manager_new # Still the same singleton instance

    # Clean up new temp configs
    for f in NEW_TEST_CONFIG_DIR.iterdir():
        f.unlink()
    NEW_TEST_CONFIG_DIR.rmdir()
