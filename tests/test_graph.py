import pytest
from src.graph.graph import build_graph, run_pipeline
from src.schemas.graph_schema import NodeError, NodeStatus, NodeTimestamp
from src.graph.router import RouteDecision
from src.state.pipeline_state import PipelineState, create_initial_state

@pytest.mark.skip(reason="Outdated")
def test_graph_compilation():
    """Test that the LangGraph graph can be built and compiled without errors."""
    graph = build_graph()
    assert graph is not None
    assert isinstance(graph.nodes, dict)
    assert len(graph.nodes) > 0


@pytest.mark.skip(reason="Outdated")
def test_state_initialization():
    """Test that the initial pipeline state is created correctly."""
    raw_jd = "Software Engineer with 5+ years experience in Python and ML."
    initial_state = create_initial_state(raw_jd)

    assert initial_state["raw_jd"] == raw_jd
    assert initial_state["structured_jd"] is None
    assert isinstance(initial_state["extracted_skills"], list)
    assert len(initial_state["errors"]) == 0
    assert isinstance(initial_state["execution_metadata"], dict)
    assert "run_id" in initial_state["execution_metadata"]


@pytest.mark.skip(reason="Outdated")
def test_pipeline_execution_mock_nodes():
    """Test end-to-end pipeline execution with mock nodes."""
    raw_jd = "Senior Machine Learning Engineer, 7 years experience, Python, PyTorch."
    final_state = run_pipeline(raw_jd)

    assert final_state is not None
    assert final_state["raw_jd"] == raw_jd
    assert final_state["structured_jd"] is not None
    assert final_state["extracted_skills"]
    assert final_state["retrieved_candidates"]
    assert final_state["feature_vectors"]
    assert final_state["ranked_candidates"]
    assert final_state["reranked_candidates"]
    assert final_state["explanations"]
    assert final_state["evaluation_metrics"] is not None
    assert final_state["submission_path"] is not None
    assert not final_state["errors"]
    assert len(final_state["timestamps"]) > 0

    # Verify CSV file existence (optional, but good for end-to-end)
    import os
    assert os.path.exists(final_state["submission_path"])
    assert os.path.getsize(final_state["submission_path"]) > 0

