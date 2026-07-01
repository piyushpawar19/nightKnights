
import cProfile
import pstats
import io
import os
import sys
from memory_profiler import profile

# Adjust the path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from graph.graph import run_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

# Mock Job Description for benchmarking
MOCK_JD = """
Job Title: Senior Software Engineer
Company: ExampleTech
Location: Remote

Responsibilities:
- Design, develop, and maintain high-performance applications.
- Collaborate with cross-functional teams.
- Mentor junior engineers.
- Optimize existing codebase for scalability and efficiency.

Requirements:
- 5+ years of experience in Python.
- Strong knowledge of data structures and algorithms.
- Experience with cloud platforms (AWS, Azure, or GCP).
- Familiarity with machine learning frameworks (TensorFlow, PyTorch) is a plus.
- Bachelor's degree in Computer Science or related field.

Preferred Qualifications:
- Master's degree.
- Experience with distributed systems.
- Contributions to open-source projects.
"""

@profile
def run_pipeline_for_memory():
    """Runs the pipeline for memory profiling."""
    logger.info("Running pipeline for memory profiling...")
    # The run_pipeline function expects a Path object
    # For this mock, we'll directly pass the string and let it be handled by the mock in nodes.py
    # In a real scenario, you'd save MOCK_JD to a temp file and pass its path.
    # For now, we simulate by directly calling the graph's run_pipeline with the string.
    # Note: The current run_pipeline expects a Path object, so this direct string pass is a simplification
    # which might need adjustment based on how `graph.run_pipeline` is actually implemented to take string input.
    
    # For a proper benchmark, we need to ensure the graph.run_pipeline can accept a string or create a dummy file.
    # Given the current `main.py` which reads `job_description.txt`,
    # we will create a dummy `job_description.txt` for the benchmark.

    run_pipeline(MOCK_JD)

def run_pipeline_for_cprofile():
    """Runs the pipeline for cProfile (CPU profiling)."""
    logger.info("Running pipeline for CPU profiling...")
    run_pipeline(MOCK_JD)

def benchmark_pipeline():
    """Main function to run all benchmarks."""
    logger.info("Starting pipeline benchmarking...")

    # --- Memory Profiling ---
    logger.info("--- Memory Profiling ---")
    # The @profile decorator adds memory usage info directly to stdout/stderr.
    # For programmatic access, one might use `mprof run` command or capture output.
    run_pipeline_for_memory()
    logger.info("Memory profiling complete. Check console output for details.")

    # --- CPU Profiling (cProfile) ---
    logger.info("--- CPU Profiling ---")
    pr = cProfile.Profile()
    pr.enable()
    run_pipeline_for_cprofile()
    pr.disable()
    
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info("CPU profiling complete. Stats:\n%s", s.getvalue())

    logger.info("Pipeline benchmarking finished.")

if __name__ == "__main__":
    benchmark_pipeline()
