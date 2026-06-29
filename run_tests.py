import os
import sys
import pytest

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# You can specify which tests to run, or let pytest discover them
# For example, to run all tests in the 'tests' directory:
pytest.main(["tests/"])
