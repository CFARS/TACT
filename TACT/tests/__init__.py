import os
import pytest


@pytest.fixture
def example_file_dir():
    return os.path.join(
        os.path.pardir(os.path.abspath(__file__)), "Example"
    )
