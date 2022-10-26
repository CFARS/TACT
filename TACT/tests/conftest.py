from pathlib import Path
import pytest
from TACT.readers.config import Config


@pytest.fixture
def example_dir_path():
    return Path(__file__).parent.parent.parent / 'Example'


@pytest.fixture
def config_object_from_example_files(example_dir_path):
    input_filename = example_dir_path / "example_project.csv"
    config_filename = example_dir_path / "configuration_example_project.xlsx"
    results_filename = example_dir_path / "test_example.xlsx"
    print(f"RESULTS FILE: {results_filename}")

    config = Config(
        input_filename=input_filename,
        config_file=config_filename,
        results_file=results_filename,
        outpath_dir="." # Path(__file__).parent.parent.parent / 'test',
    )

    config.get_site_metadata()
    config.get_filtering_metadata()
    config.get_adjustments_metadata()

    return config
