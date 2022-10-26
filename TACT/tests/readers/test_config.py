from pathlib import Path
import pytest
import TACT.tests.conftest
from TACT.readers.config import Config


def test_read_config_valid_xlsx_creates_config(example_dir_path):
    config_filename = example_dir_path / "configuration_example_project.xlsx"
    data_filename = example_dir_path / "example_project.csv"
    results_filename = example_dir_path / "test_example.xlsx"

    config = Config(
        input_filename=data_filename,
        config_file=config_filename,
        results_file=results_filename
    )

    config.get_site_metadata()

    assert(len(config.site_metadata) > 0)


def test_read_config_json():
    pass
