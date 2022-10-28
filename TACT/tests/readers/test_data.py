try:
    from TACT import logger
except ImportError:
    pass
import inspect
from TACT.readers.data import Data
from TACT.tests.conftest import config_object_from_example_files


def test_read_data_using_config_object(config_object_from_example_files):
    # Arrange
    logger.debug(f"running {inspect.getframeinfo(inspect.currentframe()).function}")
    config = config_object_from_example_files
    data = Data(config=config)

    # Act
    data.get_inputdata()
    data.get_refTI_bins()
    data.check_for_alphaConfig()

    # Assert
    assert len(data.inputdata) > 10
    assert 'RefTI_bins' in data.inputdata.columns
    assert hasattr(data, "Ht_1_rsd")
    assert hasattr(data, "Ht_2_rsd")


def test_read_data_using_file_parameters(example_dir_path):
    # Arrange
    logger.debug(f"running {inspect.getframeinfo(inspect.currentframe()).function}")
    data = Data(
        input_filename=example_dir_path / "example_project.csv",
        config_file=example_dir_path / "configuration_example_project.xlsx",
        results_file=example_dir_path / "test_example.xlsx",
        outpath_dir=".",
    )

    # Act
    data.get_inputdata()
    data.get_refTI_bins()
    data.check_for_alphaConfig()

    # Assert
    assert len(data.inputdata) > 10
    assert 'RefTI_bins' in data.inputdata.columns
    assert hasattr(data, "Ht_1_rsd")
    assert hasattr(data, "Ht_2_rsd")
