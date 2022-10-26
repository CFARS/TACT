try:
    from TACT import logger
except ImportError:
    pass
import inspect
import pytest
from TACT.readers.data import Data
from TACT.tests.conftest import config_object_from_example_files


def test_read_data(config_object_from_example_files):
    logger.debug(f"running {inspect.getframeinfo(inspect.currentframe()).function}")
    config = config_object_from_example_files
    print(f"Config filename:\t{config.input_filename}")
    print(f"Results filename:\t{config.results_file}")

    data = Data(config)
    data.get_refTI_bins()
    data.check_for_alphaConfig()

    assert(len(data.inputdata) > 10)
