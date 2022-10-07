import os
import pytest
from TACT.readers.config import Config


example_fileroot = os.path.join(
    os.path.pardir(os.path.pardir(os.path.abspath(__file__))), "Example"
)

def test_read_config_valid_xlsx_creates object():
    config_filename = os.path.join(example_fileroot, "configuration_example_project.xlsx")
    data_filename = os.path.join(example_fileroot, "example_project.csv")
    results_filename = os.path.join(example_fileroot, "test_example.xlsx")

    config = Config(
        input_filename=data_filename,
        config_file=config_filename,
        results_file=results_filename
    )

    config.get_site_metadata()
    
    assert(len(config.site_metadata) > 0)


def test_read_config_json():
    pass


"""
if __name__ == "__main__":
    # Python 2 caveat: Only working for Python 3 currently
    if sys.version_info[0] < 3:
        raise Exception(
            "Tool will not run at this time. You must be using Python 3, as running on Python 2 will encounter errors."
        )
    # ------------------------
    # set up and configuration
    # ------------------------
    parser get_input_files
    config = Config()

    metadata parser
    config.get_site_metadata()
    config.get_filtering_metadata()
    config.get_adjustments_metadata()
"""
