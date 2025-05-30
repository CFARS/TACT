import pandas as pd
from tact.core.binning.processor import BinningProcessor
from tact.core.ti.processor import TIDataProcessor

def process_data(
    data: pd.DataFrame,
    binning_processor: BinningProcessor,
    ti_processor: TIDataProcessor
) -> pd.DataFrame:
    """Process data through all processors in sequence"""
    data = binning_processor.process(data)
    data = ti_processor.process(data)
    return data