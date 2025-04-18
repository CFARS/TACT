from tact.core.binning.strategy import BinningStrategy
import pandas as pd

class BinningProcessor:
    
    """
    Generic processor for binning data.
    
    Must be given a binning strategy upon initialization.
    
    Written 4/18/2025 - CJP
    """
    
    def __init__(self, binning_strategy: BinningStrategy):
        self.binning_strategy = binning_strategy
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.binning_strategy.create_bins(data)