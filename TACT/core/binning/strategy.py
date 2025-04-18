from abc import ABC, abstractmethod
import pandas as pd

class BinningStrategy(ABC):
    
    """
    Abstract base class for binning strategies.
    
    This class defines the interface for binning strategies.
    All subclasses must implement the create_bins method.
    
    Currently, this is used only for wind speed binning, but it extends to other binning strategies.
    
    Written 4/18/2025 - CJP
    """
    @abstractmethod
    def create_bins(self, data: pd.DataFrame) -> pd.DataFrame:
        pass