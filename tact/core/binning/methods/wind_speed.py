from tact.core.binning.strategy import BinningStrategy
import pandas as pd

class WindSpeedBinning(BinningStrategy):
    
    """
    Binning strategy for wind speed.
    
    Must be given a wind speed column name upon initialization.
    
    Written 4/18/2025 - CJP
    """
    
    def __init__(self, wind_speed_col: str):
        self.wind_speed_col = wind_speed_col
    
    def create_bins(self, data: pd.DataFrame) -> pd.DataFrame:
        # Create 1 m/s bins
        data["bins"] = data[self.wind_speed_col].round(0)
        
        # Create 0.5 m/s bins
        bins_p5_interval = pd.interval_range(
            start=0.25, end=20, freq=0.5, closed="left"
        )
        out = pd.cut(x=data[self.wind_speed_col], bins=bins_p5_interval)
        data["bins_p5"] = out.apply(lambda x: x.mid)
        
        return data