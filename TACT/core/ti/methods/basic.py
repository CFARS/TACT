import pandas as pd
from tact.core.config.reader import ConfigReader
from tact.core.ti.calculator import TICalculator

class StandardTICalculator(TICalculator):
    def __init__(self, config_reader: ConfigReader):
        self.config_reader = config_reader
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        column_map = self.config_reader.get_column_mapping()
        std_col = column_map["rsd"]["primary"]["standard_deviation"]
        ws_col = column_map["rsd"]["primary"]["wind_speed"]
        return data[std_col] / data[ws_col]