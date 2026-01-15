
import pandas as pd
from tact.core.ti.calculator import TICalculator

class RepTICalculator(TICalculator):
    def __init__(self, ti_calculator: TICalculator):
        self.ti_calculator = ti_calculator
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        ti = self.ti_calculator.calculate(data)
        column_map = self.ti_calculator.config_reader.get_column_mapping()
        std_col = column_map["rsd"]["primary"]["standard_deviation"]
        return ti + 1.28 * data[std_col]