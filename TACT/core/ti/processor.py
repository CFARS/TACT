from tact.core.ti.calculator import TICalculator
import pandas as pd

class TIDataProcessor:
    def __init__(self, ti_calculator: TICalculator, rep_ti_calculator: TICalculator):
        self.ti_calculator = ti_calculator
        self.rep_ti_calculator = rep_ti_calculator
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data['RSD_TI'] = self.ti_calculator.calculate(data)
        data['RSD_Rep_TI'] = self.rep_ti_calculator.calculate(data)
        return data