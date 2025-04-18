from abc import ABC, abstractmethod
import pandas as pd

class TICalculator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        pass

