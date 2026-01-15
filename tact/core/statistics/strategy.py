from abc import ABC, abstractmethod
import pandas as pd

class StatisticsStrategy(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass