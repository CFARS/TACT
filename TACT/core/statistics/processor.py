from tact.core.statistics.strategy import StatisticsStrategy
import pandas as pd

class StatisticsProcessor:
    def __init__(self, statistics_strategy: StatisticsStrategy):
        self.statistics_strategy = statistics_strategy
    
    def process(self, data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.statistics_strategy.calculate(data, column)