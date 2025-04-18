from tact.core.statistics.strategy import StatisticsStrategy
import pandas as pd

class WindSpeedStatistics(StatisticsStrategy):
    def __init__(self, min_speed: float = 1.5, max_speed: float = 21.0):
        self.min_speed = min_speed
        self.max_speed = max_speed
    
    def calculate(self, data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Filter data by wind speed range
        filtered_data = data[
            (data["bins_p5"].astype(float) > self.min_speed)
            & (data["bins_p5"].astype(float) < self.max_speed)
        ]
        
        # Calculate statistics for 1 m/s bins
        stats_1m = (
            filtered_data[[column, "bins"]]
            .groupby(by="bins")
            .agg(["mean", "std"])
        )
        stats_1m = pd.DataFrame(stats_1m.unstack()).T
        stats_1m.index = [column]
        
        # Calculate statistics for 0.5 m/s bins
        stats_05m = (
            filtered_data[[column, "bins_p5"]]
            .groupby(by="bins_p5")
            .agg(["mean", "std"])
        )
        stats_05m = pd.DataFrame(stats_05m.unstack()).T
        stats_05m.index = [column]
        
        return stats_1m, stats_05m