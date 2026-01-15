import pandas as pd
from tact.core.statistics.processor import StatisticsProcessor
def process_statistics(adjusted_data: pd.DataFrame, stats_processor: StatisticsProcessor) -> pd.DataFrame:
    """Process statistics for all columns"""
    columns = ["RSD_TI", "RSD_Rep_TI", "Ref_TI", "adjTI_RSD_TI", "adjRepTI_RSD_RepTI"]
    headers = [f'mean_{i}' for i in range(2, 20)]
    means = pd.DataFrame(columns=headers)
    
    for column in columns:
        stats_1m, stats_05m = stats_processor.process(adjusted_data, column)
        values_list = stats_1m.values[0][0:18]
        means.loc[column] = values_list
    
    return means