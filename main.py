from tact import TACT
from tact.methods.baseline import BaselineResults
from tact.methods.SSSF import SSSF
import pandas as pd
from tact.utils.get_TI_by_bin import get_TI_by_bin
from tact.utils.get_stats_per_WS_bin import get_stats_per_WS_bin
from tact.utils.bin_by_wind_speed import bin_by_wind_speed
from tact.utils.calculate_ti import calculate_ti


from tact.core.binning.processor import BinningProcessor
from tact.core.binning.methods.wind_speed import WindSpeedBinning

from tact.core.config.reader import JsonConfigReader
from tact.core.ti.methods.basic import StandardTICalculator
from tact.core.ti.methods.representative import RepTICalculator
from tact.core.ti.methods.reference import ReferenceTICalculator
from tact.core.ti.processor import TIDataProcessor
from tact.core.statistics.processor import StatisticsProcessor
from tact.core.statistics.wind_speed import WindSpeedStatistics


def main():
    # create TACT instance
    tact = TACT()

    # Load data
    data = pd.read_csv("tact/example/data/tact-test-data.csv")

    # Define parameters for baseline adjustment
    parameters = {"config_path": "tact/example/config.json", "split": True}

    config_reader = JsonConfigReader(parameters["config_path"])
    column_map = config_reader.get_column_mapping()
    wind_speed_col = column_map["reference"]["wind_speed"]

    binning_strategy = WindSpeedBinning(wind_speed_col)
    binning_processor = BinningProcessor(binning_strategy)
    data = binning_processor.process(data)

    ti_calculator = StandardTICalculator(config_reader)
    
    reference_ti_calculator = ReferenceTICalculator(config_reader)
    rep_ti_calculator = RepTICalculator(ti_calculator)
    ti_data_processor = TIDataProcessor(ti_calculator, rep_ti_calculator, reference_ti_calculator)
    data = ti_data_processor.process(data)

    method = "ss-sf"
    # method = 'baseline'

    results = tact.adjust(data=data, method=method, parameters=parameters)

    # Access adjustment results
    adjusted_data = results["adjusted_data"]
    reg_results = results["reg_results"]

    # Process statistics
    stats_strategy = WindSpeedStatistics(min_speed=1.5, max_speed=21.0)
    stats_processor = StatisticsProcessor(stats_strategy)

    # Calculate statistics
    columns = ["RSD_TI", "RSD_Rep_TI", "Ref_TI", "adjTI_RSD_TI", "adjRepTI_RSD_RepTI"]
    headers = [f'mean_{i}' for i in range(2, 20)] 
    means = pd.DataFrame(columns=headers)
    
    for column in columns:
        stats_1m, stats_05m = stats_processor.process(adjusted_data, column)
        values_list = stats_1m.values[0][0:18]
        means.loc[column] = values_list
        
    # # Save results
    means.to_csv(f"tact/example/output/{method}_all_stats.csv")
    adjusted_data.to_csv(f"tact/example/output/{method}_adjusted_data.csv")
    reg_results.to_csv(f"tact/example/output/{method}_reg_results.csv")

    
if __name__ == "__main__":
    main()
