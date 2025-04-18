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
from tact.core.ti.processor import TIDataProcessor
def main():
    #create TACT instance
    tact = TACT()

    # Load data
    data = pd.read_csv('tact/example/data/tact-test-data.csv')
    
    method = 'ss-sf'
    # method = 'baseline'
    
    # Define parameters for baseline adjustment
    parameters = {
        'config_path': 'tact/example/config.json',
        'split': True
    }
    
    # data = bin_by_wind_speed(data, parameters)
    # data = calculate_ti(data, parameters)
    
    config_reader = JsonConfigReader(parameters["config_path"])
    column_map = config_reader.get_column_mapping()
    wind_speed_col = column_map["reference"]["wind_speed"]
    
    binning_strategy = WindSpeedBinning(wind_speed_col)
    binning_processor = BinningProcessor(binning_strategy)
    data = binning_processor.process(data)
    
    ti_calculator = StandardTICalculator(config_reader)
    rep_ti_calculator = RepTICalculator(ti_calculator)
    
    ti_data_processor = TIDataProcessor(ti_calculator, rep_ti_calculator)
    data = ti_data_processor.process(data)
    
    
    
    
    results = tact.adjust(
        data=data,
        method=method,
        parameters=parameters
    )

    # Access adjustment results
    adjusted_data = results['adjusted_data']
    reg_results = results['reg_results']
    
    # Process statistics
    
    # TI_by_WS_bin = get_stats_per_WS_bin(adjusted_data, "adjTI_RSD_TI")
    
    # print(TI_by_WS_bin)
    
    # # Save results
    adjusted_data.to_csv(f'tact/example/output/{method}_adjusted_data.csv')
    reg_results.to_csv(f'tact/example/output/{method}_reg_results.csv')


if __name__ == "__main__":  
    main()