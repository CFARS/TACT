from tact.core.config.reader import JsonConfigReader
from tact.core.ti.methods.basic import StandardTICalculator
from tact.core.ti.methods.reference import ReferenceTICalculator
from tact.core.ti.methods.representative import RepTICalculator
from tact.core.ti.processor import TIDataProcessor
from tact.core.statistics.processor import StatisticsProcessor
from tact.core.statistics.wind_speed import WindSpeedStatistics
from tact.core.binning.methods.wind_speed import WindSpeedBinning
from tact.core.binning.processor import BinningProcessor

def setup_processors(config_path: str) -> tuple:
    """Set up all processors with their strategies"""
    config_reader = JsonConfigReader(config_path)
    column_map = config_reader.get_column_mapping()
    wind_speed_col = column_map["reference"]["wind_speed"]

    # Binning setup
    binning_strategy = WindSpeedBinning(wind_speed_col)
    binning_processor = BinningProcessor(binning_strategy)

    # TI Calculator setup
    ti_calculator = StandardTICalculator(config_reader)
    reference_ti_calculator = ReferenceTICalculator(config_reader)
    rep_ti_calculator = RepTICalculator(ti_calculator)
    ti_data_processor = TIDataProcessor(ti_calculator, rep_ti_calculator, reference_ti_calculator)

    # Statistics setup
    stats_strategy = WindSpeedStatistics(min_speed=1.5, max_speed=21.0)
    stats_processor = StatisticsProcessor(stats_strategy)

    return binning_processor, ti_data_processor, stats_processor