from tact.core.config.reader import JsonConfigReader
from tact.core.ti.methods.basic import StandardTICalculator
from tact.core.ti.methods.reference import ReferenceTICalculator
from tact.core.ti.methods.representative import RepTICalculator
from tact.core.ti.processor import TIDataProcessor
from tact.core.statistics.processor import StatisticsProcessor
from tact.core.statistics.wind_speed import WindSpeedStatistics
from tact.core.binning.methods.wind_speed import WindSpeedBinning
from tact.core.binning.processor import BinningProcessor
from typing import List, Type
from tact.core.statistics.strategy import StatisticsStrategy
from tact.core.types import ProcessorConfig

# def setup_statistics_processor(
#     strategy_configs: List[dict],
#     strategy_types: List[Type[StatisticsStrategy]]
# ) -> StatisticsProcessor:
#     """
#     Set up statistics processor with multiple strategies
#     """
#     strategies = []
#     for strategy_type, config in zip(strategy_types, strategy_configs):
#         strategy = strategy_type(**config)
#         strategies.append(strategy)
#     return StatisticsProcessor(strategies)

# def setup_processors(
#     config_path: str,
#     stats_configs: List[dict] = None,
#     stats_strategies: List[Type[StatisticsStrategy]] = None
# ) -> tuple:
#     """
#     Set up all processors with configurable statistics strategies
#     """
#     # Default configuration if none provided
#     if stats_configs is None:
#         stats_configs = [{"min_speed": 1.5, "max_speed": 21.0}]
#     if stats_strategies is None:
#         stats_strategies = [WindSpeedStatistics]
    
#     config_reader = JsonConfigReader(config_path)
#     column_map = config_reader.get_column_mapping()
#     wind_speed_col = column_map["reference"]["wind_speed"]

#     binning_strategy = WindSpeedBinning(wind_speed_col)
#     binning_processor = BinningProcessor(binning_strategy)

#     # TI Calculator setup
#     ti_calculator = StandardTICalculator(config_reader)
#     reference_ti_calculator = ReferenceTICalculator(config_reader)
#     rep_ti_calculator = RepTICalculator(ti_calculator)
#     ti_data_processor = TIDataProcessor(ti_calculator, rep_ti_calculator, reference_ti_calculator)
    
#     # Statistics setup with multiple possible strategies
#     stats_processor = setup_statistics_processor(stats_configs, stats_strategies)
    
#     return binning_processor, ti_data_processor, stats_processor

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




# def setup_processors(config_path: str, processor_config: ProcessorConfig) -> tuple:
#     """Set up all processors based on configuration"""
#     config_reader = JsonConfigReader(config_path)
    
#     # Setup binning processor
#     binning_processors = []
#     for strategy_type, config in zip(
#         processor_config.binning_strategies,
#         processor_config.binning_configs
#     ):
#         strategy = strategy_type(**config)
#         binning_processors.append(BinningProcessor(strategy))
    
#     # Setup TI processor
#     ti_calculators = []
#     for strategy_type, config in zip(
#         processor_config.ti_strategies,
#         processor_config.ti_configs
#     ):
#         calculator = strategy_type(config_reader, **config)
#         ti_calculators.append(calculator)
#     ti_processor = TIDataProcessor(*ti_calculators)
    
#     # Setup statistics processor
#     stats_strategies = []
#     for strategy_type, config in zip(
#         processor_config.stats_strategies,
#         processor_config.stats_configs
#     ):
#         strategy = strategy_type(**config)
#         stats_strategies.append(strategy)
#     stats_processor = StatisticsProcessor(stats_strategies)
    
#     return binning_processors[0], ti_processor, stats_processor