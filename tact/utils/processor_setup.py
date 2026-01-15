from tact.core.types import ProcessorConfig
from tact.core.binning.processor import BinningProcessor
from tact.core.ti.processor import TIDataProcessor
from tact.core.statistics.processor import StatisticsProcessor
from tact.core.config.reader import JsonConfigReader

def setup_processors(config_path: str, processor_config: ProcessorConfig) -> tuple:
    """Set up all processors based on configuration"""
    config_reader = JsonConfigReader(config_path)
    
    # Setup binning processor
    binning_processors = []
    for strategy_type, config in zip(
        processor_config.binning_strategies,
        processor_config.binning_configs
    ):
        strategy = strategy_type(**config)
        binning_processors.append(BinningProcessor(strategy))
    
    # Setup TI processor
    ti_calculators = []
    for strategy_type, config in zip(
        processor_config.ti_strategies,
        processor_config.ti_configs
    ):
        calculator = strategy_type(config_reader, **config)
        ti_calculators.append(calculator)
    ti_processor = TIDataProcessor(*ti_calculators)
    
    # Setup statistics processor
    stats_strategies = []
    for strategy_type, config in zip(
        processor_config.stats_strategies,
        processor_config.stats_configs
    ):
        strategy = strategy_type(**config)
        stats_strategies.append(strategy)
    stats_processor = StatisticsProcessor(stats_strategies)
    
    return binning_processors[0], ti_processor, stats_processor