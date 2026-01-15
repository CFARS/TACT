# tact/core/types.py
from dataclasses import dataclass
from typing import Type, List
from tact.core.binning.strategy import BinningStrategy
from tact.core.ti.methods.basic import TICalculator
from tact.core.statistics.strategy import StatisticsStrategy

@dataclass
class ProcessorConfig:
    """Configuration for all processors"""
    binning_strategies: List[Type['BinningStrategy']]
    binning_configs: List[dict]
    ti_strategies: List[Type['TICalculator']]
    ti_configs: List[dict]
    stats_strategies: List[Type['StatisticsStrategy']]
    stats_configs: List[dict]