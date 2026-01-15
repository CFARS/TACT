# tact/methods/baseline.py
import pandas as pd
from typing import Dict, Any, List
from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry
import json
from tact.calculations.get_all_regressions import get_all_regressions


@AdjustmentRegistry.register("baseline")
class BaselineResults(AdjustmentMethod):
    def required_model_parameters(self) -> Dict[str, List]:
        return {
            "config_path": str  # Path to configuration file
        }
        
    def required_data_columns(self) -> List[str]:
        return [
            "reference.wind_speed",
            "reference.standard_deviation",
            "reference.turbulence_intensity",
            "anemometer_2.primary.wind_speed",
            "anemometer_2.primary.standard_deviation",
            "anemometer_2.primary.turbulence_intensity",
            "rsd.height_1.wind_speed",
            "rsd.height_1.turbulence_intensity",
            "rsd.height_1.standard_deviation",
        ]

    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        
        # Validate parameters
        self.validate_parameters(parameters)

        # Validate data columns
        self.validate_data(data, parameters["config_path"])

        # Get regression results using existing function
        reg_results = get_all_regressions(
            data, parameters["config_path"], title="Baseline Results"
        )

        return {
            "adjusted_data": data,  # Since baseline doesn't actually adjust the data
            "reg_results": reg_results,
        }
