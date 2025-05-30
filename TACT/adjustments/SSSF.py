# tact/methods/baseline.py
import pandas as pd
from typing import Dict, Any, List
from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry
from tact.calculations.perform_ss_sf_adjustment_ported import perform_SS_SF_adjustment_ported
from tact.calculations.get_all_regressions import get_all_regressions
import numpy as np
import json


@AdjustmentRegistry.register("ss-sf")
class SSSF(AdjustmentMethod):
    def required_model_parameters(self) -> Dict[str, List]:
        return {
            "config_path": str,  # Path to configuration file
            "split": bool  # Whether to split the data into training and testing sets
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
        
    def perform_SS_SF_adjustment_ported(self, data, parameters):
    
        with open(parameters["config_path"], 'r') as f:
            config = json.load(f)
            column_map = config['input_data_column_mapping']
        

        results = pd.DataFrame(
            columns=[
                "sensor",
                "height",
                "adjustment",
                "m",
                "c",
                "rsquared",
                "difference",
                "mse",
                "rmse",
            ]
        )
        
        if parameters["split"]:
            inputdata_train = data[data["split"] == True].copy()
            inputdata_test = data[data["split"] == False].copy()
        
        filtered_Ref_TI = inputdata_train[column_map["reference.turbulence_intensity"]][inputdata_train[column_map["rsd.primary.turbulence_intensity"]] < 0.3] # This comparison might be a mistake?
        
        filtered_RSD_TI = inputdata_train[column_map["rsd.primary.turbulence_intensity"]][inputdata_train[column_map["rsd.primary.turbulence_intensity"]] < 0.3]
        full = pd.DataFrame()
        full["filt_Ref_TI"] = filtered_Ref_TI
        full["filt_RSD_TI"] = filtered_RSD_TI
        full = full.dropna()


        if len(full) < 2:
            results = self.post_adjustment_stats(
                [None],
                results,
                "Ref_TI",
                "adjTI_RSD_TI",
            )
            m = np.NaN
            c = np.NaN
        else:
            model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
            m = model[0]
            c = model[1]
            RSD_TI = inputdata_test[column_map["rsd.primary.turbulence_intensity"]].copy()
            RSD_TI = (float(model[0]) * RSD_TI) + float(model[1])
            inputdata_test["adjTI_RSD_TI"] = RSD_TI
            inputdata_test["adjRepTI_RSD_RepTI"] = (
                RSD_TI + 1.28 * inputdata_test["RSD_SD"]
            )
            results = self.post_adjustment_stats(
                inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
            )

        results["adjustment"] = ["SS-SF"] * len(results)
        results = results.drop(columns=["sensor", "height"])

        return inputdata_test, results, m, c

    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        
        # Validate parameters
        self.validate_parameters(parameters)

        # Validate data columns
        self.validate_data(data, parameters["config_path"])

        # Get regression results using existing function
        (adjusted_data, reg_results, m, c) = perform_SS_SF_adjustment_ported(
            data, parameters)

        return {
            "adjusted_data": adjusted_data,
            "reg_results": reg_results,
        }
