"""
Site-Specific Wind Speed (SS-WS) Adjustment Method

This method adjusts the RSD wind speed first using linear regression,
then recalculates turbulence intensity using the adjusted wind speed.

TI = SD / WS_adjusted

This approach is more physically sound than directly adjusting TI values.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry
from tact.calculations.get_regression import get_regression
from tact.calculations.train_test_split import train_test_split
from tact.calculations.post_adjustment_stats import post_adjustment_stats
import json


@AdjustmentRegistry.register("ssws")
class SSWS(AdjustmentMethod):
    """
    Site-Specific Wind Speed (SS-WS) adjustment method.

    This method:
    1. Performs train/test split (80/20)
    2. Trains linear regression on wind speed: Ref_WS = m * RSD_WS + c
    3. Adjusts RSD wind speed in test data
    4. Recalculates TI = SD / adjusted_WS
    5. Calculates representative TI = TI + 1.28 * SD

    Parameters
    ----------
    config_path : str
        Path to configuration JSON file with column mappings
    """

    def __init__(self):
        super().__init__()
        self.name = "ssws"
        self.description = "Site-Specific Wind Speed adjustment"

    def required_model_parameters(self) -> Dict[str, type]:
        """Return required parameters for SSWS adjustment."""
        return {
            "config_path": str,
            "split": bool
        }

    def required_data_columns(self) -> list:
        """Return required data columns for SSWS adjustment."""
        return [
            "reference.wind_speed",
            "reference.standard_deviation",
            "reference.turbulence_intensity",
            "rsd.primary.wind_speed",
            "rsd.primary.standard_deviation",
            "rsd.primary.turbulence_intensity"
        ]

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required parameters for SS-WS adjustment."""
        required = ["config_path"]
        for param in required:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")

    def validate_data(self, data: pd.DataFrame, config_path: str) -> None:
        """Validate that required columns exist in the data."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        col_map = config["input_data_column_mapping"]

        required_cols = [
            col_map["reference"]["wind_speed"],
            col_map["reference"]["standard_deviation"],
            col_map["reference"]["turbulence_intensity"],
            col_map["rsd"]["primary"]["wind_speed"],
            col_map["rsd"]["primary"]["standard_deviation"],
            col_map["rsd"]["primary"]["turbulence_intensity"]
        ]

        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")

    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform SS-WS adjustment on the data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with wind speed, standard deviation, and TI columns
        parameters : dict
            Must contain 'config_path' key

        Returns
        -------
        dict
            Dictionary containing:
            - 'adjusted_data': DataFrame with adjusted TI columns
            - 'reg_results': DataFrame with regression statistics
        """
        # Validate inputs
        self.validate_parameters(parameters)
        config_path = parameters["config_path"]
        self.validate_data(data, config_path)

        # Load column mapping
        with open(config_path, 'r') as f:
            config = json.load(f)

        col_map = config["input_data_column_mapping"]
        ref_ws_col = col_map["reference"]["wind_speed"]
        ref_sd_col = col_map["reference"]["standard_deviation"]
        ref_ti_col = col_map["reference"]["turbulence_intensity"]
        rsd_ws_col = col_map["rsd"]["primary"]["wind_speed"]
        rsd_sd_col = col_map["rsd"]["primary"]["standard_deviation"]
        rsd_ti_col = col_map["rsd"]["primary"]["turbulence_intensity"]

        # Train/test split
        split_percent = parameters.get("split_percent", 80.0)
        data_with_split = train_test_split(split_percent, data.copy())

        inputdata_train = data_with_split[data_with_split["split"]].copy()
        inputdata_test = data_with_split[~data_with_split["split"]].copy()

        # Initialize results
        results = pd.DataFrame(columns=["adjustment", "m", "c", "rsquared", "difference", "mse", "rmse"])

        # Check if we have enough data
        if len(inputdata_train) < 2 or len(inputdata_test) < 2:
            raise ValueError("Insufficient data for train/test split. Need at least 2 samples in each set.")

        # Perform linear regression on WIND SPEED (not TI)
        # Model: Ref_WS = m * RSD_WS + c
        train_data = inputdata_train[[rsd_ws_col, ref_ws_col]].dropna()

        if len(train_data) < 2:
            raise ValueError("Insufficient training data after removing NaN values.")

        model = get_regression(train_data[rsd_ws_col], train_data[ref_ws_col])
        m, c, rsquared, difference, mse, rmse = model

        # Apply wind speed adjustment to test data
        RSD_WS = inputdata_test[rsd_ws_col].copy()
        RSD_SD = inputdata_test[rsd_sd_col].copy()

        # Adjusted wind speed
        RSD_adjWS = (m * RSD_WS) + c
        inputdata_test["RSD_adjWS"] = RSD_adjWS

        # Recalculate TI using adjusted wind speed
        # TI = SD / WS_adjusted
        # Prevent division by zero
        RSD_adjWS_safe = RSD_adjWS.replace(0, np.nan)
        adjTI = RSD_SD / RSD_adjWS_safe
        inputdata_test["adjTI_RSD_TI"] = adjTI

        # Calculate representative TI
        # RepTI = TI + 1.28 * SD
        # But we need to recalculate it with adjusted values
        # For SSWS, we use: RepTI = adjTI + 1.28 * (SD / adjWS)
        inputdata_test["adjRepTI_RSD_RepTI"] = adjTI + 1.28 * (RSD_SD / RSD_adjWS_safe)

        # Post-adjustment statistics (compare adjusted TI to reference TI)
        results = post_adjustment_stats(
            inputdata_test,
            results,
            ref_ti_col,
            "adjTI_RSD_TI"
        )

        results["adjustment"] = "SSWS"

        # Combine train and test data for output
        # Add adjusted columns to train data (filled with NaN)
        inputdata_train["RSD_adjWS"] = np.nan
        inputdata_train["adjTI_RSD_TI"] = np.nan
        inputdata_train["adjRepTI_RSD_RepTI"] = np.nan

        adjusted_data = pd.concat([inputdata_train, inputdata_test]).sort_index()

        return {
            "adjusted_data": adjusted_data,
            "reg_results": results,
            "model": {"m": m, "c": c, "rsquared": rsquared}
        }
