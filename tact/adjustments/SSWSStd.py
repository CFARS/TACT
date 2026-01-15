"""
Site-Specific Wind Speed + Standard Deviation (SS-WS-Std) Adjustment Method

This method adjusts both RSD wind speed AND standard deviation using separate
linear regressions, then recalculates turbulence intensity.

Steps:
1. Train regression: Ref_WS = m1 * RSD_WS + c1
2. Train regression: Ref_SD = m2 * RSD_SD + c2
3. Apply both adjustments
4. Recalculate: TI = adjusted_SD / adjusted_WS

This is more sophisticated than SSWS because it corrects both components of TI.
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


@AdjustmentRegistry.register("sswsstd")
class SSWSStd(AdjustmentMethod):
    """
    Site-Specific Wind Speed + Standard Deviation adjustment method.

    This method:
    1. Performs train/test split (80/20)
    2. Trains TWO linear regressions:
       - Wind speed: Ref_WS = m1 * RSD_WS + c1
       - Standard deviation: Ref_SD = m2 * RSD_SD + c2
    3. Adjusts both RSD wind speed and SD in test data
    4. Recalculates TI = adjusted_SD / adjusted_WS
    5. Calculates representative TI

    Parameters
    ----------
    config_path : str
        Path to configuration JSON file with column mappings
    """

    def __init__(self):
        super().__init__()
        self.name = "sswsstd"
        self.description = "Site-Specific Wind Speed + Std Deviation adjustment"

    def required_model_parameters(self) -> Dict[str, type]:
        """Return required parameters for SS-WS-Std adjustment."""
        return {
            "config_path": str,
            "split": bool
        }

    def required_data_columns(self) -> list:
        """Return required data columns for SS-WS-Std adjustment."""
        return [
            "reference.wind_speed",
            "reference.standard_deviation",
            "reference.turbulence_intensity",
            "rsd.primary.wind_speed",
            "rsd.primary.standard_deviation",
            "rsd.primary.turbulence_intensity"
        ]

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required parameters for SS-WS-Std adjustment."""
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
        Perform SS-WS-Std adjustment on the data.

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

        # Perform linear regression on WIND SPEED
        # Model 1: Ref_WS = m1 * RSD_WS + c1
        train_data_ws = inputdata_train[[rsd_ws_col, ref_ws_col]].dropna()

        if len(train_data_ws) < 2:
            raise ValueError("Insufficient training data for wind speed regression after removing NaN values.")

        model_ws = get_regression(train_data_ws[rsd_ws_col], train_data_ws[ref_ws_col])
        m_ws, c_ws, rsquared_ws, difference_ws, mse_ws, rmse_ws = model_ws

        # Perform linear regression on STANDARD DEVIATION
        # Model 2: Ref_SD = m2 * RSD_SD + c2
        train_data_sd = inputdata_train[[rsd_sd_col, ref_sd_col]].dropna()

        if len(train_data_sd) < 2:
            raise ValueError("Insufficient training data for SD regression after removing NaN values.")

        model_sd = get_regression(train_data_sd[rsd_sd_col], train_data_sd[ref_sd_col])
        m_sd, c_sd, rsquared_sd, difference_sd, mse_sd, rmse_sd = model_sd

        # Apply BOTH adjustments to test data
        RSD_WS = inputdata_test[rsd_ws_col].copy()
        RSD_SD = inputdata_test[rsd_sd_col].copy()

        # Adjusted wind speed
        RSD_adjWS = (m_ws * RSD_WS) + c_ws
        inputdata_test["RSD_adjWS"] = RSD_adjWS

        # Adjusted standard deviation
        RSD_adjSD = (m_sd * RSD_SD) + c_sd
        inputdata_test["RSD_adjSD"] = RSD_adjSD

        # Recalculate TI using BOTH adjusted values
        # TI = adjusted_SD / adjusted_WS
        # Prevent division by zero
        RSD_adjWS_safe = RSD_adjWS.replace(0, np.nan)
        adjTI = RSD_adjSD / RSD_adjWS_safe
        inputdata_test["adjTI_RSD_TI"] = adjTI

        # Calculate representative TI
        # RepTI = TI + 1.28 * (SD / WS)
        # Using adjusted values: RepTI = adjTI + 1.28 * (adjSD / adjWS)
        inputdata_test["adjRepTI_RSD_RepTI"] = adjTI + 1.28 * (RSD_adjSD / RSD_adjWS_safe)

        # Post-adjustment statistics (compare adjusted TI to reference TI)
        results = post_adjustment_stats(
            inputdata_test,
            results,
            ref_ti_col,
            "adjTI_RSD_TI"
        )

        results["adjustment"] = "SS-WS-Std"

        # Combine train and test data for output
        # Add adjusted columns to train data (filled with NaN)
        inputdata_train["RSD_adjWS"] = np.nan
        inputdata_train["RSD_adjSD"] = np.nan
        inputdata_train["adjTI_RSD_TI"] = np.nan
        inputdata_train["adjRepTI_RSD_RepTI"] = np.nan

        adjusted_data = pd.concat([inputdata_train, inputdata_test]).sort_index()

        return {
            "adjusted_data": adjusted_data,
            "reg_results": results,
            "models": {
                "wind_speed": {"m": m_ws, "c": c_ws, "rsquared": rsquared_ws},
                "std_dev": {"m": m_sd, "c": c_sd, "rsquared": rsquared_sd}
            }
        }
