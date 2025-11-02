"""
DNV RP-0661 Validation Module

This module implements validation according to DNV Recommended Practice RP-0661
for remote sensing device (RSD) turbulence intensity measurements.

Reference: DNV-RP-0661 - Remote Sensing Measurement Campaign for Wind Resource Assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


class DNVAcceptanceCriteria:
    """DNV RP-0661 acceptance criteria thresholds."""

    # Site Suitability criteria
    SS_MRBE_MIN_HIGH_WIND = -3.0  # For u >= 7 m/s
    SS_MRBE_MAX_HIGH_WIND = 10.0
    SS_RRMSE_MAX_HIGH_WIND = 15.0

    SS_MRBE_MIN_LOW_WIND = -6.0   # For u < 7 m/s
    SS_MRBE_MAX_LOW_WIND = 10.0
    SS_RRMSE_MAX_LOW_WIND = 30.0

    # Load Verification (LV) criteria
    LV_MRBE_MIN = -5.0
    LV_MRBE_MAX = 5.0
    LV_RRMSE_MAX = 15.0

    # Energy Production (EP) criteria
    EP_MRBE_MIN = -10.0
    EP_MRBE_MAX = 10.0


def calculate_mrbe(reference: pd.Series, measured: pd.Series) -> float:
    """
    Calculate Mean Relative Bias Error (MRBE).

    MRBE[%] = 100 * (1/N) * sum((TI_rs - TI_ann) / TI_ann)

    Parameters
    ----------
    reference : pd.Series
        Reference TI values (from cup anemometer)
    measured : pd.Series
        Measured/adjusted TI values (from RSD)

    Returns
    -------
    float
        MRBE in percentage
    """
    # Filter out invalid values (zeros, NaNs, negative)
    valid_mask = (reference > 0) & (measured > 0) & reference.notna() & measured.notna()
    ref_valid = reference[valid_mask]
    meas_valid = measured[valid_mask]

    if len(ref_valid) == 0:
        return np.nan

    relative_bias = (meas_valid - ref_valid) / ref_valid
    mrbe = 100.0 * relative_bias.mean()

    return mrbe


def calculate_rrmse(reference: pd.Series, measured: pd.Series) -> float:
    """
    Calculate Relative Root Mean Square Error (RRMSE).

    RRMSE[%] = 100 * sqrt((1/N) * sum(((TI_rs - TI_ann) / TI_ann)^2))

    Parameters
    ----------
    reference : pd.Series
        Reference TI values (from cup anemometer)
    measured : pd.Series
        Measured/adjusted TI values (from RSD)

    Returns
    -------
    float
        RRMSE in percentage
    """
    # Filter out invalid values
    valid_mask = (reference > 0) & (measured > 0) & reference.notna() & measured.notna()
    ref_valid = reference[valid_mask]
    meas_valid = measured[valid_mask]

    if len(ref_valid) == 0:
        return np.nan

    relative_error_squared = ((meas_valid - ref_valid) / ref_valid) ** 2
    rrmse = 100.0 * np.sqrt(relative_error_squared.mean())

    return rrmse


def check_acceptance_criteria(
    mrbe: float,
    rrmse: float,
    wind_speed: Optional[float] = None,
    criteria_type: str = "LV"
) -> Dict[str, bool]:
    """
    Check if metrics meet DNV RP-0661 acceptance criteria.

    Parameters
    ----------
    mrbe : float
        Mean Relative Bias Error in percentage
    rrmse : float
        Relative Root Mean Square Error in percentage
    wind_speed : float, optional
        Wind speed for site suitability criteria (affects thresholds)
    criteria_type : str
        Type of acceptance criteria: "SS" (Site Suitability), "LV" (Load Verification), or "EP" (Energy Production)

    Returns
    -------
    dict
        Dictionary with pass/fail results for each criterion
    """
    if np.isnan(mrbe) or np.isnan(rrmse):
        return {
            "pass_mrbe": False,
            "pass_rrmse": False,
            "overall_pass": False,
            "criteria_type": criteria_type
        }

    if criteria_type == "SS":
        # Site Suitability criteria depend on wind speed
        if wind_speed is not None and wind_speed >= 7.0:
            pass_mrbe = (DNVAcceptanceCriteria.SS_MRBE_MIN_HIGH_WIND <= mrbe <=
                        DNVAcceptanceCriteria.SS_MRBE_MAX_HIGH_WIND)
            pass_rrmse = rrmse <= DNVAcceptanceCriteria.SS_RRMSE_MAX_HIGH_WIND
        else:
            pass_mrbe = (DNVAcceptanceCriteria.SS_MRBE_MIN_LOW_WIND <= mrbe <=
                        DNVAcceptanceCriteria.SS_MRBE_MAX_LOW_WIND)
            pass_rrmse = rrmse <= DNVAcceptanceCriteria.SS_RRMSE_MAX_LOW_WIND

    elif criteria_type == "LV":
        # Load Verification criteria
        pass_mrbe = (DNVAcceptanceCriteria.LV_MRBE_MIN <= mrbe <=
                    DNVAcceptanceCriteria.LV_MRBE_MAX)
        pass_rrmse = rrmse <= DNVAcceptanceCriteria.LV_RRMSE_MAX

    elif criteria_type == "EP":
        # Energy Production criteria (no RRMSE limit)
        pass_mrbe = (DNVAcceptanceCriteria.EP_MRBE_MIN <= mrbe <=
                    DNVAcceptanceCriteria.EP_MRBE_MAX)
        pass_rrmse = True  # No RRMSE requirement for EP

    else:
        raise ValueError(f"Unknown criteria type: {criteria_type}. Must be 'SS', 'LV', or 'EP'")

    return {
        "pass_mrbe": pass_mrbe,
        "pass_rrmse": pass_rrmse,
        "overall_pass": pass_mrbe and pass_rrmse,
        "criteria_type": criteria_type
    }


def validate_by_wind_speed_bin(
    data: pd.DataFrame,
    reference_col: str,
    measured_col: str,
    wind_speed_col: str,
    bin_col: str = "ws_bin",
    criteria_type: str = "LV"
) -> pd.DataFrame:
    """
    Calculate DNV RP-0661 metrics for each wind speed bin.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing reference, measured, and wind speed data
    reference_col : str
        Column name for reference TI values
    measured_col : str
        Column name for measured/adjusted TI values
    wind_speed_col : str
        Column name for wind speed values
    bin_col : str
        Column name for wind speed bins
    criteria_type : str
        Type of acceptance criteria: "SS", "LV", or "EP"

    Returns
    -------
    pd.DataFrame
        DataFrame with validation results per wind speed bin
    """
    results = []

    # Get unique bins, sorted
    bins = sorted(data[bin_col].dropna().unique())

    for bin_value in bins:
        bin_data = data[data[bin_col] == bin_value].copy()

        if len(bin_data) == 0:
            continue

        # Calculate metrics for this bin
        mrbe = calculate_mrbe(bin_data[reference_col], bin_data[measured_col])
        rrmse = calculate_rrmse(bin_data[reference_col], bin_data[measured_col])

        # Get mean wind speed for this bin
        mean_ws = bin_data[wind_speed_col].mean()

        # Check acceptance criteria
        acceptance = check_acceptance_criteria(mrbe, rrmse, mean_ws, criteria_type)

        results.append({
            "wind_speed_bin": bin_value,
            "mean_wind_speed": mean_ws,
            "n_observations": len(bin_data),
            "MRBE_%": mrbe,
            "RRMSE_%": rrmse,
            "pass_MRBE": acceptance["pass_mrbe"],
            "pass_RRMSE": acceptance["pass_rrmse"],
            "overall_pass": acceptance["overall_pass"],
            "criteria_type": criteria_type
        })

    return pd.DataFrame(results)


def validate_dnv_rp0661(
    adjusted_data: pd.DataFrame,
    reference_col: str,
    adjusted_col: str,
    wind_speed_col: str,
    bin_col: str = "ws_bin",
    use_test_only: bool = True,
    criteria_type: str = "LV",
    min_ti_threshold: Optional[float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Perform complete DNV RP-0661 validation on adjusted TI data.

    Parameters
    ----------
    adjusted_data : pd.DataFrame
        DataFrame containing adjusted TI data with train/test split
    reference_col : str
        Column name for reference TI (from cup anemometer)
    adjusted_col : str
        Column name for adjusted TI (from RSD)
    wind_speed_col : str
        Column name for wind speed
    bin_col : str
        Column name for wind speed bins (default: "ws_bin")
    use_test_only : bool
        If True, only use test data (recommended). If False, use all data.
    criteria_type : str
        Type of acceptance criteria: "SS" (Site Suitability), "LV" (Load Verification), or "EP" (Energy Production)
        Default is "LV" (most commonly used)
    min_ti_threshold : float, optional
        Minimum reference TI threshold for validation. Points with reference TI below this will be excluded.
        Common values: 0.03, 0.05. If None, no filtering is applied.

    Returns
    -------
    dict
        Dictionary containing:
        - "overall": DataFrame with overall validation metrics
        - "by_bin": DataFrame with per-bin validation metrics
    """
    # Filter to test data only if requested
    if use_test_only and "split" in adjusted_data.columns:
        data = adjusted_data[~adjusted_data["split"]].copy()
    else:
        data = adjusted_data.copy()

    # Filter by minimum TI threshold if specified
    if min_ti_threshold is not None:
        n_before = len(data)
        data = data[data[reference_col] >= min_ti_threshold].copy()
        n_after = len(data)
        n_filtered = n_before - n_after
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} observations with reference TI < {min_ti_threshold} ({n_filtered/n_before*100:.1f}% of data)")

    # Calculate overall metrics
    overall_mrbe = calculate_mrbe(data[reference_col], data[adjusted_col])
    overall_rrmse = calculate_rrmse(data[reference_col], data[adjusted_col])

    # Check overall acceptance
    mean_ws = data[wind_speed_col].mean()
    overall_acceptance = check_acceptance_criteria(overall_mrbe, overall_rrmse, mean_ws, criteria_type)

    overall_results = pd.DataFrame([{
        "metric": "Overall",
        "n_observations": len(data),
        "mean_wind_speed": mean_ws,
        "MRBE_%": overall_mrbe,
        "RRMSE_%": overall_rrmse,
        "pass_MRBE": overall_acceptance["pass_mrbe"],
        "pass_RRMSE": overall_acceptance["pass_rrmse"],
        "overall_pass": overall_acceptance["overall_pass"],
        "criteria_type": criteria_type
    }])

    # Calculate per-bin metrics
    bin_results = validate_by_wind_speed_bin(
        data,
        reference_col,
        adjusted_col,
        wind_speed_col,
        bin_col=bin_col,
        criteria_type=criteria_type
    )

    return {
        "overall": overall_results,
        "by_bin": bin_results
    }
