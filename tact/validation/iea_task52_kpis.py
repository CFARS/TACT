"""
IEA Task 52 KPIs Validation Module

This module implements validation according to IEA Task 52 proposed KPIs:
- Effective TI (Ieff) bin-wise (per wind-speed bin)
- Damage Index (DI) overall (whole dataset)
- I90 (extreme) for V > 7 m/s

Reference: 20251205_IEA52_WG1_WI1_updated (Nordex slide deck)

Data conventions:
- "mast" variables in external language correspond to ref_* in TACT data
- "lidar" variables correspond to rsd_*
- lidar_sigma == rsd_sd
- mast_sigma == ref_sd
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Sequence, Tuple


def effective_sigma(sigmas, weights, m: int) -> float:
    """
    Calculate effective sigma using the IEA Task 52 formula.
    
    Formula: (∑ᵢ pᵢ σᵢᵐ)^(1/m)
    
    Parameters
    ----------
    sigmas : np.ndarray or pd.Series
        Standard deviation samples (rsd_sd or ref_sd)
    weights : np.ndarray or pd.Series
        Probabilities p_i (typically equal weights: 1/N for all valid rows)
    m : int
        Exponent parameter (typically 4, 9, or 14)
    
    Returns
    -------
    float
        Effective sigma value
    """
    # Convert to numpy arrays if needed
    # Handle pandas Series/DataFrame columns
    if isinstance(sigmas, pd.Series):
        sigmas = sigmas.values
    elif hasattr(sigmas, 'values') and not isinstance(sigmas, np.ndarray):
        sigmas = sigmas.values
    
    if isinstance(weights, pd.Series):
        weights = weights.values
    elif hasattr(weights, 'values') and not isinstance(weights, np.ndarray):
        weights = weights.values
    
    # Convert to numpy arrays, handling any type issues
    # Use pd.to_numeric to handle any string/object types gracefully
    if not isinstance(sigmas, np.ndarray):
        sigmas = pd.to_numeric(sigmas, errors='coerce').values
    else:
        # Ensure it's float64
        sigmas = sigmas.astype(np.float64)
    
    if not isinstance(weights, np.ndarray):
        weights = pd.to_numeric(weights, errors='coerce').values
    else:
        # Ensure it's float64
        weights = weights.astype(np.float64)
    
    # Filter out NaN and invalid values
    valid_mask = np.isfinite(sigmas) & np.isfinite(weights) & (sigmas > 0)
    if not np.any(valid_mask):
        return np.nan
    
    sigmas_valid = sigmas[valid_mask]
    weights_valid = weights[valid_mask]
    
    # Normalize weights to sum to 1
    weights_normalized = weights_valid / weights_valid.sum()
    
    # Calculate: (∑ᵢ pᵢ σᵢᵐ)^(1/m)
    weighted_sum = np.sum(weights_normalized * (sigmas_valid ** m))
    effective = weighted_sum ** (1.0 / m)
    
    return effective


def validate_iea_task52_kpis(
    data: pd.DataFrame,
    *,
    ws_col: str = "ref_ws",
    rsd_sd_col: str = "rsd_sd",
    ref_sd_col: str = "ref_sd",
    rsd_ti_col: str = "rsd_ti",
    ref_ti_col: str = "ref_ti",
    m_values: list[int] = [4, 9, 14],
    ws_bins: Optional[Sequence[float]] = None,
    v_i90_min: float = 7.0,
    v_ieff_range: Tuple[float, float] = (8.0, 13.0),
    quantiles: Tuple[float, float] = (0.25, 0.75),
    di_percentile: float = 0.90,
    use_test_only: bool = True
) -> Dict:
    """
    Compute IEA Task 52 proposed KPIs.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing RSD and reference data
    ws_col : str
        Column name for wind speed (default: "ref_ws")
    rsd_sd_col : str
        Column name for RSD standard deviation (default: "rsd_sd")
    ref_sd_col : str
        Column name for reference standard deviation (default: "ref_sd")
    rsd_ti_col : str
        Column name for RSD turbulence intensity (default: "rsd_ti")
    ref_ti_col : str
        Column name for reference turbulence intensity (default: "ref_ti")
    m_values : list[int]
        List of m values for Ieff and DI calculations (default: [4, 9, 14])
    ws_bins : Optional[Sequence[float]]
        Wind speed bin edges. If None, uses 1 m/s bins from 0 to 25 m/s
    v_i90_min : float
        Minimum wind speed for I90 calculation (default: 7.0 m/s)
    v_ieff_range : Tuple[float, float]
        Wind speed range for Ieff summary statistics (default: (8.0, 13.0))
    quantiles : Tuple[float, float]
        Quantiles for Ieff error summary (default: (0.25, 0.75))
    di_percentile : float
        Percentile for DI error evaluation (default: 0.90)
    use_test_only : bool
        If True, only use test data when "split" column exists (default: True)
    
    Returns
    -------
    dict
        Dictionary containing:
        - "overall": DataFrame with overall metrics (DI, I90, summary stats)
        - "by_bin": DataFrame with per-bin Ieff metrics
        - "metadata": Dictionary with configuration and metadata
    """
    # Filter to test data only if requested
    if use_test_only and "split" in data.columns:
        data = data[~data["split"]].copy()
    else:
        data = data.copy()
    
    # Create wind speed bins if not provided
    if ws_bins is None:
        # Use 1 m/s bins from 0 to 25 m/s (matching DNV style)
        ws_bins = np.arange(0, 26, 1.0)
    
    # Create bin column
    data["ws_bin"] = pd.cut(
        data[ws_col],
        bins=ws_bins,
        include_lowest=True,
        right=False
    )
    
    # Calculate bin-wise Effective TI (Ieff)
    by_bin_results = []
    
    for bin_interval in sorted(data["ws_bin"].dropna().unique(), key=lambda x: x.left):
        bin_data = data[data["ws_bin"] == bin_interval].copy()
        
        if len(bin_data) == 0:
            continue
        
        # Filter valid data
        valid_mask = (
            bin_data[rsd_sd_col].notna() & 
            bin_data[ref_sd_col].notna() &
            (bin_data[rsd_sd_col] > 0) &
            (bin_data[ref_sd_col] > 0)
        )
        bin_valid = bin_data[valid_mask]
        
        if len(bin_valid) == 0:
            continue
        
        # Get bin info
        ws_bin_left = bin_interval.left
        ws_bin_right = bin_interval.right
        ws_bin_center = (ws_bin_left + ws_bin_right) / 2.0
        n = len(bin_valid)
        
        # Calculate mean wind speed for this bin
        mean_ws = bin_valid[ws_col].mean()
        
        # Prepare weights (equal weights: 1/N)
        weights = np.ones(len(bin_valid)) / len(bin_valid)
        
        # Calculate Ieff for each m value
        bin_result = {
            "ws_bin_left": ws_bin_left,
            "ws_bin_right": ws_bin_right,
            "ws_bin_center": ws_bin_center,
            "mean_wind_speed": mean_ws,
            "N": n
        }
        
        for m in m_values:
            # Calculate Ieff for RSD and reference
            ieff_rsd = effective_sigma(
                bin_valid[rsd_sd_col].values,
                weights,
                m
            )
            ieff_ref = effective_sigma(
                bin_valid[ref_sd_col].values,
                weights,
                m
            )
            
            # Calculate ratio and relative error
            if np.isfinite(ieff_rsd) and np.isfinite(ieff_ref) and ieff_ref > 0:
                ieff_ratio = ieff_rsd / ieff_ref
                ieff_rel_error = ieff_ratio - 1.0
            else:
                ieff_ratio = np.nan
                ieff_rel_error = np.nan
            
            bin_result[f"ieff_rsd_m{m}"] = ieff_rsd
            bin_result[f"ieff_ref_m{m}"] = ieff_ref
            bin_result[f"ieff_ratio_m{m}"] = ieff_ratio
            bin_result[f"ieff_rel_error_m{m}"] = ieff_rel_error
        
        by_bin_results.append(bin_result)
    
    by_bin_df = pd.DataFrame(by_bin_results)
    
    # Calculate overall Damage Index (DI)
    # Filter valid data for overall calculation
    valid_mask_overall = (
        data[rsd_sd_col].notna() & 
        data[ref_sd_col].notna() &
        (data[rsd_sd_col] > 0) &
        (data[ref_sd_col] > 0)
    )
    data_valid = data[valid_mask_overall]
    
    overall_results = {
        "n_observations": len(data_valid),
        "mean_wind_speed": data_valid[ws_col].mean() if len(data_valid) > 0 else np.nan
    }
    
    if len(data_valid) > 0:
        # Prepare weights (equal weights: 1/N)
        weights_overall = np.ones(len(data_valid)) / len(data_valid)
        
        # Calculate DI for each m value
        for m in m_values:
            di_rsd = effective_sigma(
                data_valid[rsd_sd_col].values,
                weights_overall,
                m
            )
            di_ref = effective_sigma(
                data_valid[ref_sd_col].values,
                weights_overall,
                m
            )
            
            # Calculate ratio and relative error
            if np.isfinite(di_rsd) and np.isfinite(di_ref) and di_ref > 0:
                di_ratio = di_rsd / di_ref
                di_rel_error = di_ratio - 1.0
            else:
                di_ratio = np.nan
                di_rel_error = np.nan
            
            overall_results[f"di_rsd_m{m}"] = di_rsd
            overall_results[f"di_ref_m{m}"] = di_ref
            overall_results[f"di_ratio_m{m}"] = di_ratio
            overall_results[f"di_rel_error_m{m}"] = di_rel_error
    else:
        # No valid data
        for m in m_values:
            overall_results[f"di_rsd_m{m}"] = np.nan
            overall_results[f"di_ref_m{m}"] = np.nan
            overall_results[f"di_ratio_m{m}"] = np.nan
            overall_results[f"di_rel_error_m{m}"] = np.nan
    
    # Calculate I90 for V > 7 m/s
    i90_mask = (data[ws_col] >= v_i90_min) & data[rsd_ti_col].notna() & data[ref_ti_col].notna()
    data_i90 = data[i90_mask]
    
    if len(data_i90) > 0:
        # I90 = Imean + 1.28 * Istd
        i90_rsd = data_i90[rsd_ti_col].mean() + 1.28 * data_i90[rsd_ti_col].std()
        i90_ref = data_i90[ref_ti_col].mean() + 1.28 * data_i90[ref_ti_col].std()
        
        if np.isfinite(i90_rsd) and np.isfinite(i90_ref) and i90_ref > 0:
            i90_ratio = i90_rsd / i90_ref
            i90_rel_error = i90_ratio - 1.0
        else:
            i90_ratio = np.nan
            i90_rel_error = np.nan
        
        overall_results["i90_rsd"] = i90_rsd
        overall_results["i90_ref"] = i90_ref
        overall_results["i90_ratio"] = i90_ratio
        overall_results["i90_rel_error"] = i90_rel_error
        overall_results["N_i90"] = len(data_i90)
    else:
        overall_results["i90_rsd"] = np.nan
        overall_results["i90_ref"] = np.nan
        overall_results["i90_ratio"] = np.nan
        overall_results["i90_rel_error"] = np.nan
        overall_results["N_i90"] = 0
    
    # Calculate Ieff summary statistics within specified wind speed range
    if len(by_bin_df) > 0:
        range_mask = (
            (by_bin_df["ws_bin_center"] >= v_ieff_range[0]) &
            (by_bin_df["ws_bin_center"] <= v_ieff_range[1])
        )
        by_bin_in_range = by_bin_df[range_mask]
        
        for m in m_values:
            col_name = f"ieff_rel_error_m{m}"
            if col_name in by_bin_in_range.columns:
                errors = by_bin_in_range[col_name].dropna()
                if len(errors) > 0:
                    q25, q75 = errors.quantile(quantiles)
                    overall_results[f"ieff_rel_error_q25_m{m}"] = q25
                    overall_results[f"ieff_rel_error_q75_m{m}"] = q75
                    overall_results[f"ieff_rel_error_n_bins_m{m}"] = len(errors)
                else:
                    overall_results[f"ieff_rel_error_q25_m{m}"] = np.nan
                    overall_results[f"ieff_rel_error_q75_m{m}"] = np.nan
                    overall_results[f"ieff_rel_error_n_bins_m{m}"] = 0
            else:
                overall_results[f"ieff_rel_error_q25_m{m}"] = np.nan
                overall_results[f"ieff_rel_error_q75_m{m}"] = np.nan
                overall_results[f"ieff_rel_error_n_bins_m{m}"] = 0
    else:
        for m in m_values:
            overall_results[f"ieff_rel_error_q25_m{m}"] = np.nan
            overall_results[f"ieff_rel_error_q75_m{m}"] = np.nan
            overall_results[f"ieff_rel_error_n_bins_m{m}"] = 0
    
    overall_df = pd.DataFrame([overall_results])
    
    # Create metadata
    metadata = {
        "m_values": m_values,
        "ws_bins": ws_bins.tolist() if isinstance(ws_bins, np.ndarray) else list(ws_bins),
        "v_i90_min": v_i90_min,
        "v_ieff_range": v_ieff_range,
        "quantiles": quantiles,
        "di_percentile": di_percentile,
        "column_mapping": {
            "ws_col": ws_col,
            "rsd_sd_col": rsd_sd_col,
            "ref_sd_col": ref_sd_col,
            "rsd_ti_col": rsd_ti_col,
            "ref_ti_col": ref_ti_col
        },
        "n_total_rows": len(data),
        "n_valid_rows": len(data_valid),
        "n_i90_rows": len(data_i90) if len(data_i90) > 0 else 0
    }
    
    return {
        "overall": overall_df,
        "by_bin": by_bin_df,
        "metadata": metadata
    }
