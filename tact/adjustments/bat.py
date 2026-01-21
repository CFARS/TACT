"""
BAT (Bias Adjustment Technique) Adjustment Method

This method uses Singular Spectrum Analysis (SSA) to decompose RSD wind speed
and standard deviation time series, then applies learned gains and offsets
to correct systematic biases in the measurements.

The method:
1. Performs SSA decomposition on RSD WS and SD series (window length L=100)
2. Applies gains/offsets from trained coefficients for components 1-35
3. Reconstructs corrected WS and SD series
4. Computes corrected TI = corrected_SD / corrected_WS
"""

import pandas as pd
import numpy as np
import pickle
import importlib.resources as ir
from typing import Dict, Any, Tuple
import json
import warnings

from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry
from tact.utils.ssa import ssa_decompose, reconstruct_component


def load_bat_coefficients(coeff_path: str = None) -> Any:
    """
    Load BAT coefficients from packaged asset or custom path.

    Parameters
    ----------
    coeff_path : str, optional
        Custom path to coefficient file. If None, uses packaged Generic_BAT35_wk6.pkl

    Returns
    -------
    Loaded coefficient object (typically has WSGain, WSOffset, STDGain, STDOffset attributes)
    """
    if coeff_path is None:
        # Load from packaged asset
        try:
            with ir.files("tact.assets.bat").joinpath("Generic_BAT35_wk6.pkl").open("rb") as f:
                coeff = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "BAT coefficient file not found. Please ensure Generic_BAT35_wk6.pkl "
                "is placed in tact/assets/bat/ directory."
            )
    else:
        # Load from custom path
        with open(coeff_path, "rb") as f:
            coeff = pickle.load(f)

    return coeff


def apply_bat(ws: pd.Series, sd: pd.Series, coeff: Any, L: int = 100, 
              n_calib: int = 35) -> Tuple[pd.Series, pd.Series]:
    """
    Apply BAT correction to wind speed and standard deviation series.

    Parameters
    ----------
    ws : pd.Series
        RSD wind speed time series
    sd : pd.Series
        RSD standard deviation time series
    coeff : Any
        Coefficient object with WSGain, WSOffset, STDGain, STDOffset attributes
        (arrays or lists of length >= n_calib+1, indexed 0..n_calib)
    L : int, default 100
        SSA window length
    n_calib : int, default 35
        Number of calibrated components (components 1..n_calib get gains/offsets)

    Returns
    -------
    corr_ws : pd.Series
        Corrected wind speed series (same index as input)
    corr_sd : pd.Series
        Corrected standard deviation series (same index as input)

    Raises
    ------
    ValueError
        If series have different indices or lengths, or if L is invalid
    """
    # Validate inputs
    if len(ws) != len(sd):
        raise ValueError(f"WS and SD series must have same length: {len(ws)} vs {len(sd)}")

    if not ws.index.equals(sd.index):
        # Align indices
        common_idx = ws.index.intersection(sd.index)
        ws = ws.loc[common_idx]
        sd = sd.loc[common_idx]
        if len(ws) == 0:
            raise ValueError("WS and SD series have no overlapping indices")

    # Handle NaNs: drop rows where either series has NaN
    valid_mask = ws.notna() & sd.notna()
    if valid_mask.sum() == 0:
        # All NaNs, return NaN series
        return pd.Series(np.nan, index=ws.index), pd.Series(np.nan, index=sd.index)

    ws_valid = ws[valid_mask].copy()
    sd_valid = sd[valid_mask].copy()

    n_valid = len(ws_valid)

    # Auto-adjust L if series is too short
    L_actual = L
    if n_valid < 2 * L:
        L_actual = max(2, n_valid // 2)
        if L_actual < L:
            warnings.warn(
                f"Series length ({n_valid}) < 2*L ({2*L}). "
                f"Reducing L from {L} to {L_actual}.",
                UserWarning
            )

    # Extract coefficients and normalize to numpy arrays
    WSGain = np.asarray(coeff.WSGain)
    WSOffset = np.asarray(coeff.WSOffset)
    STDGain = np.asarray(coeff.STDGain)
    STDOffset = np.asarray(coeff.STDOffset)

    # Validate coefficient lengths
    if len(WSGain) < n_calib + 1 or len(WSOffset) < n_calib + 1:
        raise ValueError(
            f"Coefficients must have length >= {n_calib + 1}, "
            f"got WSGain={len(WSGain)}, WSOffset={len(WSOffset)}"
        )
    if len(STDGain) < n_calib + 1 or len(STDOffset) < n_calib + 1:
        raise ValueError(
            f"Coefficients must have length >= {n_calib + 1}, "
            f"got STDGain={len(STDGain)}, STDOffset={len(STDOffset)}"
        )

    # Perform SSA decomposition
    try:
        U_ws, S_ws, Vt_ws = ssa_decompose(ws_valid, L_actual)
        U_sd, S_sd, Vt_sd = ssa_decompose(sd_valid, L_actual)
    except ValueError as e:
        raise ValueError(f"SSA decomposition failed: {e}")

    # Initialize with component 0
    tempWS = reconstruct_component(ws_valid, U_ws, S_ws, Vt_ws, 0)
    tempSTD = reconstruct_component(sd_valid, U_sd, S_sd, Vt_sd, 0)

    # Get number of available components (rank of decomposition)
    d_ws = len(S_ws)
    d_sd = len(S_sd)
    d_max = min(d_ws, d_sd, L_actual)

    # Apply corrections for components 1..n_calib
    for j in range(1, min(n_calib + 1, d_max + 1)):
        recon_ws_j = reconstruct_component(ws_valid, U_ws, S_ws, Vt_ws, j)
        recon_sd_j = reconstruct_component(sd_valid, U_sd, S_sd, Vt_sd, j)

        # Apply gains and offsets (0-based indexing: WSGain[j] for component j)
        tempWS += recon_ws_j * WSGain[j] + WSOffset[j]
        tempSTD += recon_sd_j * STDGain[j] + STDOffset[j]

    # Add remaining components (>n_calib) without correction
    for j in range(n_calib + 1, d_max + 1):
        recon_ws_j = reconstruct_component(ws_valid, U_ws, S_ws, Vt_ws, j)
        recon_sd_j = reconstruct_component(sd_valid, U_sd, S_sd, Vt_sd, j)
        tempWS += recon_ws_j
        tempSTD += recon_sd_j

    # Final corrected values
    CorrWS = tempWS.copy()
    CorrSD = tempSTD.copy()

    # Guard against non-positive corrected WS
    CorrWS[CorrWS <= 0] = np.nan

    # Map back to original index (fill NaNs where data was invalid)
    corr_ws_full = pd.Series(np.nan, index=ws.index)
    corr_sd_full = pd.Series(np.nan, index=sd.index)
    corr_ws_full.loc[valid_mask] = CorrWS
    corr_sd_full.loc[valid_mask] = CorrSD

    return corr_ws_full, corr_sd_full


@AdjustmentRegistry.register("bat")
class BATAdjustment(AdjustmentMethod):
    """
    BAT (Bias Adjustment Technique) adjustment method.

    This method corrects RSD wind speed and standard deviation using SSA decomposition
    and learned coefficient gains/offsets. It does not require reference data for
    training, as it uses pre-trained coefficients.

    Parameters
    ----------
    config_path : str
        Path to configuration JSON file with column mappings
    L : int, optional (default 100)
        SSA window length. Auto-reduced if series is too short.
    n_calib : int, optional (default 35)
        Number of calibrated components (components 1..n_calib get gains/offsets)
    coeff_path : str, optional
        Custom path to coefficient file. If None, uses packaged Generic_BAT35_wk6.pkl
    """

    def __init__(self):
        super().__init__()
        self.name = "bat"
        self.description = "BAT (Bias Adjustment Technique) - SSA-based correction with learned coefficients"

    def required_model_parameters(self) -> Dict[str, type]:
        """Return required parameters for BAT adjustment."""
        return {
            "config_path": str,
            "L": int,  # Optional, but declared here for type checking
            "n_calib": int,  # Optional
            "coeff_path": str  # Optional
        }

    def required_data_columns(self) -> list:
        """Return required data columns for BAT adjustment."""
        return [
            "rsd.primary.wind_speed",
            "rsd.primary.standard_deviation"
        ]

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate required parameters for BAT adjustment."""
        if "config_path" not in parameters:
            raise ValueError("Missing required parameter: config_path")

        # Validate optional parameters if provided
        if "L" in parameters:
            L = parameters["L"]
            if not isinstance(L, int) or L < 2:
                raise ValueError(f"Parameter L must be an integer >= 2, got {L}")

        if "n_calib" in parameters:
            n_calib = parameters["n_calib"]
            if not isinstance(n_calib, int) or n_calib < 1:
                raise ValueError(f"Parameter n_calib must be an integer >= 1, got {n_calib}")

    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform BAT adjustment on the data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with RSD wind speed and standard deviation columns
        parameters : dict
            Must contain 'config_path' key. Optional: 'L', 'n_calib', 'coeff_path'

        Returns
        -------
        dict
            Dictionary containing:
            - 'adjusted_data': DataFrame with adjusted columns (RSD_adjWS, RSD_adjSD, adjTI_RSD_TI)
            - 'metrics': Dictionary with adjustment metadata
        """
        # Validate inputs
        self.validate_parameters(parameters)
        config_path = parameters["config_path"]
        self.validate_data(data, config_path)

        # Load column mapping
        with open(config_path, 'r') as f:
            config = json.load(f)

        col_map = config["input_data_column_mapping"]
        rsd_ws_col = col_map["rsd"]["primary"]["wind_speed"]
        rsd_sd_col = col_map["rsd"]["primary"]["standard_deviation"]

        # Extract series
        rsd_ws = data[rsd_ws_col].copy()
        rsd_sd = data[rsd_sd_col].copy()

        # Get parameters
        L = parameters.get("L", 100)
        n_calib = parameters.get("n_calib", 35)
        coeff_path = parameters.get("coeff_path", None)

        # Load coefficients
        try:
            coeff = load_bat_coefficients(coeff_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Failed to load BAT coefficients: {e}. "
                "Please ensure Generic_BAT35_wk6.pkl is in tact/assets/bat/ directory."
            )

        # Apply BAT correction
        try:
            corr_ws, corr_sd = apply_bat(rsd_ws, rsd_sd, coeff, L=L, n_calib=n_calib)
        except Exception as e:
            raise ValueError(f"BAT adjustment failed: {e}")

        # Create adjusted data copy
        adjusted_data = data.copy()
        adjusted_data["RSD_adjWS"] = corr_ws
        adjusted_data["RSD_adjSD"] = corr_sd

        # Compute corrected TI
        # Prevent division by zero
        corr_ws_safe = corr_ws.replace(0, np.nan)
        adj_ti = corr_sd / corr_ws_safe
        adjusted_data["adjTI_RSD_TI"] = adj_ti

        # Calculate representative TI (if needed, following other methods)
        adjusted_data["adjRepTI_RSD_RepTI"] = adj_ti + 1.28 * (corr_sd / corr_ws_safe)

        # Compute metrics
        n_total = len(adjusted_data)
        n_valid = corr_ws.notna().sum()
        pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0.0

        # Summary statistics (only on valid data)
        valid_mask = corr_ws.notna() & corr_sd.notna()
        if valid_mask.sum() > 0:
            mean_ws_before = rsd_ws[valid_mask].mean()
            mean_ws_after = corr_ws[valid_mask].mean()
            mean_sd_before = rsd_sd[valid_mask].mean()
            mean_sd_after = corr_sd[valid_mask].mean()
        else:
            mean_ws_before = mean_ws_after = mean_sd_before = mean_sd_after = np.nan

        metrics = {
            "method": "BAT",
            "L": L,
            "n_calib": n_calib,
            "coeff_file": "Generic_BAT35_wk6.pkl" if coeff_path is None else coeff_path,
            "n_total": n_total,
            "n_valid": n_valid,
            "pct_valid": round(pct_valid, 2),
            "mean_ws_before": round(mean_ws_before, 4) if not np.isnan(mean_ws_before) else None,
            "mean_ws_after": round(mean_ws_after, 4) if not np.isnan(mean_ws_after) else None,
            "mean_sd_before": round(mean_sd_before, 4) if not np.isnan(mean_sd_before) else None,
            "mean_sd_after": round(mean_sd_after, 4) if not np.isnan(mean_sd_after) else None,
        }

        return {
            "adjusted_data": adjusted_data,
            "metrics": metrics
        }
