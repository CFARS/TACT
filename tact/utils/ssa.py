"""
Singular Spectrum Analysis (SSA) utility for time series decomposition.

This module provides SSA decomposition and reconstruction functions for use
in adjustment methods like BAT.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def ssa_decompose(series: pd.Series, L: int, save_mem: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SSA decomposition on a 1D time series.

    Parameters
    ----------
    series : pd.Series
        Input time series (1D)
    L : int
        Window length (embedding dimension). Must satisfy 2 <= L <= N/2 where N is series length.
    save_mem : bool, default True
        If True, don't store all elementary matrices (memory efficient for large series)

    Returns
    -------
    U : np.ndarray
        Left singular vectors (shape: (L, d)) where d is the rank
    S : np.ndarray
        Singular values (shape: (d,))
    Vt : np.ndarray
        Right singular vectors (shape: (d, K)) where K = N - L + 1

    Raises
    ------
    ValueError
        If L is not in valid range [2, N/2] or if series is too short
    """
    # Convert to numpy array and remove NaNs for SSA computation
    # We'll handle NaNs in the calling function
    values = series.values
    n = len(values)

    if n < 2:
        raise ValueError(f"Series must have at least 2 points, got {n}")

    # Validate window length
    if L < 2:
        raise ValueError(f"Window length L must be >= 2, got {L}")
    if L > n / 2:
        raise ValueError(f"Window length L ({L}) must be <= N/2 ({n/2:.0f})")

    # Embedding: create trajectory matrix X
    K = n - L + 1
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = values[i:i+L]

    # SVD decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # S is 1D array of singular values
    # U shape: (L, d) where d = min(L, K)
    # Vt shape: (d, K)

    return U, S, Vt


def reconstruct_component(series: pd.Series, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, 
                         component_idx: int) -> pd.Series:
    """
    Reconstruct a single SSA component.

    Parameters
    ----------
    series : pd.Series
        Original time series (used for index and length)
    U : np.ndarray
        Left singular vectors from SSA decomposition
    S : np.ndarray
        Singular values from SSA decomposition
    Vt : np.ndarray
        Right singular vectors from SSA decomposition
    component_idx : int
        Index of component to reconstruct (0-based)

    Returns
    -------
    pd.Series
        Reconstructed component time series with same index as input series
    """
    n = len(series)
    L = U.shape[0]
    K = n - L + 1

    if component_idx >= len(S):
        # Component doesn't exist, return zeros
        return pd.Series(0.0, index=series.index)

    # Reconstruct elementary matrix for this component
    # X_i = S[i] * U[:, i:i+1] @ Vt[i:i+1, :]
    u_i = U[:, component_idx:component_idx+1]  # (L, 1)
    v_i = Vt[component_idx:component_idx+1, :]  # (1, K)
    X_i = S[component_idx] * (u_i @ v_i)  # (L, K)

    # Diagonal averaging (Hankelization) to convert back to time series
    reconstructed = np.zeros(n)
    for t in range(n):
        # Sum over anti-diagonal
        count = 0
        total = 0.0
        for i in range(L):
            j = t - i
            if 0 <= j < K:
                total += X_i[i, j]
                count += 1
        if count > 0:
            reconstructed[t] = total / count
        else:
            reconstructed[t] = 0.0

    return pd.Series(reconstructed, index=series.index)
