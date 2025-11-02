"""
Visualization functions for DNV RP-0661 validation results.

Creates plots similar to DNV recommended practice documentation,
showing MRBE and RRMSE metrics by wind speed bin with acceptance boundaries.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import os


def plot_mrbe_by_bin(
    validation_by_bin: pd.DataFrame,
    criteria_type: str = "LV",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot MRBE (Mean Relative Bias Error) by wind speed bin with DNV acceptance boundaries.

    Parameters
    ----------
    validation_by_bin : pd.DataFrame
        Validation results by wind speed bin from validate_dnv_rp0661
    criteria_type : str
        Type of criteria: "SS", "LV", or "EP"
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    wind_speed_bins = validation_by_bin['wind_speed_bin']
    mrbe = validation_by_bin['MRBE_%']
    n_obs = validation_by_bin['n_observations']

    # Plot MRBE values
    scatter = ax.scatter(wind_speed_bins, mrbe, c='blue', s=100, alpha=0.7,
                        label='MRBE', zorder=5, edgecolors='darkblue', linewidths=1.5)

    # Add observation counts as text
    for i, (ws, m, n) in enumerate(zip(wind_speed_bins, mrbe, n_obs)):
        ax.text(ws, m + 5, str(int(n)), ha='center', va='bottom',
               fontsize=8, color='blue', fontweight='bold')

    # Add acceptance criteria boundaries
    if criteria_type == "LV":
        ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='LV Upper Limit (+5%)')
        ax.axhline(y=-5, color='red', linestyle='--', linewidth=2, label='LV Lower Limit (-5%)')
        ax.fill_between(wind_speed_bins.unique(), -5, 5, alpha=0.2, color='green', label='LV Acceptance Zone')
    elif criteria_type == "EP":
        ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='EP Upper Limit (+10%)')
        ax.axhline(y=-10, color='orange', linestyle='--', linewidth=2, label='EP Lower Limit (-10%)')
        ax.fill_between(wind_speed_bins.unique(), -10, 10, alpha=0.2, color='green', label='EP Acceptance Zone')
    elif criteria_type == "SS":
        # SS has different limits for high/low wind speeds
        ax.axhline(y=10, color='purple', linestyle='--', linewidth=2, label='SS Upper Limit (+10%)')
        ax.axhline(y=-3, color='purple', linestyle='--', linewidth=2, label='SS Lower Limit (u≥7: -3%)')
        ax.axhline(y=-6, color='purple', linestyle=':', linewidth=2, label='SS Lower Limit (u<7: -6%)')

    # Formatting
    ax.set_xlabel('Wind Speed Bin (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MRBE (%)', fontsize=12, fontweight='bold')
    ax.set_title(title or f'DNV {criteria_type} - MRBE by Wind Speed Bin', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved MRBE plot to: {save_path}")

    return fig


def plot_rrmse_by_bin(
    validation_by_bin: pd.DataFrame,
    criteria_type: str = "LV",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot RRMSE (Relative Root Mean Square Error) by wind speed bin with DNV acceptance boundaries.

    Parameters
    ----------
    validation_by_bin : pd.DataFrame
        Validation results by wind speed bin from validate_dnv_rp0661
    criteria_type : str
        Type of criteria: "SS", "LV", or "EP"
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    wind_speed_bins = validation_by_bin['wind_speed_bin']
    rrmse = validation_by_bin['RRMSE_%']
    n_obs = validation_by_bin['n_observations']

    # Plot RRMSE values
    scatter = ax.scatter(wind_speed_bins, rrmse, c='darkgreen', s=100, alpha=0.7,
                        label='RRMSE', zorder=5, edgecolors='black', linewidths=1.5)

    # Add observation counts as text
    for i, (ws, r, n) in enumerate(zip(wind_speed_bins, rrmse, n_obs)):
        ax.text(ws, r + 3, str(int(n)), ha='center', va='bottom',
               fontsize=8, color='darkgreen', fontweight='bold')

    # Add acceptance criteria boundaries
    if criteria_type == "LV":
        ax.axhline(y=15, color='red', linestyle='--', linewidth=2, label='LV Limit (15%)')
        ax.fill_between(wind_speed_bins.unique(), 0, 15, alpha=0.2, color='green', label='LV Acceptance Zone')
    elif criteria_type == "SS":
        # SS has different limits for high/low wind speeds
        ax.axhline(y=15, color='purple', linestyle='--', linewidth=2, label='SS Limit (u≥7: 15%)')
        ax.axhline(y=30, color='purple', linestyle=':', linewidth=2, label='SS Limit (u<7: 30%)')
        ax.axvline(x=7, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # EP has no RRMSE limit
    if criteria_type == "EP":
        ax.text(0.5, 0.95, 'No RRMSE limit for EP criteria', transform=ax.transAxes,
               ha='center', va='top', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Formatting
    ax.set_xlabel('Wind Speed Bin (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RRMSE (%)', fontsize=12, fontweight='bold')
    ax.set_title(title or f'DNV {criteria_type} - RRMSE by Wind Speed Bin', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved RRMSE plot to: {save_path}")

    return fig


def plot_ti_scatter(
    adjusted_data: pd.DataFrame,
    reference_col: str,
    adjusted_col: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Create scatter plot of adjusted TI vs reference TI with 1:1 line.

    Parameters
    ----------
    adjusted_data : pd.DataFrame
        DataFrame containing adjusted and reference TI
    reference_col : str
        Column name for reference TI
    adjusted_col : str
        Column name for adjusted TI
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter test data only if split column exists
    if 'split' in adjusted_data.columns:
        plot_data = adjusted_data[~adjusted_data['split']].copy()
    else:
        plot_data = adjusted_data.copy()

    # Remove NaN values
    valid_mask = plot_data[reference_col].notna() & plot_data[adjusted_col].notna()
    plot_data = plot_data[valid_mask]

    # Scatter plot
    ax.scatter(plot_data[reference_col], plot_data[adjusted_col],
              alpha=0.3, s=20, c='blue', edgecolors='none')

    # 1:1 line
    max_val = max(plot_data[reference_col].max(), plot_data[adjusted_col].max())
    min_val = min(plot_data[reference_col].min(), plot_data[adjusted_col].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

    # ±20% error bands
    x_range = np.linspace(min_val, max_val, 100)
    ax.plot(x_range, x_range * 1.2, 'g:', linewidth=1, alpha=0.5, label='±20% Error')
    ax.plot(x_range, x_range * 0.8, 'g:', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel('Reference TI', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted TI', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Adjusted TI vs Reference TI', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')

    # Add statistics text box
    bias = (plot_data[adjusted_col].mean() - plot_data[reference_col].mean()) / plot_data[reference_col].mean() * 100
    stats_text = f'N = {len(plot_data)}\nBias = {bias:.2f}%'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to: {save_path}")

    return fig


def plot_ti_comparison(
    adjusted_data: pd.DataFrame,
    reference_col: str,
    unadjusted_col: str,
    adjusted_col: str,
    wind_speed_col: str,
    bin_col: str = "bins",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot comparison of reference, unadjusted, and adjusted TI by wind speed bin.

    Parameters
    ----------
    adjusted_data : pd.DataFrame
        DataFrame containing all TI data
    reference_col : str
        Column name for reference TI
    unadjusted_col : str
        Column name for unadjusted RSD TI
    adjusted_col : str
        Column name for adjusted TI
    wind_speed_col : str
        Column name for wind speed
    bin_col : str
        Column name for wind speed bins
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter test data only
    if 'split' in adjusted_data.columns:
        plot_data = adjusted_data[~adjusted_data['split']].copy()
    else:
        plot_data = adjusted_data.copy()

    # Calculate means by bin
    bins = sorted(plot_data[bin_col].dropna().unique())
    ref_means = []
    unadj_means = []
    adj_means = []
    counts = []

    for bin_val in bins:
        bin_data = plot_data[plot_data[bin_col] == bin_val]
        ref_means.append(bin_data[reference_col].mean())
        unadj_means.append(bin_data[unadjusted_col].mean())
        adj_means.append(bin_data[adjusted_col].mean())
        counts.append(len(bin_data))

    # Plot
    width = 0.3
    x = np.array(bins)

    ax.plot(x, ref_means, 'o-', color='black', linewidth=2, markersize=8, label='Reference TI', zorder=5)
    ax.plot(x, unadj_means, 's--', color='red', linewidth=2, markersize=8, label='RSD TI (Unadjusted)', alpha=0.7)
    ax.plot(x, adj_means, '^-', color='blue', linewidth=2, markersize=8, label='Adjusted TI', zorder=4)

    # Add observation counts
    for i, (xi, count) in enumerate(zip(x, counts)):
        ax.text(xi, max(ref_means[i], unadj_means[i], adj_means[i]) + 0.01,
               str(count), ha='center', va='bottom', fontsize=8, color='gray')

    # Formatting
    ax.set_xlabel('Wind Speed Bin (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TI', fontsize=12, fontweight='bold')
    ax.set_title(title or 'TI Comparison by Wind Speed Bin', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(left=min(bins)-0.5, right=max(bins)+0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")

    return fig


def plot_dnv_validation(
    validation_results: Dict[str, pd.DataFrame],
    adjusted_data: pd.DataFrame,
    reference_col: str,
    unadjusted_col: str,
    adjusted_col: str,
    wind_speed_col: str,
    bin_col: str = "bins",
    criteria_type: str = "LV",
    method_name: str = "Adjustment",
    output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Create complete set of DNV validation plots.

    Parameters
    ----------
    validation_results : dict
        Validation results from validate_dnv_rp0661
    adjusted_data : pd.DataFrame
        DataFrame containing adjusted data
    reference_col : str
        Column name for reference TI
    unadjusted_col : str
        Column name for unadjusted RSD TI
    adjusted_col : str
        Column name for adjusted TI
    wind_speed_col : str
        Column name for wind speed
    bin_col : str
        Column name for wind speed bins
    criteria_type : str
        Type of criteria: "SS", "LV", or "EP"
    method_name : str
        Name of adjustment method for titles
    output_dir : str, optional
        Directory to save plots

    Returns
    -------
    dict
        Dictionary of figure objects keyed by plot name
    """
    figures = {}

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. MRBE by bin
    save_path = f"{output_dir}/{method_name}_mrbe_by_bin.png" if output_dir else None
    figures['mrbe'] = plot_mrbe_by_bin(
        validation_results['by_bin'],
        criteria_type=criteria_type,
        title=f'{method_name} - MRBE by Wind Speed Bin',
        save_path=save_path
    )

    # 2. RRMSE by bin
    save_path = f"{output_dir}/{method_name}_rrmse_by_bin.png" if output_dir else None
    figures['rrmse'] = plot_rrmse_by_bin(
        validation_results['by_bin'],
        criteria_type=criteria_type,
        title=f'{method_name} - RRMSE by Wind Speed Bin',
        save_path=save_path
    )

    # 3. TI scatter
    save_path = f"{output_dir}/{method_name}_ti_scatter.png" if output_dir else None
    figures['scatter'] = plot_ti_scatter(
        adjusted_data,
        reference_col,
        adjusted_col,
        title=f'{method_name} - Adjusted vs Reference TI',
        save_path=save_path
    )

    # 4. TI comparison by bin
    save_path = f"{output_dir}/{method_name}_ti_comparison.png" if output_dir else None
    figures['comparison'] = plot_ti_comparison(
        adjusted_data,
        reference_col,
        unadjusted_col,
        adjusted_col,
        wind_speed_col,
        bin_col=bin_col,
        title=f'{method_name} - TI Comparison by Wind Speed',
        save_path=save_path
    )

    if output_dir:
        print(f"\n✅ All plots saved to: {output_dir}")

    return figures
