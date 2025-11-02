"""
Compare all TACT adjustment methods with DNV RP-0661 validation.

This script runs all implemented methods on the test dataset and compares
their performance against DNV acceptance criteria.
"""

from tact import TACT
from tact.adjustments.baseline import BaselineResults
from tact.adjustments.SSSF import SSSF
from tact.adjustments.SSWS import SSWS
from tact.adjustments.SSWSStd import SSWSStd
from tact.utils.setup_processors import setup_processors
from tact.utils.load_data import load_data
from tact.validation import validate_dnv_rp0661
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_method_comparison(comparison_df: pd.DataFrame, output_dir: str):
    """Create comparison plots for all methods including baseline.

    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid results
    df = comparison_df[comparison_df["MRBE_%"].notna()].copy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DNV RP-0661 Validation Results - Method Comparison",
                 fontsize=14, fontweight='bold', y=0.98)

    # Define colors - baseline in different color
    colors = ['#d62728' if m == 'BASELINE' else '#5899DA' if m == 'SS-SF'
              else '#c088b8' for m in df['Method']]

    # Plot 1: MRBE by Method
    x_pos = np.arange(len(df))
    bars1 = ax1.bar(x_pos, df['MRBE_%'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, df['MRBE_%'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # DNV acceptance zone for MRBE (±5%)
    ax1.axhline(y=5, color='green', linestyle='--', linewidth=2, label='LV Upper Limit (+5%)', alpha=0.7)
    ax1.axhline(y=-5, color='green', linestyle='--', linewidth=2, label='LV Lower Limit (-5%)', alpha=0.7)
    ax1.fill_between(x_pos, -5, 5, color='green', alpha=0.1, label='LV Acceptance Zone')

    ax1.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MRBE (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Relative Bias Error by Method', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['Method'], rotation=0, ha='center')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.set_axisbelow(True)

    # Plot 2: RRMSE by Method
    bars2 = ax2.bar(x_pos, df['RRMSE_%'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, df['RRMSE_%'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # DNV acceptance limit for RRMSE (15%)
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='LV Limit (15%)', alpha=0.7)
    ax2.fill_between(x_pos, 0, 15, color='green', alpha=0.1, label='LV Acceptance Zone')

    ax2.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('RRMSE (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Relative Root Mean Square Error by Method', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df['Method'], rotation=0, ha='center')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    output_file = plots_dir / "method_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Comparison plot saved to: {output_file}")


def compare_all_methods():
    """Run all methods and compare DNV validation results."""

    # Configuration
    config = {
        "data_path": "tact/example/data/tact-test-data.csv",
        "config_path": "tact/example/config.json",
        "output_dir": "tact/example/output",
    }

    # Methods to test
    methods = ["baseline", "ss-sf", "ssws", "sswsstd"]

    # Initialize TACT
    tact = TACT()

    # Load and process data once
    print("Loading and processing data...")
    data = load_data(config["data_path"])
    binning_processor, ti_data_processor, stats_processor = setup_processors(config["config_path"])
    data = binning_processor.process(data)
    data = ti_data_processor.process(data)

    # Load config for column mapping
    with open(config["config_path"], 'r') as f:
        column_config = json.load(f)

    col_map = column_config["input_data_column_mapping"]
    ref_ti_col = col_map["reference"]["turbulence_intensity"]
    ref_ws_col = col_map["reference"]["wind_speed"]

    # Store results
    results_list = []

    print("\n" + "="*80)
    print("RUNNING ALL ADJUSTMENT METHODS")
    print("="*80)

    for method in methods:
        print(f"\n[{method.upper()}] Running adjustment...")

        try:
            # Run adjustment
            results = tact.adjust(
                data=data,
                method=method,
                parameters={"split": True, "config_path": config["config_path"]}
            )

            # Run validation (no filtering)
            # Baseline uses unadjusted RSD_TI, others use adjTI_RSD_TI
            rsd_ti_col = col_map["rsd"]["primary"]["turbulence_intensity"]
            adjusted_col = rsd_ti_col if method == "baseline" else "adjTI_RSD_TI"

            print(f"[{method.upper()}] Running DNV validation...")
            validation = validate_dnv_rp0661(
                adjusted_data=results["adjusted_data"],
                reference_col=ref_ti_col,
                adjusted_col=adjusted_col,
                wind_speed_col=ref_ws_col,
                bin_col="bins",
                use_test_only=True,
                criteria_type="LV",
                min_ti_threshold=None
            )

            overall = validation["overall"].iloc[0]

            results_list.append({
                "Method": method.upper(),
                "N_obs": int(overall["n_observations"]),
                "MRBE_%": round(overall["MRBE_%"], 2),
                "RRMSE_%": round(overall["RRMSE_%"], 2),
                "Pass_MRBE": overall["pass_MRBE"],
                "Pass_RRMSE": overall["pass_RRMSE"],
                "Overall_Pass": overall["overall_pass"]
            })

            print(f"[{method.upper()}] ✓ Complete")

        except Exception as e:
            print(f"[{method.upper()}] ✗ Error: {str(e)}")
            results_list.append({
                "Method": method.upper(),
                "N_obs": 0,
                "MRBE_%": None,
                "RRMSE_%": None,
                "Pass_MRBE": False,
                "Pass_RRMSE": False,
                "Overall_Pass": False
            })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_list)

    # Print results
    print("\n" + "="*80)
    print("DNV RP-0661 VALIDATION RESULTS - ALL METHODS")
    print("="*80)
    print(comparison_df.to_string(index=False))

    print("\n" + "="*80)
    print("DNV LV Acceptance Criteria:")
    print("  |MRBE| ≤ 5%")
    print("  RRMSE ≤ 15%")
    print("="*80)

    # Identify best method
    valid_results = comparison_df[comparison_df["MRBE_%"].notna()].copy()
    if len(valid_results) > 0:
        # Best by MRBE
        valid_results["MRBE_abs"] = valid_results["MRBE_%"].abs()
        best_mrbe = valid_results.loc[valid_results["MRBE_abs"].idxmin()]

        # Best by RRMSE
        best_rrmse = valid_results.loc[valid_results["RRMSE_%"].idxmin()]

        # Best overall (closest to passing both criteria)
        valid_results["Distance"] = (valid_results["MRBE_abs"] / 5.0) + (valid_results["RRMSE_%"] / 15.0)
        best_overall = valid_results.loc[valid_results["Distance"].idxmin()]

        print("\n" + "="*80)
        print("BEST PERFORMING METHODS")
        print("="*80)
        print(f"Best MRBE:    {best_mrbe['Method']:<12} (MRBE = {best_mrbe['MRBE_%']:+.2f}%)")
        print(f"Best RRMSE:   {best_rrmse['Method']:<12} (RRMSE = {best_rrmse['RRMSE_%']:.2f}%)")
        print(f"Best Overall: {best_overall['Method']:<12} (Distance = {best_overall['Distance']:.2f})")

        # Check if any pass
        passing = comparison_df[comparison_df["Overall_Pass"] == True]
        if len(passing) > 0:
            print("\n✅ PASSING METHODS:")
            for _, row in passing.iterrows():
                print(f"   {row['Method']}")
        else:
            print("\n❌ NO METHODS PASS DNV LV CRITERIA")

    # Save comparison
    output_file = "tact/example/output/method_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📊 Comparison saved to: {output_file}")

    # Generate comparison plots
    plot_method_comparison(comparison_df, config["output_dir"])

    return comparison_df


if __name__ == "__main__":
    compare_all_methods()
