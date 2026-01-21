import pandas as pd
import os
from typing import Optional, Dict

def save_results(means: pd.DataFrame, adjusted_data: pd.DataFrame,
                reg_results: Optional[pd.DataFrame], method: str, output_dir: str,
                validation_results: Optional[Dict[str, pd.DataFrame]] = None,
                iea_validation_results: Optional[Dict] = None):
    """
    Save all results to files.

    Parameters
    ----------
    means : pd.DataFrame
        Mean statistics by wind speed bin
    adjusted_data : pd.DataFrame
        Full adjusted dataset
    reg_results : pd.DataFrame
        Regression results
    method : str
        Adjustment method name
    output_dir : str
        Output directory path
    validation_results : dict, optional
        DNV RP-0661 validation results with "overall" and "by_bin" DataFrames
    iea_validation_results : dict, optional
        IEA Task 52 KPIs validation results with "overall", "by_bin", and "metadata"
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use os.path.join for proper path handling across platforms
    means.to_csv(os.path.join(output_dir, f"{method}_all_stats.csv"), index=False)
    adjusted_data.to_csv(os.path.join(output_dir, f"{method}_adjusted_data.csv"), index=False)
    if reg_results is not None:
        reg_results.to_csv(os.path.join(output_dir, f"{method}_reg_results.csv"), index=False)

    if validation_results is not None:
        # Save overall validation metrics
        validation_results["overall"].to_csv(
            os.path.join(output_dir, f"{method}_validation_overall.csv"),
            index=False
        )

        # Save per-bin validation metrics
        validation_results["by_bin"].to_csv(
            os.path.join(output_dir, f"{method}_validation_by_bin.csv"),
            index=False
        )

    if iea_validation_results is not None:
        # Save IEA Task 52 KPIs overall metrics
        iea_validation_results["overall"].to_csv(
            os.path.join(output_dir, f"{method}_iea_task52_overall.csv"),
            index=False
        )

        # Save IEA Task 52 KPIs per-bin metrics
        iea_validation_results["by_bin"].to_csv(
            os.path.join(output_dir, f"{method}_iea_task52_by_bin.csv"),
            index=False
        )

    # Print success message
    saved_items = ["Adjusted data", "statistics"]
    if reg_results is not None:
        saved_items.append("regression results")
    if validation_results is not None:
        saved_items.append("DNV RP-0661 validation")
    if iea_validation_results is not None:
        saved_items.append("IEA Task 52 KPIs validation")
    
    print(f"✅ Success! Results for {method} adjustment method saved to: {output_dir}")
    print(f"   - {', '.join(saved_items)}")