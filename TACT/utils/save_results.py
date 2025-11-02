import pandas as pd
from typing import Optional, Dict

def save_results(means: pd.DataFrame, adjusted_data: pd.DataFrame,
                reg_results: pd.DataFrame, method: str, output_dir: str,
                validation_results: Optional[Dict[str, pd.DataFrame]] = None):
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
    """
    means.to_csv(f"{output_dir}/{method}_all_stats.csv", index=False)
    adjusted_data.to_csv(f"{output_dir}/{method}_adjusted_data.csv", index=False)
    reg_results.to_csv(f"{output_dir}/{method}_reg_results.csv", index=False)

    if validation_results is not None:
        # Save overall validation metrics
        validation_results["overall"].to_csv(
            f"{output_dir}/{method}_validation_overall.csv",
            index=False
        )

        # Save per-bin validation metrics
        validation_results["by_bin"].to_csv(
            f"{output_dir}/{method}_validation_by_bin.csv",
            index=False
        )

        print(f"✅ Success! Results for {method} adjustment method saved to: {output_dir}")
        print(f"   - Adjusted data, statistics, regression results, and DNV RP-0661 validation")
    else:
        print(f"✅ Success! Results for {method} adjustment method saved to: {output_dir}")