import pandas as pd

def save_results(means: pd.DataFrame, adjusted_data: pd.DataFrame, 
                reg_results: pd.DataFrame, method: str, output_dir: str):
    """Save all results to files"""
    means.to_csv(f"{output_dir}/{method}_all_stats.csv")
    adjusted_data.to_csv(f"{output_dir}/{method}_adjusted_data.csv")
    reg_results.to_csv(f"{output_dir}/{method}_reg_results.csv")