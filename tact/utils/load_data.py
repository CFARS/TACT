import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)