import pandas as pd
import os
from trash.BaselineResults import BaselineResults

test_name = 'baseline'

# Load the data and config using absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'tact-test-data.csv')
config_path = os.path.join(script_dir, 'config.json')
metrics_path = os.path.join(script_dir, 'test_outputs', f'{test_name}_metrics.csv')

data = pd.read_csv(data_path)

# Create and run adjustment
method = BaselineResults()
result = method.adjust(data, {"config_path": config_path})

# Convert metrics list to DataFrame
metrics_df = pd.DataFrame(result.metrics)

# Save metrics to CSV
metrics_df.to_csv(metrics_path, index=False)

