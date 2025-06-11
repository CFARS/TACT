# Quick Start Guide

This guide will walk you through using TACT to adjust turbulence intensity measurements using the Site-Specific Simple + Filter (SS-SF) method.

## Installation

### From GitHub

1. Clone the repository:
```bash
git clone https://github.com/CFARS/TACT.git
cd TACT
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Required Files

### 1. Input Data

Your input CSV file should contain the following columns:
- Reference wind speed measurements
- Reference turbulence intensity
- RSD (Remote Sensing Device) wind speed measurements
- RSD turbulence intensity

Example data format (`data.csv`):
```csv
timestamp,ref_ws,ref_ti,rsd_ws,rsd_ti
2024-01-01 00:00:00,5.2,0.12,5.1,0.15
2024-01-01 00:10:00,6.1,0.14,6.0,0.16
...
```

### 2. Configuration File

Create a `config.json` file that maps your data columns:

```json
{
    "input_data_column_mapping": {
        "reference": {
            "wind_speed": "ref_ws",
            "turbulence_intensity": "ref_ti",
            "standard_deviation": "ref_sd"
        },
        "rsd": {
            "primary": {
                "wind_speed": "rsd_ws",
                "turbulence_intensity": "rsd_ti",
                "standard_deviation": "rsd_sd"
            }
        }
    }
}
```

## Running the Adjustment

Here's a complete example of how to use TACT with the SS-SF method:

```python
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors
from tact.utils.process_statistics import process_statistics
from tact.utils.save_results import save_results

def main():
    # Configuration
    config = {
        "data_path": "data.csv",              # Your input data file
        "config_path": "config.json",         # Your column mapping file
        "output_dir": "output",              # Where to save results
        "method": "ss-sf",                   # Using SS-SF method
        "parameters": {"split": True}         # Split data into training/testing sets
    }
    
    # Initialize TACT
    tact = TACT()
    
    # Load and process data
    data = load_data(config["data_path"])
    
    # Setup data processors
    binning_processor, ti_data_processor, stats_processor = setup_processors(config["config_path"])
    
    # Process data through pipeline
    data = binning_processor.process(data)    # Bin wind speed data
    data = ti_data_processor.process(data)    # Calculate TI metrics
    
    # Perform SS-SF adjustment
    results = tact.adjust(
        data=data, 
        method=config["method"], 
        parameters={**config["parameters"], "config_path": config["config_path"]}
    )
    
    # Calculate statistics
    means = process_statistics(results["adjusted_data"], stats_processor)
    
    # Save all results
    save_results(
        means=means,
        adjusted_data=results["adjusted_data"],
        reg_results=results["reg_results"],
        method=config["method"],
        output_dir=config["output_dir"]
    )

if __name__ == "__main__":
    main()
```

## Understanding the Process

The SS-SF adjustment method:
1. **Data Preparation**:
   - Bins wind speed data into categories
   - Calculates turbulence intensity metrics
   - Splits data into training (80%) and testing (20%) sets

2. **Adjustment**:
   - Filters out TI measurements > 0.3 from training data
   - Performs linear regression between RSD and reference TI
   - Applies the correction to the test data

3. **Results**:
   - `adjusted_data`: DataFrame containing:
     - Original RSD and reference TI values
     - Adjusted TI values
     - Binning information
   - `reg_results`: Regression statistics including:
     - Slope and intercept
     - R² value
     - RMSE (Root Mean Square Error)

## Next Steps

- Check out the [API Reference](api/core/tact.md) for detailed documentation
- Learn about other [adjustment methods](api/adjustments/baseline.md)
- Explore [statistical utilities](api/utils/statistics.md) for analysis 