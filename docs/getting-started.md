# Getting Started with TACT

This guide will walk you through everything you need to know to run TACT on your own data.

## Table of Contents

1. [Installation](#installation)
2. [Data Requirements](#data-requirements)
3. [Configuration Setup](#configuration-setup)
4. [Running Your First Adjustment](#running-your-first-adjustment)
5. [Understanding the Output](#understanding-the-output)
6. [DNV Validation](#dnv-validation)
7. [Comparing Methods](#comparing-methods)
8. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/CFARS/TACT.git
cd TACT
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

**On Windows:**
```bash
python -m venv env
env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from tact import TACT; print('✅ TACT installed successfully')"
```

---

## Data Requirements

### Input Data Format

TACT requires CSV data with the following structure:

```csv
timestamp,ref_ws,ref_sd,ref_ti,rsd_ws,rsd_sd,rsd_ti
2024-01-01 00:00:00,8.5,1.2,0.141,8.3,1.3,0.157
2024-01-01 00:10:00,9.2,1.4,0.152,9.0,1.5,0.167
...
```

### Required Columns

At minimum, your CSV must include:

| Measurement Type | Required Columns |
|------------------|------------------|
| **Reference (Anemometer)** | Wind speed, Standard deviation, Turbulence intensity |
| **RSD (LiDAR)** | Wind speed, Standard deviation, Turbulence intensity |
| **Timestamp** | DateTime column (any format) |

### Column Naming

You can use any column names - just map them in the config file (see below).

**Example column names:**
- Reference: `ref_ws`, `ref_sd`, `ref_ti`
- RSD: `rsd_ws`, `rsd_sd`, `rsd_ti`
- Timestamp: `timestamp`, `datetime`, `time`

### Data Quality Requirements

For best results:

- ✅ **Synchronized measurements**: Reference and RSD data from same time periods
- ✅ **Sufficient data**: At least 500+ paired observations (2000+ recommended)
- ✅ **Complete wind speed range**: Data across 4-20 m/s range
- ✅ **Quality-filtered**: Remove invalid/flagged measurements before import
- ✅ **Consistent units**: Wind speed in m/s, TI as decimal (not percentage)

### Example Dataset

TACT includes example data for testing:

```bash
tact/example/data/tact-test-data.csv
```

This contains:
- 15,484 observations
- Reference tower and RSD measurements
- Pre-filtered and quality-checked
- Ready to use for testing

---

## Configuration Setup

### Step 1: Create Configuration File

Create a JSON file (e.g., `config.json`) with your column mappings:

```json
{
    "input_data_column_mapping": {
        "reference": {
            "wind_speed": "ref_ws",
            "wind_speed_std": "ref_sd",
            "turbulence_intensity": "ref_ti"
        },
        "rsd": {
            "primary": {
                "wind_speed": "rsd_ws",
                "wind_speed_std": "rsd_sd",
                "turbulence_intensity": "rsd_ti"
            }
        }
    },
    "binning_config": {
        "bin_size": 1.0,
        "bin_min": 4.0,
        "bin_max": 20.0
    }
}
```

### Configuration Sections

#### Column Mapping

Maps your CSV column names to TACT's internal names:

```json
"input_data_column_mapping": {
    "reference": {
        "wind_speed": "YOUR_REF_WS_COLUMN",
        "wind_speed_std": "YOUR_REF_SD_COLUMN",
        "turbulence_intensity": "YOUR_REF_TI_COLUMN"
    },
    "rsd": {
        "primary": {
            "wind_speed": "YOUR_RSD_WS_COLUMN",
            "wind_speed_std": "YOUR_RSD_SD_COLUMN",
            "turbulence_intensity": "YOUR_RSD_TI_COLUMN"
        }
    }
}
```

#### Binning Configuration

Controls how data is grouped by wind speed:

```json
"binning_config": {
    "bin_size": 1.0,    // Bin width in m/s (typically 0.5 or 1.0)
    "bin_min": 4.0,     // Minimum wind speed (m/s)
    "bin_max": 20.0     // Maximum wind speed (m/s)
}
```

**Common bin sizes:**
- 1.0 m/s: Standard, good balance
- 0.5 m/s: Higher resolution, needs more data
- 2.0 m/s: Coarser, for limited datasets

### Step 2: Use Example Configuration

For testing, use the provided example:

```bash
cp tact/example/config.json my_config.json
# Edit my_config.json to match your data
```

---

## Running Your First Adjustment

### Basic Workflow

Here's a complete example using TACT:

```python
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors

# 1. Load your data
data = load_data("your_data.csv")

# 2. Set up processors
binning_processor, ti_processor, stats_processor = setup_processors("config.json")

# 3. Process data
data = binning_processor.process(data)  # Bins by wind speed
data = ti_processor.process(data)        # Calculates TI metrics

# 4. Initialize TACT
tact = TACT()

# 5. Run adjustment
results = tact.adjust(
    data=data,
    method="ss-sf",  # Site-Specific Simple + Filter (best performer)
    parameters={
        "split": True,
        "config_path": "config.json"
    }
)

# 6. Access results
adjusted_data = results["adjusted_data"]
regression_stats = results["reg_results"]
all_stats = results["all_stats"]

print("✅ Adjustment complete!")
print(f"Processed {len(adjusted_data)} observations")
```

### Save as Script

Create `run_tact.py`:

```python
#!/usr/bin/env python3
"""
Run TACT adjustment on your data
"""
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors
from tact.utils.save_results import save_results

# Configuration
DATA_PATH = "your_data.csv"
CONFIG_PATH = "config.json"
OUTPUT_DIR = "output"
METHOD = "ss-sf"

def main():
    # Load and process
    print("Loading data...")
    data = load_data(DATA_PATH)

    print("Processing data...")
    binning_proc, ti_proc, stats_proc = setup_processors(CONFIG_PATH)
    data = binning_proc.process(data)
    data = ti_proc.process(data)

    # Run adjustment
    print(f"Running {METHOD} adjustment...")
    tact = TACT()
    results = tact.adjust(
        data=data,
        method=METHOD,
        parameters={"split": True, "config_path": CONFIG_PATH}
    )

    # Save results
    print(f"Saving results to {OUTPUT_DIR}/...")
    save_results(
        adjusted_data=results["adjusted_data"],
        reg_results=results["reg_results"],
        all_stats=results["all_stats"],
        method=METHOD,
        output_dir=OUTPUT_DIR
    )

    print("✅ Complete!")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python run_tact.py
```

### Test with Example Data

Before using your own data, test with the example:

```bash
python main.py
```

This runs the complete pipeline with example data and generates:
- Adjusted data CSV
- Regression results CSV
- Statistics CSV
- DNV validation results
- Visualization plots

---

## Understanding the Output

### Output Files

After running TACT, you'll find these files in your output directory:

```
output/
├── ss-sf_adjusted_data.csv       # Full dataset with adjustments
├── ss-sf_reg_results.csv         # Regression model parameters
├── ss-sf_all_stats.csv           # Statistical summary
├── ss-sf_validation_overall.csv  # DNV validation summary
├── ss-sf_validation_by_bin.csv   # DNV validation per bin
└── plots/                        # Visualization plots
    ├── ss-sf_mrbe_by_bin.png
    ├── ss-sf_rrmse_by_bin.png
    ├── ss-sf_ti_scatter.png
    └── ss-sf_ti_comparison.png
```

### Key Output Columns

**Adjusted Data** (`*_adjusted_data.csv`):

| Column | Description |
|--------|-------------|
| `adjTI_RSD_TI` | Adjusted turbulence intensity |
| `adjRepTI_RSD_RepTI` | Adjusted representative TI |
| `bins` | Wind speed bin assignment |
| `split` | Train (1) or test (0) data |

**Regression Results** (`*_reg_results.csv`):

| Column | Description |
|--------|-------------|
| `m` | Regression slope |
| `c` | Regression intercept |
| `rsquared` | R² goodness of fit |

**Validation Results** (`*_validation_overall.csv`):

| Column | Description |
|--------|-------------|
| `MRBE_%` | Mean Relative Bias Error (%) |
| `RRMSE_%` | Relative Root Mean Square Error (%) |
| `pass_MRBE` | True if \|MRBE\| ≤ 5% |
| `pass_RRMSE` | True if RRMSE ≤ 15% |
| `overall_pass` | True if both criteria pass |

### Interpreting Validation Results

**DNV RP-0661 Load Verification (LV) Criteria:**
- ✅ **Pass**: |MRBE| ≤ 5% AND RRMSE ≤ 15%
- ❌ **Fail**: Either criterion exceeded

**What the metrics mean:**
- **MRBE**: Systematic bias (on average, is measurement too high or low?)
- **RRMSE**: Scatter/consistency (how much do individual points vary?)

**Example interpretation:**
```
MRBE = +37.8%  →  On average, adjusted TI is 37.8% too high
RRMSE = 111.1% →  Individual measurements have 111% scatter
```

---

## DNV Validation

### Running Validation

```python
from tact.validation import validate_dnv_rp0661

validation = validate_dnv_rp0661(
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    wind_speed_col="ref_ws",
    bin_col="bins",
    use_test_only=True,    # Validate only test data
    criteria_type="LV",    # Load Verification (most common)
    min_ti_threshold=None  # Optional: filter low TI values
)

# Check overall results
overall = validation["overall"].iloc[0]
print(f"MRBE: {overall['MRBE_%']:.2f}%")
print(f"RRMSE: {overall['RRMSE_%']:.2f}%")
print(f"Pass: {overall['overall_pass']}")

# Examine by wind speed bin
by_bin = validation["by_bin"]
print(by_bin[["bin_center", "MRBE_%", "RRMSE_%", "n_observations"]])
```

### Generating Validation Plots

```python
from tact.visualization import plot_dnv_validation

plot_dnv_validation(
    validation_results=validation,
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    unadjusted_col="rsd_ti",
    wind_speed_col="ref_ws",
    bin_col="bins",
    method_name="SS-SF",
    output_dir="output/plots"
)
```

This generates 4 plots:
1. **MRBE by wind speed bin** - Shows bias with acceptance limits
2. **RRMSE by wind speed bin** - Shows scatter with acceptance zone
3. **TI scatter plot** - Adjusted vs reference with 1:1 line
4. **TI comparison** - Reference vs unadjusted vs adjusted by bin

### Criteria Types

TACT supports 3 DNV criteria types:

| Type | Use Case | MRBE Limit | RRMSE Limit |
|------|----------|------------|-------------|
| **LV** | Load Verification (most common) | ≤5% | ≤15% |
| **EP** | Energy Production | ≤10% | No limit |
| **SS** | Site Suitability | Varies by WS | Varies by WS |

Specify with `criteria_type` parameter:
```python
validation = validate_dnv_rp0661(
    ...,
    criteria_type="LV"  # or "EP" or "SS"
)
```

---

## Comparing Methods

### Running All Methods

Use the comparison script to test all available methods:

```python
from compare_all_methods import compare_all_methods

# Runs baseline, ss-sf, ssws, sswsstd
comparison = compare_all_methods()

# View results
print(comparison)
```

Output:
```
  Method  N_obs  MRBE_%  RRMSE_%  Pass_MRBE  Pass_RRMSE  Overall_Pass
BASELINE  15484   90.39   183.04      False       False         False
   SS-SF   3036   39.47   109.43      False       False         False
    SSWS   3054   94.77   188.03      False       False         False
 SSWSSTD   3137   37.73   121.61      False       False         False
```

### Comparison Plots

The script automatically generates comparison plots showing all methods side-by-side:

```
output/plots/method_comparison.png
```

### Choosing the Best Method

**For most use cases**: Use **SS-SF**
- Simplest implementation
- Best or tied-best performance on most datasets
- Direct TI adjustment (no error propagation)
- Industry-proven approach

**Consider alternatives if**:
- SS-SF fails validation but others pass
- You have specific requirements (e.g., must adjust WS and SD separately)
- Site-specific testing shows better performance with another method

---

## Next Steps

### Adding Your Own Method

Follow the [Custom Model Tutorial](add-custom-model.md) to create custom adjustment methods.

Quick overview:
1. Extend `AdjustmentMethod` base class
2. Register with `@AdjustmentRegistry.register("method-name")`
3. Implement required methods
4. Use immediately with TACT

### Advanced Usage

**Multi-height analysis:**
- Configure multiple RSD heights in config file
- Access height-specific results in output

**Batch processing:**
- Loop over multiple sites/datasets
- Aggregate validation results
- Compare site-specific performance

**Integration with workflows:**
- Call TACT from shell scripts
- Integrate with data pipelines
- Automate reporting

### Documentation

- **[API Reference](api/core/tact.md)** - Detailed technical docs
- **[Method Comparison Results](../METHOD_COMPARISON_RESULTS.md)** - Performance analysis
- **[Adding Custom Models](add-custom-model.md)** - Extensibility guide
- **[DNV Validation](api/validation/dnv_rp0661.md)** - Validation details

### Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/CFARS/TACT/issues)
- **Email**: aea@nrgsystems.com
- **Documentation**: This docs directory

---

## Troubleshooting

### Common Issues

**Import Error: `ModuleNotFoundError: No module named 'tact'`**

Solution: Activate virtual environment
```bash
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows
```

---

**KeyError: Column not found**

Solution: Check config file column mappings match your CSV
```bash
# View your CSV columns
python -c "import pandas as pd; print(pd.read_csv('your_data.csv').columns.tolist())"
```

---

**ValueError: Not enough data after filtering**

Solution:
- Check data quality (nulls, invalid values)
- Verify wind speed range (need 4-20 m/s coverage)
- Reduce bin size if data is sparse

---

**Poor validation results (high MRBE/RRMSE)**

This is often a data quality issue, not a code problem. Check:
1. RSD and reference are properly time-synchronized
2. Data has been quality-filtered (CNR, availability, etc.)
3. RSD is close enough to reference tower (<200m separation)
4. Sensors are properly calibrated

---

## Quick Reference

### Minimal Example

```python
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors

data = load_data("data.csv")
bp, tp, sp = setup_processors("config.json")
data = bp.process(tp.process(data))

tact = TACT()
results = tact.adjust(data, "ss-sf", {"split": True, "config_path": "config.json"})
```

### Available Methods

- `baseline` - No adjustment
- `ss-sf` - Site-Specific Simple + Filter (recommended)
- `ssws` - Site-Specific Wind Speed
- `sswsstd` - Site-Specific Wind Speed + Standard Deviation

### Important Paths

- Example data: `tact/example/data/tact-test-data.csv`
- Example config: `tact/example/config.json`
- Example script: `main.py`
- Comparison script: `compare_all_methods.py`

---

**Ready to get started? Run the example:**

```bash
python main.py
```

Then adapt it for your own data!
