# <img src="docs/assets/cfars_logo_transparent.png" alt="CFARS" width="40" height="40"> TACT - Turbulence intensity Adjustment Comparison Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](legacy/LICENSE)

**TACT** is a Python package for processing, adjusting, and comparing LiDAR-based turbulence intensity measurements with traditional anemometer-based measurements. It provides standardized methods for analyzing wind energy site data and improving measurement accuracy.

## 🚀 Quick Start

### Option 1: Standalone Usage
Get up and running with TACT in minutes:

```bash
# Clone the repository
git clone https://github.com/CFARS/TACT.git
cd TACT

# Install dependencies
pip install -r requirements.txt

# Run with example data
python main.py
```

### Option 2: Install as Python Package (Recommended)
Use TACT in your own scripts from any directory:

```bash
# Clone and install
git clone https://github.com/CFARS/TACT.git
cd TACT
pip install -e .

# Now use from anywhere
cd ~/my_project
python
>>> from tact import TACT
>>> tact = TACT()
>>> # Use TACT in your own scripts!
```

**📖 See the [Installation Guide](docs/installation-guide.md) for detailed setup options**
**📖 For usage instructions, see the [Getting Started Guide](docs/getting-started.md)**

## ✨ Key Features

- **Standardized Data Processing**: Consistent formatting and validation for LiDAR and anemometer data
- **Multiple Adjustment Methods**:
  - Baseline (no adjustment reference)
  - Site-Specific Simple + Filter (SS-SF) - recommended
  - Site-Specific Wind Speed (SSWS)
  - Site-Specific Wind Speed + Std Deviation (SSWSStd)
- **DNV RP-0661 Validation**: Industry-standard validation with MRBE/RRMSE metrics
- **Professional Visualization**: Publication-ready plots with DNV acceptance criteria
- **Method Comparison**: Automated benchmarking framework for all methods
- **Built-in Data Binning**: Wind speed binning for statistical analysis
- **Extensible Architecture**: Easy to add custom adjustment methods
- **Flexible Configuration**: JSON-based configuration system for easy customization

## 📊 What TACT Does

TACT addresses the challenge of comparing turbulence intensity measurements between different sensor types:

1. **Data Input**: Accepts CSV files with reference (anemometer) and RSD (LiDAR) measurements
2. **Processing**: Bins data by wind speed and calculates turbulence intensity metrics
3. **Adjustment**: Applies correction algorithms to improve measurement accuracy
4. **Analysis**: Provides statistical comparisons and validation metrics
5. **Output**: Generates comprehensive reports and adjusted datasets

## 📚 Documentation

Our comprehensive documentation covers everything you need to know:

### Getting Started
- **[Getting Started Guide](docs/getting-started.md)** - Complete walkthrough from installation to results
- **[Data Import Guide](docs/data-import-guide.md)** - How to prepare and load your data
- **[Quick Start](docs/quickstart.md)** - Basic usage examples

### API Reference
- **[TACT Core](docs/api/core/tact.md)** - Main TACT class and factory
- **[Adjustment Methods](docs/api/adjustments/)** - All implemented methods
  - [Baseline](docs/api/adjustments/baseline.md) - No adjustment reference
  - [SS-SF](docs/api/adjustments/sssf.md) - Site-Specific Simple + Filter (recommended)
  - [SSWS](docs/api/adjustments/ssws.md) - Site-Specific Wind Speed
  - [SSWSStd](docs/api/adjustments/sswsstd.md) - SS Wind Speed + Std Deviation
- **[DNV Validation](tact/validation/)** - Industry-standard validation framework
- **[Visualization](tact/visualization/)** - Professional plotting tools

### Guides & Examples
- **[Adding Custom Models](docs/add-custom-model.md)** - Extend TACT with your own methods
- **[Method Comparison Results](METHOD_COMPARISON_RESULTS.md)** - Performance analysis
- **[Example Scripts](tact/example/)** - Working examples and sample data
- **[Complete Documentation](docs/index.md)** - Full documentation index

## 🔧 Usage Examples

### Basic Usage
```python
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors

# Load and process data
data = load_data("your_data.csv")
binning_proc, ti_proc, stats_proc = setup_processors("config.json")
data = binning_proc.process(ti_proc.process(data))

# Initialize TACT and run adjustment
tact = TACT()
results = tact.adjust(
    data=data,
    method="ss-sf",  # Recommended method
    parameters={"split": True, "config_path": "config.json"}
)
```

### With DNV Validation
```python
from tact.validation import validate_dnv_rp0661
from tact.visualization import plot_dnv_validation

# Run adjustment
results = tact.adjust(data, "ss-sf", {"split": True, "config_path": "config.json"})

# Validate against DNV RP-0661
validation = validate_dnv_rp0661(
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    wind_speed_col="ref_ws",
    bin_col="bins",
    criteria_type="LV"  # Load Verification criteria
)

# Generate professional plots
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

### Compare All Methods
```python
from compare_all_methods import compare_all_methods

# Runs baseline, ss-sf, ssws, sswsstd and compares performance
comparison = compare_all_methods()
print(comparison)
```

### Configuration
```json
{
    "input_data_column_mapping": {
        "reference": {
            "wind_speed": "ref_ws",
            "turbulence_intensity": "ref_ti"
        },
        "rsd": {
            "primary": {
                "wind_speed": "rsd_ws",
                "turbulence_intensity": "rsd_ti"
            }
        }
    }
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](legacy/LICENSE) file for details.

## 📞 Support

- **Documentation**: [docs/index.md](docs/index.md)
- **Issues**: [GitHub Issues](https://github.com/CFARS/TACT/issues)
- **Contact**: aea@nrgsystems.com

---

**Note**: For the original legacy TACT implementation, see the [legacy](legacy/) directory.