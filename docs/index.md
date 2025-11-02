# TACT Documentation

**TACT** (Turbulence Adjustment Computation Tool) is a Python package for processing, adjusting, and comparing LiDAR-based turbulence intensity measurements with traditional anemometer-based measurements.

## 🚀 Getting Started

New to TACT? Start here:

- **[Installation Guide](installation-guide.md)** - Install TACT as a Python package
- **[Getting Started Guide](getting-started.md)** - Complete walkthrough from installation to first results
- **[Data Import Guide](data-import-guide.md)** - How to prepare and load your data
- **[Quick Start](quickstart.md)** - Basic usage examples

## ✨ Features

- **Standardized Data Processing**: Consistent formatting and validation for LiDAR and anemometer data
- **Multiple Adjustment Methods**:
    - Baseline (no adjustment reference)
    - Site-Specific Simple + Filter (SS-SF) - recommended
    - Site-Specific Wind Speed (SSWS)
    - Site-Specific Wind Speed + Std Dev (SSWSStd)
- **DNV RP-0661 Validation**: Industry-standard validation with MRBE/RRMSE metrics
- **Professional Visualization**: Publication-ready plots with DNV acceptance criteria
- **Method Comparison Framework**: Automated benchmarking of all methods
- **Extensible Architecture**: Easy to add custom adjustment methods
- **Statistical Analysis Tools**: Comprehensive metrics and regression analysis

## 📖 User Guides

### Core Workflows

- **[Getting Started](getting-started.md)** - Installation, first run, and basic usage
- **[Data Import Guide](data-import-guide.md)** - Preparing and loading your data
- **[Adding Custom Models](add-custom-model.md)** - Extending TACT with your own methods

### Example Scripts

- **[main.py](../main.py)** - Complete pipeline with DNV validation and visualization
- **[compare_all_methods.py](../compare_all_methods.py)** - Compare all adjustment methods
- **[Example Test Scripts](../tact/example/)** - Method-specific examples

## 📚 API Documentation

### Core Components

- **[TACT Core](api/core/tact.md)** - Main TACT class and factory
- **[Base Classes](api/core/base.md)** - Abstract base classes for methods
- **[Registry System](api/core/registry.md)** - Method registration

### Adjustment Methods

- **[Baseline](api/adjustments/baseline.md)** - No adjustment (reference comparison)
- **[SS-SF](api/adjustments/sssf.md)** - Site-Specific Simple + Filter (recommended)
- **[SSWS](api/adjustments/ssws.md)** - Site-Specific Wind Speed
- **[SSWSStd](api/adjustments/sswsstd.md)** - SS Wind Speed + Standard Deviation

### Validation & Visualization

- **DNV RP-0661 Validation** (`tact/validation/dnv_rp0661.py`) - Industry-standard validation
- **Visualization Suite** (`tact/visualization/dnv_plots.py`) - Professional plotting

### Utilities

- **[Data Processing](api/utils/data_processing.md)** - Loading and processing data
- **[Statistics](api/utils/statistics.md)** - Statistical computations
- **[Setup](api/utils/setup.md)** - Configuration and processors

## 📊 Method Comparison

See **[Method Comparison Results](../METHOD_COMPARISON_RESULTS.md)** for detailed performance analysis of all implemented methods on example data.

**Quick Summary:**
- **SS-SF**: Best overall performance (MRBE: 37.8%, RRMSE: 111.1%)
- **SSWSStd**: Second best (MRBE: 37.7%, RRMSE: 121.6%)
- **SSWS**: Third (MRBE: 92.1%, RRMSE: 188.0%)
- **Baseline**: Reference (MRBE: 90.4%, RRMSE: 183.0%)

## 🔧 Configuration

TACT uses JSON configuration files to map your data columns and set processing parameters:

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

See [Data Import Guide](data-import-guide.md#configuration-mapping) for details.

## 🎯 Quick Examples

### Run Adjustment with Validation

```python
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors
from tact.validation import validate_dnv_rp0661

# Load and process data
data = load_data("data.csv")
bp, tp, sp = setup_processors("config.json")
data = bp.process(tp.process(data))

# Run adjustment
tact = TACT()
results = tact.adjust(data, "ss-sf", {"split": True, "config_path": "config.json"})

# Validate with DNV RP-0661
validation = validate_dnv_rp0661(
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    wind_speed_col="ref_ws",
    bin_col="bins",
    criteria_type="LV"
)

print(f"MRBE: {validation['overall']['MRBE_%'].iloc[0]:.2f}%")
print(f"RRMSE: {validation['overall']['RRMSE_%'].iloc[0]:.2f}%")
```

### Compare All Methods

```python
from compare_all_methods import compare_all_methods

# Runs baseline, ss-sf, ssws, sswsstd
comparison = compare_all_methods()
print(comparison)
```

### Add Custom Method

```python
from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry

@AdjustmentRegistry.register("my-method")
class MyMethod(AdjustmentMethod):
    def required_model_parameters(self):
        return {"config_path": str}

    def required_data_columns(self):
        return ["reference.wind_speed", ...]

    def adjust(self, data, parameters):
        # Your implementation
        return {"adjusted_data": adjusted_data}
```

See [Adding Custom Models](add-custom-model.md) for complete tutorial.

## 📦 Project Structure

```
TACT/
├── tact/                      # Main package
│   ├── adjustments/          # Adjustment method implementations
│   ├── validation/           # DNV RP-0661 validation
│   ├── visualization/        # Plotting functions
│   ├── core/                 # Core infrastructure
│   ├── utils/                # Utility functions
│   ├── example/              # Example data and configs
│   └── tests/                # Test suite
├── docs/                     # Documentation (you are here!)
├── legacy/                   # Original monolithic implementation
├── main.py                   # Example pipeline script
├── compare_all_methods.py    # Method comparison script
└── requirements.txt          # Dependencies
```

## 🤝 Contributing

We welcome contributions! See the [Contributing Guide](contributing.md) for:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](../legacy/LICENSE) file for details.

## 📞 Support

- **Documentation**: You're reading it!
- **Issues**: [GitHub Issues](https://github.com/CFARS/TACT/issues)
- **Email**: aea@nrgsystems.com

## 📝 Additional Resources

- **[Status Report](../STATUS.md)** - Current implementation status and comparison with legacy
- **[Method Comparison Results](../METHOD_COMPARISON_RESULTS.md)** - Detailed performance analysis
- **[README](../README.md)** - Project overview and quick start

---

**Ready to get started?** → [Getting Started Guide](getting-started.md)