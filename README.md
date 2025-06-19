# <img src="docs/assets/cfars_logo_transparent.png" alt="CFARS" width="40" height="40"> TACT - Turbulence intensity Adjustment Comparison Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](legacy/LICENSE)

**TACT** is a Python package for processing, adjusting, and comparing LiDAR-based turbulence intensity measurements with traditional anemometer-based measurements. It provides standardized methods for analyzing wind energy site data and improving measurement accuracy.

## 🚀 Quick Start

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

**📖 For detailed setup and usage instructions, see the [Quick Start Guide](docs/quickstart.md)**

## ✨ Key Features

- **Standardized Data Processing**: Consistent formatting and validation for LiDAR and anemometer data
- **Multiple Adjustment Methods**: 
  - Baseline adjustments for fundamental comparisons
  - Site-Specific Simple + Filter (SS-SF) for advanced analysis
  - More models soon!
- **Built-in Data Binning**: Compare results across wind speeds for binned statistical analysis
- **Statistical Analysis Tools**: Comprehensive metrics and regression analysis for comparing adjustment methods
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

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in minutes
- **[API Reference](docs/api/core/tact.md)** - Detailed technical documentation
- **[Adjustment Methods](docs/api/adjustments/)** - Learn about different correction algorithms
- **[Configuration Guide](docs/api/core/config.md)** - Set up your data and parameters
- **[Examples](tact/example/)** - Working examples and sample data

## 🔧 Usage Examples

### Basic Usage
```python
from tact import TACT

# Initialize TACT
tact = TACT()

# Load and process data
data = load_data("your_data.csv")

# Perform SS-SF adjustment
results = tact.adjust(
    data=data, 
    method="ss-sf", 
    parameters={"split": True}
)
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