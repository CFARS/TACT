# TACT - Turbulence Adjustment Comparison Tool

<img src="assets/cfars_logo_transparent.png" alt="CFARS" width="60" height="60" />

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](legacy/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-tact.akleao.com-blue)](https://tact.akleao.com)

**TACT** is a Python package for correcting remote sensing device (RSD) turbulence intensity measurements to match traditional cup anemometer standards. By training on concurrent measurement campaigns, TACT produces adjusted turbulence data suitable for turbine load calculations, DNV RP-0661 compliant site assessments, and IEC-standard power performance testing.

## 📖 Documentation

**Complete documentation is available at [tact.akleao.com](https://tact.akleao.com)**

- [Getting Started](https://tact.akleao.com/getting-started) - Installation and first adjustment
- [Data Import Guide](https://tact.akleao.com/data-import-guide) - Prepare your data
- [API Reference](https://tact.akleao.com/api/core/tact) - Complete API documentation
- [Add Custom Methods](https://tact.akleao.com/add-custom-model) - Extend TACT

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/CFARS/TACT.git
cd TACT
pip install -r requirements.txt

# Run example
python main.py
```

See the [installation guide](https://tact.akleao.com/installation-guide) for detailed setup options.

## ✨ Features

- **Multiple Adjustment Methods**: SS-SF, SSWSStd, SSWS, and Baseline
- **DNV RP-0661 Validation**: Industry-standard validation with MRBE/RRMSE metrics
- **Visualization**: Generate validation plots with acceptance criteria
- **Extensible**: Add custom adjustment methods using the plugin system
- **Method Comparison**: Compare all methods on your data automatically

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/CFARS/TACT.git
cd TACT

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
pytest
```

### Contribution Guidelines

1. **Fork the repository** and create a feature branch
2. **Make your changes** following our code style:
   - Use type hints for function parameters and returns
   - Follow PEP 8 style guidelines
   - Add docstrings to new functions/classes
   - Include unit tests for new features
3. **Test your changes** - ensure all tests pass
4. **Submit a pull request** with a clear description

### Areas for Contribution

- **New adjustment methods**: Implement additional TI adjustment algorithms
- **Validation improvements**: Add support for other validation standards
- **Documentation**: Improve guides, add examples, fix typos
- **Bug fixes**: Address issues from the issue tracker
- **Performance**: Optimize data processing and computation

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

## 🏗️ Project Structure

```
TACT/
├── tact/                  # Main package
│   ├── computation/       # Adjustment method implementations
│   ├── core/             # Core TACT class and base classes
│   ├── utils/            # Data processing utilities
│   ├── validation/       # DNV validation framework
│   └── visualization/    # Plotting functions
├── docs/                 # Documentation (Mintlify)
├── tests/                # Unit tests
└── legacy/              # Original implementation (archived)
```

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](legacy/LICENSE) file for details.

## 📞 Support

- **Documentation**: [tact.akleao.com](https://tact.akleao.com)
- **Issues**: [GitHub Issues](https://github.com/CFARS/TACT/issues)
- **Contact**: aea@nrgsystems.com

---

For detailed usage instructions, API reference, and guides, visit **[tact.akleao.com](https://tact.akleao.com)**
