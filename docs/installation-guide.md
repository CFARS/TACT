# Installation Guide

TACT can be used in two ways:
1. **As a standalone tool** - Run scripts from the TACT directory
2. **As an importable Python package** - Use TACT in your own scripts anywhere

---

## Method 1: Standalone Usage (Current Setup)

This is how you're currently using TACT - running scripts from within the TACT directory.

### Installation
```bash
# Clone repository
git clone https://github.com/CFARS/TACT.git
cd TACT

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run example
python main.py
```

### Usage
You must be in the TACT directory or adjust your Python path:
```python
# my_script.py (in TACT directory)
from tact import TACT
tact = TACT()
# ... use TACT
```

---

## Method 2: Install as Python Package (Recommended for External Use)

This allows you to use TACT from **any directory** in your own scripts.

### Installation

#### Option A: Install in Development Mode (Editable)
Best for development - changes to TACT source code are immediately available:

```bash
# Clone repository
git clone https://github.com/CFARS/TACT.git
cd TACT

# Create/activate virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install in editable mode
pip install -e .
```

#### Option B: Install as Regular Package
Best for production use - installs a snapshot:

```bash
# Clone repository
git clone https://github.com/CFARS/TACT.git
cd TACT

# Create/activate virtual environment
python3 -m venv env
source env/bin/activate

# Install package
pip install .
```

#### Option C: Install from GitHub (Future)
Once published, users could install directly:
```bash
pip install git+https://github.com/CFARS/TACT.git
```

### Usage After Installation

Once installed, you can import TACT from **anywhere**:

```python
# my_analysis.py (can be in ANY directory)
from tact import TACT
from tact.validation import validate_dnv_rp0661
from tact.visualization import plot_dnv_validation
from tact.utils.load_data import load_data

# Use TACT normally
tact = TACT()
data = load_data("/path/to/my/data.csv")
results = tact.adjust(data, "ss-sf", {"split": True, "config_path": "/path/to/config.json"})
```

---

## Verification

Test that installation worked:

```bash
# Activate your environment
source env/bin/activate  # or env\Scripts\activate on Windows

# Test import
python -c "from tact import TACT; print('✅ TACT installed successfully')"

# Test from different directory
cd /tmp
python -c "from tact import TACT; tact = TACT(); print('✅ Can import from anywhere')"
```

---

## Example: Using TACT as a Module in Your Own Script

Create a script anywhere on your system:

```python
#!/usr/bin/env python3
"""
my_wind_analysis.py - Custom wind analysis using TACT

This script can be located anywhere, not just in the TACT directory.
"""

from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors
from tact.validation import validate_dnv_rp0661
from tact.visualization import plot_dnv_validation
import pandas as pd

def analyze_site(data_path, config_path, output_dir):
    """Run complete TACT analysis on a site."""

    # Load data
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)

    # Process
    print("Processing data...")
    binning_proc, ti_proc, stats_proc = setup_processors(config_path)
    data = binning_proc.process(data)
    data = ti_proc.process(data)

    # Run adjustment
    print("Running SS-SF adjustment...")
    tact = TACT()
    results = tact.adjust(
        data=data,
        method="ss-sf",
        parameters={"split": True, "config_path": config_path}
    )

    # Validate
    print("Validating with DNV RP-0661...")
    validation = validate_dnv_rp0661(
        adjusted_data=results["adjusted_data"],
        reference_col="ref_ti",
        adjusted_col="adjTI_RSD_TI",
        wind_speed_col="ref_ws",
        bin_col="bins",
        criteria_type="LV"
    )

    # Report
    overall = validation["overall"].iloc[0]
    print(f"\nResults:")
    print(f"  MRBE: {overall['MRBE_%']:.2f}%")
    print(f"  RRMSE: {overall['RRMSE_%']:.2f}%")
    print(f"  Pass: {overall['overall_pass']}")

    # Save
    results["adjusted_data"].to_csv(f"{output_dir}/adjusted_data.csv", index=False)
    validation["overall"].to_csv(f"{output_dir}/validation.csv", index=False)

    # Visualize
    plot_dnv_validation(
        validation_results=validation,
        adjusted_data=results["adjusted_data"],
        reference_col="ref_ti",
        adjusted_col="adjTI_RSD_TI",
        unadjusted_col="rsd_ti",
        wind_speed_col="ref_ws",
        bin_col="bins",
        method_name="SS-SF",
        output_dir=f"{output_dir}/plots"
    )

    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    return results, validation

if __name__ == "__main__":
    # This script can be run from anywhere
    analyze_site(
        data_path="/path/to/my/site_data.csv",
        config_path="/path/to/my/config.json",
        output_dir="/path/to/my/output"
    )
```

Run it from anywhere:
```bash
# From your project directory (not TACT directory)
cd ~/my_projects/wind_analysis/
python my_wind_analysis.py
```

---

## Directory Structure for External Projects

When using TACT as a module, organize your project like this:

```
my_wind_project/
├── data/
│   ├── site1_data.csv
│   ├── site2_data.csv
│   └── config.json
├── scripts/
│   ├── analyze_all_sites.py
│   ├── compare_methods.py
│   └── generate_report.py
├── output/
│   ├── site1/
│   └── site2/
├── requirements.txt          # Include: tact>=0.1.0
└── README.md
```

Your `requirements.txt`:
```txt
# If TACT is installed via pip (future)
tact>=0.1.0

# Or if using local installation
-e /path/to/TACT

# Plus your own dependencies
openpyxl
reportlab
# etc.
```

---

## Integration with Other Tools

### Jupyter Notebooks

```python
# In notebook cell
from tact import TACT
from tact.utils.load_data import load_data

data = load_data("my_data.csv")
# ... analyze
```

### Data Processing Pipelines

```python
# pipeline.py
import luigi
from tact import TACT

class TACTAdjustmentTask(luigi.Task):
    def run(self):
        tact = TACT()
        # ... run adjustment
        # ... save results
```

### Automated Workflows

```bash
#!/bin/bash
# analyze_all_sites.sh

for site in site1 site2 site3; do
    python -c "
from tact import TACT
# ... analyze $site
"
done
```

---

## Uninstallation

If you need to uninstall:

```bash
pip uninstall tact
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tact'"

**Solution**: Install the package
```bash
cd /path/to/TACT
pip install -e .
```

---

### Issue: "ImportError: cannot import name 'TACT'"

**Solution**: Check you're in the right virtual environment
```bash
which python  # Should point to your venv
pip list | grep tact  # Should show tact 0.1.0
```

---

### Issue: Changes to TACT code not reflected

**Solution**:
- If installed with `pip install .`: Reinstall with `pip install -e .`
- If installed with `-e`: Changes are automatic, restart Python interpreter

---

## Best Practices

### For Development
- Use `pip install -e .` (editable mode)
- Keep TACT in a separate directory
- Import specific functions you need

### For Production
- Use `pip install .` or specific version
- Pin version in requirements.txt
- Test thoroughly before deploying

### For Sharing Code
- Include TACT in requirements.txt
- Document installation steps
- Provide example scripts

---

## What Gets Installed

When you install TACT, you get:

- ✅ Core `tact` package (all modules)
- ✅ All adjustment methods (baseline, ss-sf, ssws, sswsstd)
- ✅ Validation framework (DNV RP-0661)
- ✅ Visualization tools
- ✅ Utility functions
- ✅ All dependencies (numpy, pandas, scikit-learn, scipy, matplotlib)

You do NOT get (these stay in TACT directory):
- ❌ Example data files
- ❌ Example scripts (main.py, compare_all_methods.py)
- ❌ Documentation files
- ❌ Legacy code

To use example data/scripts, reference them by path or copy to your project.

---

## Summary

| Installation Method | Use Case | Command |
|---------------------|----------|---------|
| **Development** | Modifying TACT code | `pip install -e .` |
| **Production** | Using TACT as-is | `pip install .` |
| **Standalone** | Quick testing in TACT dir | Just use requirements.txt |

**Recommendation**: For most users, use `pip install -e .` to install TACT as a package, then import it in your own scripts anywhere on your system.

---

## Next Steps

After installation:
1. Read [Getting Started Guide](getting-started.md)
2. Review [Data Import Guide](data-import-guide.md)
3. Try the examples in your own scripts
4. Build your analysis workflows

---

**Quick Start:**
```bash
cd /path/to/TACT
pip install -e .
cd ~/my_project
python -c "from tact import TACT; print('Ready!')"
```
