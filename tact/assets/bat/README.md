# BAT Coefficient Files

This directory contains coefficient files for the BAT (Bias Adjustment Technique) adjustment method.

## Required File

- **Generic_BAT35_wk6.pkl**: Pre-trained coefficients for BAT adjustment
  - Contains WSGain, WSOffset, STDGain, STDOffset arrays
  - Used by default when `coeff_path` parameter is not specified

## Installation

The coefficient file must be placed in this directory (`tact/assets/bat/`) for BAT to work.

If you're installing TACT as a package, the file will be included automatically via `setup.py` package_data configuration.

## Custom Coefficients

You can use custom coefficient files by specifying the `coeff_path` parameter:

```python
results = tact.adjust(
    data=data,
    method="bat",
    parameters={
        "config_path": "config.json",
        "coeff_path": "/path/to/custom_coefficients.pkl"
    }
)
```

## File Format

The pickle file should contain an object with the following attributes:

- `WSGain`: numpy array or list of wind speed gains (length ≥ n_calib+1)
- `WSOffset`: numpy array or list of wind speed offsets (length ≥ n_calib+1)
- `STDGain`: numpy array or list of standard deviation gains (length ≥ n_calib+1)
- `STDOffset`: numpy array or list of standard deviation offsets (length ≥ n_calib+1)

Where `n_calib` is the number of calibrated components (default 35).
