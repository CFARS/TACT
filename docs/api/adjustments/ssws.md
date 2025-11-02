# SSWS - Site-Specific Wind Speed Adjustment

**Location:** `tact/adjustments/SSWS.py`

## Overview

The SSWS (Site-Specific Wind Speed) method adjusts turbulence intensity measurements by first correcting wind speed measurements, then recalculating turbulence intensity using the adjusted wind speed.

## Methodology

### Algorithm Steps

1. **Train Wind Speed Regression** (on training data):
   ```
   Reference_WS = m * RSD_WS + c
   ```

2. **Apply Wind Speed Adjustment** (to test data):
   ```
   Adjusted_WS = m * RSD_WS + c
   ```

3. **Recalculate Turbulence Intensity**:
   ```
   Adjusted_TI = RSD_SD / Adjusted_WS
   ```

4. **Calculate Representative TI**:
   ```
   Adjusted_RepTI = Adjusted_TI + 1.28 * (RSD_SD / Adjusted_WS)
   ```

### Key Characteristics

- **Two-stage adjustment**: Wind speed first, then TI calculation
- **Error propagation**: Errors in WS adjustment affect TI calculation
- **Linear regression**: Uses simple linear model for WS correction
- **Train/test split**: Builds model on training data, applies to test data

## Usage

### Basic Example

```python
from tact import TACT

# Initialize
tact = TACT()

# Run SSWS adjustment
results = tact.adjust(
    data=your_data,
    method="ssws",
    parameters={
        "split": True,              # Enable train/test split
        "config_path": "config.json"  # Path to configuration
    }
)
```

### With DNV Validation

```python
from tact import TACT
from tact.validation import validate_dnv_rp0661
from tact.visualization import plot_dnv_validation

# Run adjustment
tact = TACT()
results = tact.adjust(
    data=data,
    method="ssws",
    parameters={"split": True, "config_path": "config.json"}
)

# Validate against DNV RP-0661
validation = validate_dnv_rp0661(
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    wind_speed_col="ref_ws",
    bin_col="bins",
    criteria_type="LV"
)

# Generate plots
plot_dnv_validation(
    validation_results=validation,
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    unadjusted_col="rsd_ti",
    wind_speed_col="ref_ws",
    bin_col="bins",
    method_name="SSWS",
    output_dir="output/plots"
)
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` | Path to configuration JSON file |
| `split` | `bool` | Whether to split data into train/test sets |

### Configuration Requirements

The config file must specify column mappings:

```json
{
    "input_data_column_mapping": {
        "reference": {
            "wind_speed": "ref_ws",
            "wind_speed_std": "ref_sd"
        },
        "rsd": {
            "primary": {
                "wind_speed": "rsd_ws",
                "wind_speed_std": "rsd_sd"
            }
        }
    }
}
```

## Required Data Columns

The input data must contain:

- Reference wind speed (e.g., `ref_ws`)
- Reference wind speed standard deviation (e.g., `ref_sd`)
- RSD wind speed (e.g., `rsd_ws`)
- RSD wind speed standard deviation (e.g., `rsd_sd`)
- Train/test split indicator (e.g., `split`)
- Wind speed bins (e.g., `bins`)

## Output Format

### Returned Dictionary

```python
{
    "adjusted_data": pd.DataFrame,  # Data with adjusted columns
    "reg_results": pd.DataFrame,    # Regression statistics
    "all_stats": pd.DataFrame       # Post-adjustment statistics
}
```

### Adjusted Data Columns

The method adds these columns to your data:

| Column | Description |
|--------|-------------|
| `RSD_adjWS` | Adjusted RSD wind speed |
| `adjTI_RSD_TI` | Adjusted turbulence intensity |
| `adjRepTI_RSD_RepTI` | Adjusted representative TI |

### Regression Results

| Column | Description |
|--------|-------------|
| `sensor` | Sensor identifier |
| `height` | Measurement height |
| `adjustment` | Adjustment method name |
| `m` | Regression slope |
| `c` | Regression intercept |
| `rsquared` | R² value |

## Performance Characteristics

### When SSWS Works Well

- Strong linear relationship between RSD and reference wind speed
- Low noise in wind speed measurements
- Consistent wind speed bias across all speeds

### When SSWS May Struggle

- **Error propagation**: WS errors amplify in TI calculation
- **Low wind speeds**: Division by small values creates large relative errors
- **Non-linear relationships**: Linear regression can't capture complex patterns
- **High scatter**: R² < 0.8 indicates poor model fit

### Performance on Example Dataset

Based on DNV RP-0661 LV criteria validation:

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| MRBE | +94.77% | ≤5% | ❌ |
| RRMSE | 188.03% | ≤15% | ❌ |
| N observations | 3,054 | - | - |

**Note**: SSWS performs worse than baseline (90.39% MRBE) on the example dataset due to error propagation. See [Method Comparison](../../../METHOD_COMPARISON_RESULTS.md) for details.

## Comparison with Other Methods

| Aspect | SSWS | SS-SF | SSWSStd |
|--------|------|-------|---------|
| **Complexity** | Medium | Low | High |
| **Parameters** | 2 (m, c for WS) | 2 (m, c for TI) | 4 (m, c for WS and SD) |
| **Error Propagation** | Yes | No | Yes |
| **Performance** | Poor | Best | Medium |
| **Use Case** | WS correction needed | General purpose | WS+SD correction |

## Implementation Details

### Class Definition

```python
from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry

@AdjustmentRegistry.register("ssws")
class SSWS(AdjustmentMethod):
    """Site-Specific Wind Speed adjustment method."""

    def required_model_parameters(self) -> Dict[str, type]:
        return {"config_path": str, "split": bool}

    def required_data_columns(self) -> list:
        return [
            "reference.wind_speed",
            "reference.wind_speed_std",
            "rsd.height_1.wind_speed",
            "rsd.height_1.wind_speed_std"
        ]

    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]):
        # Implementation...
```

### Key Functions Used

- `get_regression()` - Performs linear regression
- `post_adjustment_stats()` - Calculates statistics
- Column mapping from config file

## Best Practices

### Pre-Flight Checks

```python
# Check data quality before adjustment
print(f"Data points: {len(data)}")
print(f"WS correlation: {data[['ref_ws', 'rsd_ws']].corr().iloc[0,1]:.3f}")
print(f"Missing values: {data[['ref_ws', 'rsd_ws']].isnull().sum()}")
```

### Post-Adjustment Validation

```python
# Always validate results
from tact.validation import validate_dnv_rp0661

validation = validate_dnv_rp0661(
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    wind_speed_col="ref_ws",
    bin_col="bins",
    criteria_type="LV"
)

# Check if passed
if validation["overall"]["overall_pass"].iloc[0]:
    print("✅ Passed DNV LV criteria")
else:
    print("❌ Failed DNV LV criteria")
    print(f"MRBE: {validation['overall']['MRBE_%'].iloc[0]:.2f}%")
    print(f"RRMSE: {validation['overall']['RRMSE_%'].iloc[0]:.2f}%")
```

### Comparing with Other Methods

```python
# Run comparison
from compare_all_methods import compare_all_methods

comparison = compare_all_methods()
print(comparison)
```

## Troubleshooting

### Common Issues

**Issue**: High MRBE/RRMSE despite good R²

**Cause**: Error propagation from WS adjustment to TI calculation

**Solution**: Try SS-SF method instead (adjusts TI directly)

---

**Issue**: Negative adjusted wind speeds

**Cause**: Large negative intercept with low wind speeds

**Solution**: Check regression intercept, consider filtering low WS data

---

**Issue**: Very different train vs test performance

**Cause**: Overfitting or non-representative split

**Solution**: Verify train/test split is random and representative

## See Also

- [SS-SF Method](sssf.md) - Simpler method with better performance
- [SSWSStd Method](sswsstd.md) - Extended version with SD adjustment
- [Baseline Method](baseline.md) - No adjustment reference
- [DNV Validation](../validation/dnv_rp0661.md) - Validation framework
- [Method Comparison Results](../../../METHOD_COMPARISON_RESULTS.md)

## References

- DNV GL: DNV-RP-0661 - Remote Sensing Measurement Verification
- IEC 61400-12-1 - Wind turbine power performance testing

## Source Code

Full implementation: [tact/adjustments/SSWS.py](../../../tact/adjustments/SSWS.py)
