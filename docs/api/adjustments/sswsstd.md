# SSWSStd - Site-Specific Wind Speed + Standard Deviation Adjustment

**Location:** `tact/adjustments/SSWSStd.py`

## Overview

The SSWSStd method extends the SSWS approach by adjusting both wind speed AND standard deviation independently before recalculating turbulence intensity. This provides two-parameter correction for comprehensive measurement adjustment.

## Methodology

### Algorithm Steps

1. **Train Two Independent Regressions** (on training data):
   ```
   Reference_WS = m_ws * RSD_WS + c_ws
   Reference_SD = m_sd * RSD_SD + c_sd
   ```

2. **Apply Both Adjustments** (to test data):
   ```
   Adjusted_WS = m_ws * RSD_WS + c_ws
   Adjusted_SD = m_sd * RSD_SD + c_sd
   ```

3. **Recalculate Turbulence Intensity**:
   ```
   Adjusted_TI = Adjusted_SD / Adjusted_WS
   ```

4. **Calculate Representative TI**:
   ```
   Adjusted_RepTI = Adjusted_TI + 1.28 * (Adjusted_SD / Adjusted_WS)
   ```

### Key Characteristics

- **Dual adjustment**: Corrects both WS and SD separately
- **Four parameters**: Two slopes + two intercepts
- **Independent corrections**: WS and SD adjusted separately, then combined
- **Higher complexity**: More parameters than SSWS but potentially better accuracy
- **Error propagation**: Still subject to error amplification in division

## Usage

### Basic Example

```python
from tact import TACT

# Initialize
tact = TACT()

# Run SSWSStd adjustment
results = tact.adjust(
    data=your_data,
    method="sswsstd",
    parameters={
        "split": True,              # Enable train/test split
        "config_path": "config.json"  # Path to configuration
    }
)
```

### Complete Workflow

```python
from tact import TACT
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors
from tact.validation import validate_dnv_rp0661
from tact.visualization import plot_dnv_validation

# Load and process data
data = load_data("data.csv")
binning_proc, ti_proc, stats_proc = setup_processors("config.json")
data = binning_proc.process(data)
data = ti_proc.process(data)

# Run SSWSStd adjustment
tact = TACT()
results = tact.adjust(
    data=data,
    method="sswsstd",
    parameters={"split": True, "config_path": "config.json"}
)

# Validate
validation = validate_dnv_rp0661(
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    wind_speed_col="ref_ws",
    bin_col="bins",
    criteria_type="LV"
)

# Visualize
plot_dnv_validation(
    validation_results=validation,
    adjusted_data=results["adjusted_data"],
    reference_col="ref_ti",
    adjusted_col="adjTI_RSD_TI",
    unadjusted_col="rsd_ti",
    wind_speed_col="ref_ws",
    bin_col="bins",
    method_name="SSWSStd",
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

The config file must specify complete column mappings:

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
    }
}
```

## Required Data Columns

The input data must contain:

- Reference wind speed (e.g., `ref_ws`)
- Reference wind speed standard deviation (e.g., `ref_sd`)
- Reference turbulence intensity (e.g., `ref_ti`)
- RSD wind speed (e.g., `rsd_ws`)
- RSD wind speed standard deviation (e.g., `rsd_sd`)
- RSD turbulence intensity (e.g., `rsd_ti`)
- Train/test split indicator (e.g., `split`)
- Wind speed bins (e.g., `bins`)

## Output Format

### Returned Dictionary

```python
{
    "adjusted_data": pd.DataFrame,  # Data with adjusted columns
    "reg_results": pd.DataFrame,    # Regression statistics for both models
    "all_stats": pd.DataFrame       # Post-adjustment statistics
}
```

### Adjusted Data Columns

The method adds these columns to your data:

| Column | Description |
|--------|-------------|
| `RSD_adjWS` | Adjusted RSD wind speed |
| `RSD_adjSD` | Adjusted RSD standard deviation |
| `adjTI_RSD_TI` | Adjusted turbulence intensity |
| `adjRepTI_RSD_RepTI` | Adjusted representative TI |

### Regression Results

Contains two rows (one for WS model, one for SD model):

| Column | Description |
|--------|-------------|
| `sensor` | Sensor identifier (WS or SD) |
| `height` | Measurement height |
| `adjustment` | Adjustment method name |
| `m` | Regression slope |
| `c` | Regression intercept |
| `rsquared` | R² value |

## Performance Characteristics

### When SSWSStd Works Well

- Both WS and SD have strong linear correlations with reference
- Systematic bias in both measurements
- High R² (>0.9) for both regression models
- Consistent relationships across wind speed ranges

### When SSWSStd May Struggle

- **Dual error propagation**: Errors from both WS and SD adjustments combine
- **Low wind speeds**: Small denominators amplify errors
- **Overfitting risk**: Four parameters may overfit to training data
- **Non-linear relationships**: Linear models can't capture complex patterns
- **Poor correlation**: Low R² in either model degrades performance

### Performance on Example Dataset

Based on DNV RP-0661 LV criteria validation:

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| MRBE | +37.73% | ≤5% | ❌ |
| RRMSE | 121.61% | ≤15% | ❌ |
| N observations | 3,137 | - | - |

**Ranking**: 2nd best MRBE, 3rd best RRMSE (out of 4 methods)

**Analysis**: Better than SSWS but not as good as SS-SF. The dual adjustment adds complexity without improving performance enough to justify it. Error propagation through division still dominates.

## Comparison with Other Methods

### Performance Comparison

| Method | MRBE | RRMSE | Complexity | Rank |
|--------|------|-------|------------|------|
| **SS-SF** | 37.80% | 111.13% | Low | 🥇 1st |
| **SSWSStd** | 37.73% | 121.61% | High | 🥈 2nd |
| **SSWS** | 92.06% | 189.39% | Medium | 🥉 3rd |
| Baseline | 90.39% | 183.04% | None | - |

### Feature Comparison

| Aspect | SSWSStd | SS-SF | SSWS |
|--------|---------|-------|------|
| **Parameters** | 4 (m_ws, c_ws, m_sd, c_sd) | 2 (m, c) | 2 (m, c) |
| **Adjusts** | WS + SD | TI directly | WS only |
| **Error Propagation** | Yes (dual) | No | Yes |
| **Training Complexity** | Two models | One model | One model |
| **Best For** | Complex bias in both WS and SD | General purpose | WS-only bias |

## Implementation Details

### Class Definition

```python
from tact.core.base import AdjustmentMethod
from tact.core.registry import AdjustmentRegistry

@AdjustmentRegistry.register("sswsstd")
class SSWSStd(AdjustmentMethod):
    """Site-Specific Wind Speed + Standard Deviation adjustment."""

    def required_model_parameters(self) -> Dict[str, type]:
        return {"config_path": str, "split": bool}

    def required_data_columns(self) -> list:
        return [
            "reference.wind_speed",
            "reference.wind_speed_std",
            "reference.turbulence_intensity",
            "rsd.height_1.wind_speed",
            "rsd.height_1.wind_speed_std",
            "rsd.height_1.turbulence_intensity"
        ]

    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]):
        # Implementation...
```

### Mathematical Formulation

The method solves two independent least-squares problems:

**Wind Speed Model:**
```
minimize: Σ(Reference_WS - (m_ws * RSD_WS + c_ws))²
```

**Standard Deviation Model:**
```
minimize: Σ(Reference_SD - (m_sd * RSD_SD + c_sd))²
```

Then combines them:
```
TI = Adjusted_SD / Adjusted_WS
```

## Best Practices

### Pre-Adjustment Analysis

```python
import matplotlib.pyplot as plt

# Check correlations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# WS correlation
ax1.scatter(data['rsd_ws'], data['ref_ws'], alpha=0.5)
ax1.plot([0, 20], [0, 20], 'r--', label='1:1 line')
ax1.set_xlabel('RSD Wind Speed (m/s)')
ax1.set_ylabel('Reference Wind Speed (m/s)')
ax1.set_title(f'WS Correlation: {data[["rsd_ws", "ref_ws"]].corr().iloc[0,1]:.3f}')
ax1.legend()

# SD correlation
ax2.scatter(data['rsd_sd'], data['ref_sd'], alpha=0.5)
ax2.plot([0, 3], [0, 3], 'r--', label='1:1 line')
ax2.set_xlabel('RSD Std Dev (m/s)')
ax2.set_ylabel('Reference Std Dev (m/s)')
ax2.set_title(f'SD Correlation: {data[["rsd_sd", "ref_sd"]].corr().iloc[0,1]:.3f}')
ax2.legend()

plt.tight_layout()
plt.savefig('correlation_analysis.png')
```

### Examining Regression Quality

```python
# After running adjustment
reg_results = results["reg_results"]

print("Wind Speed Model:")
print(f"  Slope: {reg_results[reg_results['sensor']=='WS']['m'].values[0]:.3f}")
print(f"  Intercept: {reg_results[reg_results['sensor']=='WS']['c'].values[0]:.3f}")
print(f"  R²: {reg_results[reg_results['sensor']=='WS']['rsquared'].values[0]:.3f}")

print("\nStandard Deviation Model:")
print(f"  Slope: {reg_results[reg_results['sensor']=='SD']['m'].values[0]:.3f}")
print(f"  Intercept: {reg_results[reg_results['sensor']=='SD']['c'].values[0]:.3f}")
print(f"  R²: {reg_results[reg_results['sensor']=='SD']['rsquared'].values[0]:.3f}")

# Warning if R² is low
if any(reg_results['rsquared'] < 0.8):
    print("\n⚠️  Warning: Low R² detected. Consider alternative methods.")
```

### Validation and Comparison

```python
# Compare SSWSStd with other methods
from compare_all_methods import compare_all_methods

comparison = compare_all_methods()

# Filter to see where SSWSStd ranks
sswsstd_results = comparison[comparison['Method'] == 'SSWSSTD']
print(f"SSWSStd MRBE: {sswsstd_results['MRBE_%'].values[0]:.2f}%")
print(f"SSWSStd RRMSE: {sswsstd_results['RRMSE_%'].values[0]:.2f}%")

# Check if it's the best method for your data
best_method = comparison.loc[comparison['MRBE_%'].abs().idxmin(), 'Method']
print(f"\nBest method for this dataset: {best_method}")
```

## Troubleshooting

### High MRBE/RRMSE Despite Good R² Values

**Cause**: Error propagation when dividing adjusted SD by adjusted WS

**Solution**:
- Try SS-SF method (adjusts TI directly, no division)
- Consider filtering low wind speed data where division errors are largest
- Check for systematic bias patterns by wind speed bin

### Large Differences Between Training and Test Performance

**Cause**: Overfitting with 4 parameters, or non-representative split

**Solution**:
```python
# Verify train/test split
train = data[data['split'] == 1]
test = data[data['split'] == 0]

print(f"Train size: {len(train)}, Test size: {len(test)}")
print(f"Train WS range: {train['ref_ws'].min():.1f}-{train['ref_ws'].max():.1f} m/s")
print(f"Test WS range: {test['ref_ws'].min():.1f}-{test['ref_ws'].max():.1f} m/s")
```

### Negative Adjusted Values

**Cause**: Large negative intercepts with low WS or SD values

**Solution**:
```python
# Check for negative values
adjusted = results["adjusted_data"]
neg_ws = (adjusted['RSD_adjWS'] < 0).sum()
neg_sd = (adjusted['RSD_adjSD'] < 0).sum()

if neg_ws > 0 or neg_sd > 0:
    print(f"⚠️  Warning: {neg_ws} negative WS, {neg_sd} negative SD")
    print("Consider filtering low-value training data")
```

## When to Use SSWSStd

### ✅ Use SSWSStd When:
- Both WS and SD show systematic bias vs reference
- You have large dataset (>2000 points for training)
- R² > 0.9 for both WS and SD models
- You need independent control over WS and SD corrections

### ❌ Don't Use SSWSStd When:
- SS-SF performs better (check comparison results)
- You have limited data (less than 500 points)
- Either WS or SD has poor correlation (R² less than 0.8)
- Error propagation is a concern (low wind speeds common)

## See Also

- [SS-SF Method](sssf.md) - Simpler, better-performing alternative
- [SSWS Method](ssws.md) - Simpler version (WS adjustment only)
- [Baseline Method](baseline.md) - No adjustment reference
- [Method Comparison Tool](../../../compare_all_methods.py) - Compare all methods
- [DNV Validation](../validation/dnv_rp0661.md) - Validation framework

## References

- DNV GL: DNV-RP-0661 - Remote Sensing Measurement Verification
- IEC 61400-12-1 - Wind turbine power performance testing

## Source Code

Full implementation: [tact/adjustments/SSWSStd.py](../../../tact/adjustments/SSWSStd.py)
