# Data Import Guide

This guide explains how to prepare and import your data into TACT.

## Table of Contents

1. [Data Requirements](#data-requirements)
2. [CSV Format](#csv-format)
3. [Data Preparation](#data-preparation)
4. [Configuration Mapping](#configuration-mapping)
5. [Loading Data](#loading-data)
6. [Troubleshooting](#troubleshooting)

---

## Data Requirements

### Minimum Requirements

TACT requires paired measurements from two sensors:

1. **Reference Sensor** (typically cup anemometer on meteorological tower)
   - Wind speed (m/s)
   - Wind speed standard deviation (m/s)
   - Turbulence intensity (decimal, e.g., 0.15 for 15%)

2. **RSD Sensor** (Remote Sensing Device - LiDAR or SoDAR)
   - Wind speed (m/s)
   - Wind speed standard deviation (m/s)
   - Turbulence intensity (decimal, e.g., 0.15 for 15%)

3. **Metadata**
   - Timestamp (any standard datetime format)
   - Optional: Quality flags, availability, CNR, etc.

### Data Volume Recommendations

| Dataset Size | Recommendation | Notes |
|--------------|----------------|-------|
| < 500 points | ❌ Too small | Results unreliable |
| 500-1000 points | ⚠️ Minimum | Use with caution |
| 1000-2000 points | ✅ Good | Suitable for most analyses |
| 2000+ points | ✅ Ideal | Best statistical power |

**Note**: These are after quality filtering. Aim for 2000+ valid paired observations in the 4-20 m/s wind speed range.

### Wind Speed Coverage

Ensure data coverage across operational range:

| Wind Speed Range | Coverage Needed | Why |
|------------------|-----------------|-----|
| < 4 m/s | Optional | Below typical cut-in |
| 4-12 m/s | ✅ **Critical** | Primary production range |
| 12-20 m/s | ✅ **Important** | High production |
| > 20 m/s | Optional | Above rated speed |

**Tip**: Check coverage with:
```python
import pandas as pd
data = pd.read_csv("your_data.csv")
print(data['ref_ws'].describe())
print(data.groupby(pd.cut(data['ref_ws'], bins=[0,4,8,12,16,20,25])).size())
```

---

## CSV Format

### Required Structure

TACT expects CSV files with these characteristics:

1. **Header row** with column names
2. **One row per observation** (timestamp)
3. **Numeric values** for wind data
4. **Consistent units** throughout

### Example CSV Structure

```csv
timestamp,ref_ws,ref_sd,ref_ti,rsd_ws,rsd_sd,rsd_ti
2024-01-01 00:00:00,8.5,1.2,0.141,8.3,1.3,0.157
2024-01-01 00:10:00,9.2,1.4,0.152,9.0,1.5,0.167
2024-01-01 00:20:00,7.8,1.1,0.141,7.6,1.2,0.158
2024-01-01 00:30:00,8.9,1.3,0.146,8.7,1.4,0.161
```

### Flexible Column Names

You can use **any column names** - just map them in the configuration file:

**Example 1: Short names**
```csv
time,r_ws,r_sd,r_ti,lidar_ws,lidar_sd,lidar_ti
```

**Example 2: Descriptive names**
```csv
DateTime,Tower_WindSpeed_58m_Avg,Tower_WindSpeed_58m_Std,Tower_TI_58m,ZX300_WindSpeed_60m_Avg,ZX300_WindSpeed_60m_Std,ZX300_TI_60m
```

**Example 3: Multi-height**
```csv
timestamp,ref_ws_60m,ref_sd_60m,ref_ti_60m,rsd_ws_60m,rsd_sd_60m,rsd_ti_60m,rsd_ws_80m,rsd_sd_80m,rsd_ti_80m
```

All work fine - just update your `config.json` accordingly!

### Units

| Measurement | Required Unit | Notes |
|-------------|---------------|-------|
| Wind Speed | m/s | Convert from mph, km/h, etc. |
| Standard Deviation | m/s | Same as wind speed |
| Turbulence Intensity | Decimal | 0.15, not 15% |
| Temperature | °C or °F | Optional, not used |
| Direction | Degrees | Optional, not used |

### Turbulence Intensity Format

**✅ Correct (decimal)**:
```csv
ref_ti,rsd_ti
0.15,0.16
0.12,0.14
```

**❌ Incorrect (percentage)**:
```csv
ref_ti,rsd_ti
15,16
12,14
```

If your data is in percentage format, convert it:
```python
import pandas as pd
data = pd.read_csv("data.csv")
data['ref_ti'] = data['ref_ti'] / 100
data['rsd_ti'] = data['rsd_ti'] / 100
data.to_csv("data_converted.csv", index=False)
```

---

## Data Preparation

### Step 1: Quality Filtering

Before importing to TACT, filter your data for quality:

```python
import pandas as pd

# Load raw data
data = pd.read_csv("raw_data.csv")

# Remove invalid wind speeds
data = data[(data['ref_ws'] > 0) & (data['rsd_ws'] > 0)]
data = data[(data['ref_ws'] < 40) & (data['rsd_ws'] < 40)]

# Remove invalid TI values
data = data[(data['ref_ti'] > 0) & (data['rsd_ti'] > 0)]
data = data[(data['ref_ti'] < 1) & (data['rsd_ti'] < 1)]

# Remove nulls
data = data.dropna(subset=['ref_ws', 'ref_sd', 'ref_ti', 'rsd_ws', 'rsd_sd', 'rsd_ti'])

# Optional: Filter by CNR (LiDAR signal quality)
if 'cnr' in data.columns:
    data = data[data['cnr'] > -25]  # Typical threshold

# Optional: Filter by availability
if 'availability' in data.columns:
    data = data[data['availability'] > 0.9]

print(f"Retained {len(data)} observations after filtering")

# Save filtered data
data.to_csv("filtered_data.csv", index=False)
```

### Step 2: Calculate TI if Missing

If you only have wind speed and standard deviation:

```python
# Calculate turbulence intensity
data['ref_ti'] = data['ref_sd'] / data['ref_ws']
data['rsd_ti'] = data['rsd_sd'] / data['rsd_ws']

# Handle division by zero
data['ref_ti'] = data['ref_ti'].replace([float('inf'), -float('inf')], float('nan'))
data['rsd_ti'] = data['rsd_ti'].replace([float('inf'), -float('inf')], float('nan'))
data = data.dropna(subset=['ref_ti', 'rsd_ti'])
```

### Step 3: Time Alignment

Ensure reference and RSD measurements are time-aligned:

```python
# Convert to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Round to nearest 10-minute period (common averaging period)
data['timestamp'] = data['timestamp'].dt.round('10min')

# Remove duplicates (keep first)
data = data.drop_duplicates(subset=['timestamp'], keep='first')

# Sort by time
data = data.sort_values('timestamp')
```

### Step 4: Data Summary

Before proceeding, validate your prepared data:

```python
def summarize_data(data):
    print("="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total observations: {len(data)}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print()

    print("Reference Statistics:")
    print(f"  WS: {data['ref_ws'].mean():.2f} ± {data['ref_ws'].std():.2f} m/s")
    print(f"  TI: {data['ref_ti'].mean():.3f} ± {data['ref_ti'].std():.3f}")
    print()

    print("RSD Statistics:")
    print(f"  WS: {data['rsd_ws'].mean():.2f} ± {data['rsd_ws'].std():.2f} m/s")
    print(f"  TI: {data['rsd_ti'].mean():.3f} ± {data['rsd_ti'].std():.3f}")
    print()

    print("Correlation:")
    print(f"  WS: {data[['ref_ws', 'rsd_ws']].corr().iloc[0,1]:.3f}")
    print(f"  TI: {data[['ref_ti', 'rsd_ti']].corr().iloc[0,1]:.3f}")
    print()

    print("Wind Speed Distribution:")
    ws_bins = pd.cut(data['ref_ws'], bins=[0,4,8,12,16,20,30])
    print(ws_bins.value_counts().sort_index())
    print("="*60)

summarize_data(data)
```

---

## Configuration Mapping

### Creating the Config File

Create `config.json` to map your CSV columns:

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

### Column Mapping Explanation

**Reference Section**: Tower anemometer measurements
```json
"reference": {
    "wind_speed": "YOUR_CSV_COLUMN_FOR_REF_WS",
    "wind_speed_std": "YOUR_CSV_COLUMN_FOR_REF_SD",
    "turbulence_intensity": "YOUR_CSV_COLUMN_FOR_REF_TI"
}
```

**RSD Section**: LiDAR measurements
```json
"rsd": {
    "primary": {
        "wind_speed": "YOUR_CSV_COLUMN_FOR_RSD_WS",
        "wind_speed_std": "YOUR_CSV_COLUMN_FOR_RSD_SD",
        "turbulence_intensity": "YOUR_CSV_COLUMN_FOR_RSD_TI"
    }
}
```

**Binning Section**: Controls wind speed grouping
```json
"binning_config": {
    "bin_size": 1.0,    // Width of each bin (m/s)
    "bin_min": 4.0,     // Minimum WS to analyze (m/s)
    "bin_max": 20.0     // Maximum WS to analyze (m/s)
}
```

### Example Configurations

**Example 1: IEC standard naming**
```json
{
    "input_data_column_mapping": {
        "reference": {
            "wind_speed": "Anem_WS_Avg",
            "wind_speed_std": "Anem_WS_Std",
            "turbulence_intensity": "Anem_TI"
        },
        "rsd": {
            "primary": {
                "wind_speed": "Lidar_WS_Avg",
                "wind_speed_std": "Lidar_WS_Std",
                "turbulence_intensity": "Lidar_TI"
            }
        }
    },
    "binning_config": {
        "bin_size": 0.5,
        "bin_min": 3.0,
        "bin_max": 25.0
    }
}
```

**Example 2: Multi-sensor naming**
```json
{
    "input_data_column_mapping": {
        "reference": {
            "wind_speed": "tower_353012_Ch1_Anem_55.50m_WSW_Avg_m/s",
            "wind_speed_std": "tower_353012_Ch1_Anem_55.50m_WSW_SD_m/s",
            "turbulence_intensity": "tower_TI_calc"
        },
        "rsd": {
            "primary": {
                "wind_speed": "ZX300_60m_WS_Avg",
                "wind_speed_std": "ZX300_60m_WS_Std",
                "turbulence_intensity": "ZX300_60m_TI"
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

---

## Loading Data

### Using TACT's Load Function

```python
from tact.utils.load_data import load_data

# Load CSV
data = load_data("your_data.csv")

# Verify loaded
print(f"Loaded {len(data)} rows")
print(f"Columns: {data.columns.tolist()}")
```

### Custom Loading

For more control:

```python
import pandas as pd

# Load with specific options
data = pd.read_csv(
    "your_data.csv",
    parse_dates=['timestamp'],  # Parse datetime column
    na_values=['', 'NA', 'NaN', '-999'],  # Define null values
    dtype={  # Specify data types
        'ref_ws': float,
        'ref_sd': float,
        'ref_ti': float,
        'rsd_ws': float,
        'rsd_sd': float,
        'rsd_ti': float
    }
)

# Additional filtering
data = data[data['ref_ws'].notna()]  # Remove nulls
data = data[data['ref_ws'] > 0]      # Remove zeros
```

### Processing Pipeline

Complete data loading and processing:

```python
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors

# 1. Load data
data = load_data("your_data.csv")
print(f"Loaded: {len(data)} observations")

# 2. Set up processors
binning_processor, ti_processor, stats_processor = setup_processors("config.json")

# 3. Apply binning (groups by wind speed)
data = binning_processor.process(data)
print(f"Binned into {data['bins'].nunique()} wind speed bins")

# 4. Calculate TI metrics (if not present)
data = ti_processor.process(data)

# 5. Verify processed data
print(f"Final dataset: {len(data)} rows, {len(data.columns)} columns")
print(f"Columns: {data.columns.tolist()}")
```

---

## Troubleshooting

### Issue: KeyError when loading config

**Error**: `KeyError: 'ref_ws'` or similar

**Cause**: Column name in config doesn't match CSV

**Solution**: Check column names match exactly
```python
import pandas as pd
data = pd.read_csv("your_data.csv")
print("Available columns:")
print(data.columns.tolist())

# Update config.json to match these names
```

---

### Issue: All TI values are very high

**Cause**: TI is in percentage format, not decimal

**Check**:
```python
print(data['ref_ti'].describe())
# If mean > 1.0, it's likely in percentage format
```

**Fix**:
```python
data['ref_ti'] = data['ref_ti'] / 100
data['rsd_ti'] = data['rsd_ti'] / 100
```

---

### Issue: Not enough data after binning

**Error**: `ValueError: Not enough observations in bin`

**Cause**: Sparse data across wind speed range

**Solution**: Widen bins or extend data collection
```json
{
    "binning_config": {
        "bin_size": 2.0,  // Increase from 1.0 to 2.0
        "bin_min": 4.0,
        "bin_max": 20.0
    }
}
```

---

### Issue: Poor correlation between RSD and reference

**Check**:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(data['ref_ws'], data['rsd_ws'], alpha=0.3)
plt.plot([0, 20], [0, 20], 'r--', label='1:1')
plt.xlabel('Reference WS (m/s)')
plt.ylabel('RSD WS (m/s)')
plt.title(f"Correlation: {data[['ref_ws','rsd_ws']].corr().iloc[0,1]:.3f}")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data['ref_ti'], data['rsd_ti'], alpha=0.3)
plt.plot([0, 0.5], [0, 0.5], 'r--', label='1:1')
plt.xlabel('Reference TI')
plt.ylabel('RSD TI')
plt.title(f"Correlation: {data[['ref_ti','rsd_ti']].corr().iloc[0,1]:.3f}")
plt.legend()

plt.tight_layout()
plt.savefig('correlation_check.png', dpi=150)
print("Saved correlation_check.png")
```

**If correlation is low** (less than 0.7):
- Check time synchronization
- Verify sensor heights match
- Check sensor calibration
- Review data quality filtering

---

### Issue: Missing standard deviation columns

**Error**: `KeyError: 'ref_sd'` or similar

**Cause**: CSV doesn't have standard deviation columns

**Solution**: Calculate from 10-minute raw data or use empirical model

If you have 1-Hz data, calculate 10-minute statistics:
```python
# Example: Calculate from high-frequency data
raw_data = pd.read_csv("1hz_data.csv", parse_dates=['timestamp'])
raw_data['time_bin'] = raw_data['timestamp'].dt.floor('10min')

# Group and calculate
stats = raw_data.groupby('time_bin').agg({
    'ref_ws': ['mean', 'std'],
    'rsd_ws': ['mean', 'std']
})

stats.columns = ['ref_ws', 'ref_sd', 'rsd_ws', 'rsd_sd']
stats['ref_ti'] = stats['ref_sd'] / stats['ref_ws']
stats['rsd_ti'] = stats['rsd_sd'] / stats['rsd_ws']

stats.to_csv("10min_stats.csv")
```

---

## Data Validation Checklist

Before running TACT, verify:

- [ ] CSV loads without errors
- [ ] All required columns present and mapped in config
- [ ] Units correct (m/s, decimal TI)
- [ ] No missing/null values in critical columns
- [ ] Reasonable value ranges (WS: 0-40 m/s, TI: 0-1)
- [ ] Sufficient data volume (>1000 observations)
- [ ] Wind speed coverage 4-20 m/s
- [ ] Good correlation between reference and RSD (R > 0.7)
- [ ] Time stamps are sequential and unique
- [ ] Config file matches CSV column names exactly

**Validation script**:
```python
def validate_data(data, config_path):
    import json

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    col_map = config['input_data_column_mapping']
    ref_ws = col_map['reference']['wind_speed']
    rsd_ws = col_map['rsd']['primary']['wind_speed']
    ref_ti = col_map['reference']['turbulence_intensity']
    rsd_ti = col_map['rsd']['primary']['turbulence_intensity']

    print("VALIDATION REPORT")
    print("="*60)

    # Check columns
    required = [ref_ws, rsd_ws, ref_ti, rsd_ti]
    missing = [c for c in required if c not in data.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    print(f"✅ All required columns present")

    # Check nulls
    null_counts = data[required].isnull().sum()
    if null_counts.any():
        print(f"⚠️  Null values detected:\n{null_counts[null_counts > 0]}")
    else:
        print(f"✅ No null values")

    # Check ranges
    if (data[ref_ws] < 0).any() or (data[rsd_ws] < 0).any():
        print("❌ Negative wind speeds detected")
        return False
    if (data[ref_ti] > 1).any() or (data[rsd_ti] > 1).any():
        print("⚠️  TI > 1.0 detected - check if using percentage format")
    print("✅ Value ranges OK")

    # Check volume
    n = len(data)
    if n < 500:
        print(f"❌ Insufficient data: {n} observations (need >500)")
        return False
    elif n < 1000:
        print(f"⚠️  Limited data: {n} observations (recommend >1000)")
    else:
        print(f"✅ Good data volume: {n} observations")

    # Check correlation
    corr = data[[ref_ws, rsd_ws]].corr().iloc[0,1]
    if corr < 0.7:
        print(f"⚠️  Low WS correlation: {corr:.3f} (target >0.7)")
    else:
        print(f"✅ Good WS correlation: {corr:.3f}")

    print("="*60)
    print("✅ VALIDATION PASSED - Ready for TACT")
    return True

# Run validation
validate_data(data, "config.json")
```

---

## Next Steps

Once your data is loaded and validated:

1. **[Run Your First Adjustment](getting-started.md#running-your-first-adjustment)**
2. **[Validate Results with DNV](getting-started.md#dnv-validation)**
3. **[Compare Methods](getting-started.md#comparing-methods)**

---

## Quick Reference

### Minimal Data Requirements
```python
required_columns = [
    'timestamp',      # DateTime
    'ref_ws',        # Reference wind speed (m/s)
    'ref_sd',        # Reference std dev (m/s)
    'ref_ti',        # Reference TI (decimal)
    'rsd_ws',        # RSD wind speed (m/s)
    'rsd_sd',        # RSD std dev (m/s)
    'rsd_ti'         # RSD TI (decimal)
]
```

### Quick Load Template
```python
from tact.utils.load_data import load_data
from tact.utils.setup_processors import setup_processors

data = load_data("data.csv")
bp, tp, sp = setup_processors("config.json")
data = bp.process(tp.process(data))
print(f"✅ Ready: {len(data)} observations")
```

---

**Need help?** See [Getting Started Guide](getting-started.md) or check [Troubleshooting](#troubleshooting).
