# DNV RP-0661 Validation Plots - Interpretation Guide

This directory contains validation plots generated according to DNV Recommended Practice RP-0661 for remote sensing device (RSD) turbulence intensity measurements.

## Plot Types

### 1. MRBE by Wind Speed Bin (`*_mrbe_by_bin.png`)

**What it shows:**
- Mean Relative Bias Error (MRBE) for each wind speed bin
- Red dashed lines show DNV acceptance boundaries
- Green shaded area = acceptance zone

**Numbers above dots:**
- These are **observation counts** for each bin
- Example: "255" means 255 data points in that wind speed bin
- More observations = more statistically reliable

**How to interpret:**
- Points inside the green zone = PASS for that wind speed
- Points outside = FAIL
- Overall pass requires ALL bins to be within boundaries

**DNV LV Acceptance Criteria:**
- MRBE must be between -5% and +5%

---

### 2. RRMSE by Wind Speed Bin (`*_rrmse_by_bin.png`)

**What it shows:**
- Relative Root Mean Square Error (RRMSE) for each wind speed bin
- Red dashed line shows DNV acceptance limit
- Green shaded area = acceptance zone

**Numbers above dots:**
- Same as MRBE - observation counts per bin

**How to interpret:**
- Points below the red line = PASS
- Points above = FAIL
- RRMSE measures scatter/variance, not bias

**DNV LV Acceptance Criteria:**
- RRMSE must be ≤ 15%

---

### 3. TI Scatter Plot (`*_ti_scatter.png`)

**What it shows:**
- Each dot is one observation
- X-axis = Reference TI (truth from cup anemometer)
- Y-axis = Adjusted TI (corrected RSD measurement)

**Key features:**
- Red dashed line = perfect 1:1 agreement
- Green dotted lines = ±20% error bands
- Stats box shows total observations and overall bias

**How to interpret:**
- Points on the red line = perfect adjustment
- Points clustered around the line = good performance
- Points scattered or offset = poor performance
- Pattern reveals systematic errors (e.g., all points above = positive bias)

---

### 4. TI Comparison by Wind Speed (`*_ti_comparison.png`)

**What it shows:**
- Mean TI values for each wind speed bin
- Black circles = Reference TI (truth)
- Red squares = RSD TI (unadjusted)
- Blue triangles = Adjusted TI (after correction)

**Numbers above lines:**
- Observation counts per bin (same as MRBE/RRMSE plots)

**How to interpret:**
- **Goal**: Make blue triangles match black circles
- Gap between red and black = how wrong the RSD is initially
- Gap between blue and black = remaining error after adjustment
- Blue close to black = successful adjustment
- Blue still far from black = adjustment didn't work

---

## DNV RP-0661 Criteria Types

### Load Verification (LV) - Most Common
- **MRBE**: -5% to +5%
- **RRMSE**: ≤ 15%
- Used for turbine load calculations

### Site Suitability (SS) - Most Lenient
- **MRBE**:
  - High wind (≥7 m/s): -3% to +10%
  - Low wind (<7 m/s): -6% to +10%
- **RRMSE**:
  - High wind: ≤ 15%
  - Low wind: ≤ 30%
- Used for general site assessment

### Energy Production (EP) - Strictest MRBE, No RRMSE
- **MRBE**: -10% to +10%
- **RRMSE**: No limit
- Used for energy yield predictions

---

## Common Issues Revealed by Plots

### Issue 1: High Bias at Low Wind Speeds (Seen in Current Data)
- **MRBE plot**: High positive MRBE at low wind speed bins
- **Cause**: Linear regression intercept dominates low TI values
- **Solution**: Filter low TI, use different model, or bin-wise regression

### Issue 2: High Scatter
- **RRMSE plot**: All bins exceed 15% limit
- **Scatter plot**: Points widely dispersed around 1:1 line
- **Cause**: High variance in measurements or adjustment doesn't capture complexity
- **Solution**: More sophisticated model (ML methods)

### Issue 3: Systematic Bias
- **Comparison plot**: Blue line consistently above or below black line
- **Cause**: Model over/under-corrects
- **Solution**: Adjust regression, check data quality, try different method

---

## Quick Checklist for DNV Compliance

For **Load Verification (LV)** criteria:

- [ ] All MRBE values between -5% and +5% (green zone)
- [ ] All RRMSE values below 15% (green zone)
- [ ] Scatter plot shows points clustered near 1:1 line
- [ ] Comparison plot shows blue triangles close to black circles
- [ ] Sufficient observations per bin (typically >30 recommended)

If ANY checkbox fails, the adjustment does not meet DNV LV criteria.

---

## Current Results Summary

**SS-SF Method:**
- Overall MRBE: 38.8% ❌ (need ≤5%)
- Overall RRMSE: 114.7% ❌ (need ≤15%)
- **Status**: Does not meet DNV LV criteria
- **Issue**: Low TI values have very high errors (265% mean error for TI < 2%)

**SSWS Method:**
- Overall MRBE: 87.9% ❌ (need ≤5%)
- Overall RRMSE: 175.4% ❌ (need ≤15%)
- **Status**: Does not meet DNV LV criteria (worse than SS-SF)
- **Issue**: Wind speed adjustment amplifies errors

---

## Next Steps

1. **Filter low TI values** from validation (<0.05) - may improve metrics significantly
2. **Try SS criteria** instead of LV - more lenient thresholds
3. **Implement bin-wise regression** - different models for different TI/WS ranges
4. **Port advanced methods** - ML-based methods from legacy code
5. **Investigate data quality** - Check for sensor issues, calibration errors

---

Generated by TACT DNV RP-0661 Validation Module
