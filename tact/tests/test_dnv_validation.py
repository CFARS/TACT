"""
Test DNV RP-0661 validation module
"""

import pandas as pd
import numpy as np
import pytest
from tact.validation.dnv_rp0661 import (
    calculate_mrbe,
    calculate_rrmse,
    check_acceptance_criteria,
    validate_dnv_rp0661,
    DNVAcceptanceCriteria
)


class TestMRBECalculation:
    """Test MRBE calculation"""

    def test_mrbe_zero_bias(self):
        """Test MRBE when measured equals reference (should be 0)"""
        reference = pd.Series([0.1, 0.2, 0.3, 0.4])
        measured = pd.Series([0.1, 0.2, 0.3, 0.4])
        mrbe = calculate_mrbe(reference, measured)
        assert abs(mrbe) < 1e-10, "MRBE should be 0 when measured equals reference"

    def test_mrbe_positive_bias(self):
        """Test MRBE with positive bias (measured > reference)"""
        reference = pd.Series([0.1, 0.2, 0.3, 0.4])
        measured = pd.Series([0.11, 0.22, 0.33, 0.44])  # 10% higher
        mrbe = calculate_mrbe(reference, measured)
        assert abs(mrbe - 10.0) < 0.1, "MRBE should be approximately 10%"

    def test_mrbe_negative_bias(self):
        """Test MRBE with negative bias (measured < reference)"""
        reference = pd.Series([0.1, 0.2, 0.3, 0.4])
        measured = pd.Series([0.09, 0.18, 0.27, 0.36])  # 10% lower
        mrbe = calculate_mrbe(reference, measured)
        assert abs(mrbe - (-10.0)) < 0.1, "MRBE should be approximately -10%"

    def test_mrbe_filters_invalid(self):
        """Test that MRBE filters out zeros and NaNs"""
        reference = pd.Series([0.1, 0.2, 0.0, 0.4, np.nan])
        measured = pd.Series([0.1, 0.2, 0.0, 0.4, 0.5])
        mrbe = calculate_mrbe(reference, measured)
        assert not np.isnan(mrbe), "MRBE should handle invalid values gracefully"


class TestRRMSECalculation:
    """Test RRMSE calculation"""

    def test_rrmse_zero_error(self):
        """Test RRMSE when measured equals reference (should be 0)"""
        reference = pd.Series([0.1, 0.2, 0.3, 0.4])
        measured = pd.Series([0.1, 0.2, 0.3, 0.4])
        rrmse = calculate_rrmse(reference, measured)
        assert abs(rrmse) < 1e-10, "RRMSE should be 0 when measured equals reference"

    def test_rrmse_with_error(self):
        """Test RRMSE with some error"""
        reference = pd.Series([0.1, 0.2, 0.3, 0.4])
        measured = pd.Series([0.11, 0.22, 0.33, 0.44])  # 10% higher
        rrmse = calculate_rrmse(reference, measured)
        assert rrmse > 0, "RRMSE should be positive"
        assert abs(rrmse - 10.0) < 0.1, "RRMSE should be approximately 10%"

    def test_rrmse_filters_invalid(self):
        """Test that RRMSE filters out zeros and NaNs"""
        reference = pd.Series([0.1, 0.2, 0.0, 0.4, np.nan])
        measured = pd.Series([0.1, 0.2, 0.0, 0.4, 0.5])
        rrmse = calculate_rrmse(reference, measured)
        assert not np.isnan(rrmse), "RRMSE should handle invalid values gracefully"


class TestAcceptanceCriteria:
    """Test DNV acceptance criteria checking"""

    def test_lv_criteria_pass(self):
        """Test Load Verification criteria - passing case"""
        result = check_acceptance_criteria(mrbe=3.0, rrmse=10.0, criteria_type="LV")
        assert result["pass_mrbe"] is True
        assert result["pass_rrmse"] is True
        assert result["overall_pass"] is True

    def test_lv_criteria_fail_mrbe(self):
        """Test Load Verification criteria - fail MRBE"""
        result = check_acceptance_criteria(mrbe=6.0, rrmse=10.0, criteria_type="LV")
        assert result["pass_mrbe"] is False
        assert result["overall_pass"] is False

    def test_lv_criteria_fail_rrmse(self):
        """Test Load Verification criteria - fail RRMSE"""
        result = check_acceptance_criteria(mrbe=3.0, rrmse=20.0, criteria_type="LV")
        assert result["pass_rrmse"] is False
        assert result["overall_pass"] is False

    def test_ss_criteria_high_wind(self):
        """Test Site Suitability criteria for high wind speeds (>= 7 m/s)"""
        result = check_acceptance_criteria(mrbe=5.0, rrmse=12.0, wind_speed=10.0, criteria_type="SS")
        assert result["pass_mrbe"] is True
        assert result["pass_rrmse"] is True
        assert result["overall_pass"] is True

    def test_ss_criteria_low_wind(self):
        """Test Site Suitability criteria for low wind speeds (< 7 m/s)"""
        result = check_acceptance_criteria(mrbe=5.0, rrmse=25.0, wind_speed=5.0, criteria_type="SS")
        assert result["pass_mrbe"] is True
        assert result["pass_rrmse"] is True
        assert result["overall_pass"] is True

    def test_ep_criteria(self):
        """Test Energy Production criteria (no RRMSE limit)"""
        result = check_acceptance_criteria(mrbe=8.0, rrmse=50.0, criteria_type="EP")
        assert result["pass_mrbe"] is True
        assert result["pass_rrmse"] is True  # EP has no RRMSE requirement
        assert result["overall_pass"] is True


class TestValidationIntegration:
    """Test full validation workflow"""

    def test_validate_dnv_rp0661(self):
        """Test complete validation with synthetic data"""
        # Create synthetic test data
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            "ref_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ws": np.random.uniform(4, 16, n),
            "split": [True] * 800 + [False] * 200,  # 80/20 train/test split
            "ws_bin": np.random.randint(4, 16, n)
        })

        # Add adjusted TI with small bias
        data["adj_ti"] = data["ref_ti"] * 1.02  # 2% positive bias

        # Run validation
        results = validate_dnv_rp0661(
            adjusted_data=data,
            reference_col="ref_ti",
            adjusted_col="adj_ti",
            wind_speed_col="ref_ws",
            use_test_only=True,
            criteria_type="LV"
        )

        # Check structure
        assert "overall" in results
        assert "by_bin" in results
        assert isinstance(results["overall"], pd.DataFrame)
        assert isinstance(results["by_bin"], pd.DataFrame)

        # Check overall results
        overall = results["overall"].iloc[0]
        assert "MRBE_%" in overall.index
        assert "RRMSE_%" in overall.index
        assert "overall_pass" in overall.index

        # Check by-bin results
        assert len(results["by_bin"]) > 0
        assert "wind_speed_bin" in results["by_bin"].columns
        assert "MRBE_%" in results["by_bin"].columns
        assert "RRMSE_%" in results["by_bin"].columns

    def test_validation_uses_test_only(self):
        """Test that validation uses only test data when specified"""
        n = 100

        data = pd.DataFrame({
            "ref_ti": [0.1] * n,
            "adj_ti": [0.1] * n,
            "ref_ws": [10.0] * n,
            "ws_bin": [10] * n,
            "split": [True] * 80 + [False] * 20
        })

        results = validate_dnv_rp0661(
            adjusted_data=data,
            reference_col="ref_ti",
            adjusted_col="adj_ti",
            wind_speed_col="ref_ws",
            use_test_only=True,
            criteria_type="LV"
        )

        # Should only use 20 test observations
        n_obs = results["overall"].iloc[0]["n_observations"]
        assert n_obs == 20, f"Should use 20 test observations, got {n_obs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
