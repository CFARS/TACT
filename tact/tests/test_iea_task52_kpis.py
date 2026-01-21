"""
Test IEA Task 52 KPIs validation module
"""

import pandas as pd
import numpy as np
import pytest
from tact.validation.iea_task52_kpis import (
    validate_iea_task52_kpis,
    effective_sigma
)


class TestEffectiveSigma:
    """Test effective_sigma helper function"""

    def test_effective_sigma_constant_values(self):
        """Test effective_sigma with constant sigma values"""
        sigmas = np.array([1.0, 1.0, 1.0, 1.0])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        m = 4
        
        result = effective_sigma(sigmas, weights, m)
        assert abs(result - 1.0) < 1e-10, "Effective sigma should equal constant value"

    def test_effective_sigma_varying_values(self):
        """Test effective_sigma with varying values"""
        sigmas = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        m = 2
        
        result = effective_sigma(sigmas, weights, m)
        # For m=2: sqrt(0.25*(1^2 + 2^2 + 3^2 + 4^2)) = sqrt(0.25*30) = sqrt(7.5) ≈ 2.74
        expected = np.sqrt(0.25 * (1**2 + 2**2 + 3**2 + 4**2))
        assert abs(result - expected) < 1e-6

    def test_effective_sigma_filters_nan(self):
        """Test that effective_sigma filters out NaN values"""
        sigmas = np.array([1.0, np.nan, 2.0, 3.0])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        m = 4
        
        result = effective_sigma(sigmas, weights, m)
        # Should only use valid values: 1.0, 2.0, 3.0
        assert not np.isnan(result), "Should handle NaN values gracefully"

    def test_effective_sigma_filters_zero_weights(self):
        """Test that effective_sigma handles zero weights"""
        sigmas = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.5, 0.5])
        m = 4
        
        result = effective_sigma(sigmas, weights, m)
        # Should only use non-zero weights: 2.0, 3.0
        assert not np.isnan(result)


class TestValidateIEA52KPIs:
    """Test validate_iea_task52_kpis function"""

    def test_validate_iea_task52_kpis_structure(self):
        """Test that validation returns correct structure"""
        # Create synthetic test data
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(4, 16, n),
            "rsd_sd": np.random.uniform(0.5, 2.0, n),
            "ref_sd": np.random.uniform(0.5, 2.0, n),
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n),
            "split": [True] * 800 + [False] * 200
        })
        
        results = validate_iea_task52_kpis(data, use_test_only=True)
        
        # Check structure
        assert "overall" in results
        assert "by_bin" in results
        assert "metadata" in results
        assert isinstance(results["overall"], pd.DataFrame)
        assert isinstance(results["by_bin"], pd.DataFrame)
        assert isinstance(results["metadata"], dict)

    def test_validate_iea_task52_kpis_overall_columns(self):
        """Test that overall DataFrame has expected columns"""
        np.random.seed(42)
        n = 500
        
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(4, 16, n),
            "rsd_sd": np.random.uniform(0.5, 2.0, n),
            "ref_sd": np.random.uniform(0.5, 2.0, n),
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n)
        })
        
        results = validate_iea_task52_kpis(data, m_values=[4, 9])
        
        overall = results["overall"].iloc[0]
        
        # Check required columns exist
        assert "n_observations" in overall.index
        assert "di_ratio_m4" in overall.index
        assert "di_rel_error_m4" in overall.index
        assert "di_ratio_m9" in overall.index
        assert "di_rel_error_m9" in overall.index
        assert "i90_ratio" in overall.index
        assert "i90_rel_error" in overall.index
        assert "N_i90" in overall.index

    def test_validate_iea_task52_kpis_by_bin_columns(self):
        """Test that by_bin DataFrame has expected columns"""
        np.random.seed(42)
        n = 500
        
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(4, 16, n),
            "rsd_sd": np.random.uniform(0.5, 2.0, n),
            "ref_sd": np.random.uniform(0.5, 2.0, n),
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n)
        })
        
        results = validate_iea_task52_kpis(data, m_values=[4, 9])
        
        by_bin = results["by_bin"]
        
        if len(by_bin) > 0:
            # Check required columns exist
            assert "ws_bin_left" in by_bin.columns
            assert "ws_bin_right" in by_bin.columns
            assert "ws_bin_center" in by_bin.columns
            assert "N" in by_bin.columns
            assert "ieff_ratio_m4" in by_bin.columns
            assert "ieff_rel_error_m4" in by_bin.columns
            assert "ieff_ratio_m9" in by_bin.columns
            assert "ieff_rel_error_m9" in by_bin.columns

    def test_validate_iea_task52_kpis_ratio_calculation(self):
        """Test that ratio calculation is correct for constant bias"""
        # Create data where rsd_sd = 1.05 * ref_sd (5% bias)
        n = 1000
        ref_sd = np.random.uniform(0.5, 2.0, n)
        rsd_sd = ref_sd * 1.05
        
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(4, 16, n),
            "rsd_sd": rsd_sd,
            "ref_sd": ref_sd,
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n)
        })
        
        results = validate_iea_task52_kpis(data, m_values=[4])
        
        # DI ratio should be approximately 1.05
        di_ratio = results["overall"].iloc[0]["di_ratio_m4"]
        assert abs(di_ratio - 1.05) < 0.01, f"DI ratio should be ~1.05, got {di_ratio}"
        
        # Relative error should be approximately 0.05
        di_rel_error = results["overall"].iloc[0]["di_rel_error_m4"]
        assert abs(di_rel_error - 0.05) < 0.01, f"DI rel error should be ~0.05, got {di_rel_error}"

    def test_validate_iea_task52_kpis_uses_test_only(self):
        """Test that validation uses only test data when specified"""
        n = 100
        
        data = pd.DataFrame({
            "ref_ws": [10.0] * n,
            "rsd_sd": [1.0] * n,
            "ref_sd": [1.0] * n,
            "rsd_ti": [0.1] * n,
            "ref_ti": [0.1] * n,
            "split": [True] * 80 + [False] * 20
        })
        
        results = validate_iea_task52_kpis(data, use_test_only=True)
        
        # Should only use 20 test observations
        n_obs = results["overall"].iloc[0]["n_observations"]
        assert n_obs == 20, f"Should use 20 test observations, got {n_obs}"

    def test_validate_iea_task52_kpis_handles_missing_columns(self):
        """Test that validation raises error for missing columns"""
        data = pd.DataFrame({
            "ref_ws": [10.0] * 100,
            "rsd_sd": [1.0] * 100
            # Missing ref_sd, rsd_ti, ref_ti
        })
        
        with pytest.raises(KeyError):
            validate_iea_task52_kpis(data)

    def test_validate_iea_task52_kpis_handles_nan(self):
        """Test that validation handles NaN values gracefully"""
        n = 100
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(4, 16, n),
            "rsd_sd": np.random.uniform(0.5, 2.0, n),
            "ref_sd": np.random.uniform(0.5, 2.0, n),
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n)
        })
        
        # Add some NaN values
        data.loc[0:10, "rsd_sd"] = np.nan
        data.loc[20:30, "ref_sd"] = np.nan
        
        results = validate_iea_task52_kpis(data)
        
        # Should not crash and should reduce N appropriately
        n_obs = results["overall"].iloc[0]["n_observations"]
        assert n_obs < n, "Should filter out NaN values"
        assert n_obs > 0, "Should still have valid data"

    def test_validate_iea_task52_kpis_i90_calculation(self):
        """Test I90 calculation for V > 7 m/s"""
        n = 500
        # Create data with V > 7 m/s
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(7.5, 16, n),
            "rsd_sd": np.random.uniform(0.5, 2.0, n),
            "ref_sd": np.random.uniform(0.5, 2.0, n),
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n)
        })
        
        results = validate_iea_task52_kpis(data, v_i90_min=7.0)
        
        overall = results["overall"].iloc[0]
        
        # Check I90 values exist
        assert "i90_rsd" in overall.index
        assert "i90_ref" in overall.index
        assert "i90_ratio" in overall.index
        assert "i90_rel_error" in overall.index
        assert overall["N_i90"] > 0, "Should have data for I90 calculation"

    def test_validate_iea_task52_kpis_metadata(self):
        """Test that metadata contains expected information"""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            "ref_ws": np.random.uniform(4, 16, n),
            "rsd_sd": np.random.uniform(0.5, 2.0, n),
            "ref_sd": np.random.uniform(0.5, 2.0, n),
            "rsd_ti": np.random.uniform(0.05, 0.25, n),
            "ref_ti": np.random.uniform(0.05, 0.25, n)
        })
        
        results = validate_iea_task52_kpis(
            data,
            m_values=[4, 9, 14],
            v_i90_min=7.0,
            v_ieff_range=(8.0, 13.0)
        )
        
        metadata = results["metadata"]
        
        assert "m_values" in metadata
        assert metadata["m_values"] == [4, 9, 14]
        assert "v_i90_min" in metadata
        assert metadata["v_i90_min"] == 7.0
        assert "v_ieff_range" in metadata
        assert metadata["v_ieff_range"] == (8.0, 13.0)
        assert "column_mapping" in metadata
        assert "n_total_rows" in metadata
        assert "n_valid_rows" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
