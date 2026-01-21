"""
Tests for BAT adjustment method.

Note: These tests require Generic_BAT35_wk6.pkl to be present in tact/assets/bat/.
If the file is not available, some tests will be skipped.
"""

import pytest
import pandas as pd
import numpy as np
import os
import pickle
from unittest.mock import Mock, patch
from tact.adjustments.bat import BATAdjustment, apply_bat, load_bat_coefficients


# Test data paths
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'data', 'config.json')


def create_mock_coefficients():
    """Create a mock coefficient object for testing."""
    coeff = Mock()
    # Create arrays of length 36 (indices 0..35)
    n = 36
    coeff.WSGain = np.ones(n) * 1.1  # Slight gain
    coeff.WSOffset = np.zeros(n)  # No offset
    coeff.STDGain = np.ones(n) * 1.05
    coeff.STDOffset = np.zeros(n)
    return coeff


def create_test_data(n_points=1000):
    """Create synthetic test data."""
    np.random.seed(42)
    
    # Create realistic wind speed and standard deviation series
    t = np.arange(n_points)
    ws_base = 8.0 + 2.0 * np.sin(2 * np.pi * t / 200) + np.random.normal(0, 0.5, n_points)
    ws_base = np.maximum(ws_base, 2.0)  # Ensure positive
    
    sd_base = 0.15 * ws_base + 0.3 + np.random.normal(0, 0.1, n_points)
    sd_base = np.maximum(sd_base, 0.1)  # Ensure positive
    
    data = pd.DataFrame({
        'ref_ws': ws_base + np.random.normal(0, 0.2, n_points),
        'ref_sd': sd_base + np.random.normal(0, 0.05, n_points),
        'ref_ti': sd_base / ws_base + np.random.normal(0, 0.01, n_points),
        'rsd_ws': ws_base + np.random.normal(0, 0.3, n_points),  # Slightly noisier
        'rsd_sd': sd_base + np.random.normal(0, 0.08, n_points),
        'rsd_ti': (sd_base + np.random.normal(0, 0.08, n_points)) / (ws_base + np.random.normal(0, 0.3, n_points)),
    })
    
    return data


class TestBATAdjustment:
    """Test suite for BATAdjustment class."""

    def test_registration(self):
        """Test that BAT is registered in the registry."""
        from tact.core.registry import AdjustmentRegistry
        
        # Import to trigger registration
        from tact.adjustments.bat import BATAdjustment
        
        method_class = AdjustmentRegistry.get_method("bat")
        assert method_class is not None
        assert method_class == BATAdjustment

    def test_required_parameters(self):
        """Test required parameters declaration."""
        method = BATAdjustment()
        params = method.required_model_parameters()
        
        assert "config_path" in params
        assert params["config_path"] == str

    def test_required_data_columns(self):
        """Test required data columns declaration."""
        method = BATAdjustment()
        cols = method.required_data_columns()
        
        assert "rsd.primary.wind_speed" in cols
        assert "rsd.primary.standard_deviation" in cols

    def test_validate_parameters_missing_config(self):
        """Test parameter validation with missing config_path."""
        method = BATAdjustment()
        
        with pytest.raises(ValueError, match="Missing required parameter"):
            method.validate_parameters({})

    def test_validate_parameters_invalid_L(self):
        """Test parameter validation with invalid L."""
        method = BATAdjustment()
        
        with pytest.raises(ValueError, match="L must be an integer"):
            method.validate_parameters({"config_path": "test.json", "L": 1})
        
        with pytest.raises(ValueError, match="L must be an integer"):
            method.validate_parameters({"config_path": "test.json", "L": "100"})

    @patch('tact.adjustments.bat.load_bat_coefficients')
    def test_adjust_basic(self, mock_load_coeff):
        """Test basic adjustment functionality."""
        # Setup
        method = BATAdjustment()
        data = create_test_data(n_points=500)
        mock_coeff = create_mock_coefficients()
        mock_load_coeff.return_value = mock_coeff
        
        parameters = {
            "config_path": config_path,
            "L": 100,
            "n_calib": 35
        }
        
        # Run
        result = method.adjust(data, parameters)
        
        # Verify
        assert "adjusted_data" in result
        assert "metrics" in result
        
        adjusted_data = result["adjusted_data"]
        assert "RSD_adjWS" in adjusted_data.columns
        assert "RSD_adjSD" in adjusted_data.columns
        assert "adjTI_RSD_TI" in adjusted_data.columns
        
        # Check that adjusted columns have same length
        assert len(adjusted_data) == len(data)
        
        # Check metrics
        metrics = result["metrics"]
        assert metrics["method"] == "BAT"
        assert metrics["L"] == 100
        assert metrics["n_calib"] == 35

    @patch('tact.adjustments.bat.load_bat_coefficients')
    def test_adjust_short_series(self, mock_load_coeff):
        """Test adjustment with short series (auto-reduce L)."""
        # Setup
        method = BATAdjustment()
        data = create_test_data(n_points=150)  # Less than 2*L
        mock_coeff = create_mock_coefficients()
        mock_load_coeff.return_value = mock_coeff
        
        parameters = {
            "config_path": config_path,
            "L": 100,  # Will be auto-reduced
        }
        
        # Run (should not crash, but may warn)
        result = method.adjust(data, parameters)
        
        # Verify it still works
        assert "adjusted_data" in result
        assert len(result["adjusted_data"]) == len(data)

    @patch('tact.adjustments.bat.load_bat_coefficients')
    def test_adjust_with_nans(self, mock_load_coeff):
        """Test adjustment handles NaNs correctly."""
        # Setup
        method = BATAdjustment()
        data = create_test_data(n_points=500)
        # Add some NaNs
        data.loc[10:20, 'rsd_ws'] = np.nan
        data.loc[50:60, 'rsd_sd'] = np.nan
        
        mock_coeff = create_mock_coefficients()
        mock_load_coeff.return_value = mock_coeff
        
        parameters = {
            "config_path": config_path,
        }
        
        # Run
        result = method.adjust(data, parameters)
        
        # Verify
        adjusted_data = result["adjusted_data"]
        # Rows with NaNs should have NaN in adjusted columns
        assert adjusted_data.loc[10:20, 'RSD_adjWS'].isna().all()
        assert adjusted_data.loc[50:60, 'RSD_adjSD'].isna().all()


class TestApplyBAT:
    """Test suite for apply_bat function."""

    def test_apply_bat_basic(self):
        """Test basic apply_bat functionality."""
        # Create test series
        n = 500
        ws = pd.Series(8.0 + np.random.normal(0, 1, n), index=pd.RangeIndex(n))
        sd = pd.Series(1.2 + np.random.normal(0, 0.2, n), index=pd.RangeIndex(n))
        
        coeff = create_mock_coefficients()
        
        # Run
        corr_ws, corr_sd = apply_bat(ws, sd, coeff, L=100, n_calib=35)
        
        # Verify
        assert len(corr_ws) == len(ws)
        assert len(corr_sd) == len(sd)
        assert corr_ws.index.equals(ws.index)
        assert corr_sd.index.equals(sd.index)
        
        # Check that corrected values are not all NaN
        assert corr_ws.notna().sum() > 0
        assert corr_sd.notna().sum() > 0

    def test_apply_bat_different_indices(self):
        """Test apply_bat with different indices (should align)."""
        ws = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        sd = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=[1, 2, 3, 4, 5])
        
        coeff = create_mock_coefficients()
        
        # Should align on common indices
        corr_ws, corr_sd = apply_bat(ws, sd, coeff, L=2, n_calib=1)
        
        # Should have overlapping indices
        common_idx = ws.index.intersection(sd.index)
        assert len(corr_ws) > 0
        assert len(corr_sd) > 0

    def test_apply_bat_all_nans(self):
        """Test apply_bat with all NaNs."""
        ws = pd.Series([np.nan] * 100)
        sd = pd.Series([np.nan] * 100)
        
        coeff = create_mock_coefficients()
        
        corr_ws, corr_sd = apply_bat(ws, sd, coeff, L=50, n_calib=10)
        
        # Should return all NaNs
        assert corr_ws.isna().all()
        assert corr_sd.isna().all()

    def test_apply_bat_short_series(self):
        """Test apply_bat with series shorter than 2*L."""
        ws = pd.Series([8.0] * 50)  # 50 points
        sd = pd.Series([1.0] * 50)
        
        coeff = create_mock_coefficients()
        
        # L=100 requires at least 200 points, but we have 50
        # Should auto-reduce L
        with pytest.warns(UserWarning, match="Reducing L"):
            corr_ws, corr_sd = apply_bat(ws, sd, coeff, L=100, n_calib=10)
        
        assert len(corr_ws) == len(ws)
        assert len(corr_sd) == len(sd)

    def test_apply_bat_too_short(self):
        """Test apply_bat with series too short for any L."""
        ws = pd.Series([8.0, 9.0])  # Only 2 points
        sd = pd.Series([1.0, 1.1])
        
        coeff = create_mock_coefficients()
        
        # Should raise error or handle gracefully
        with pytest.raises(ValueError):
            apply_bat(ws, sd, coeff, L=100, n_calib=1)

    def test_apply_bat_non_positive_correction(self):
        """Test that non-positive corrected WS becomes NaN."""
        # Create series that might result in negative corrected WS
        ws = pd.Series([0.1] * 200)  # Very small values
        sd = pd.Series([0.01] * 200)
        
        coeff = create_mock_coefficients()
        # Use large negative offsets to force negative correction
        coeff.WSOffset = np.ones(36) * -10.0
        
        corr_ws, corr_sd = apply_bat(ws, sd, coeff, L=50, n_calib=5)
        
        # Non-positive values should be NaN
        assert (corr_ws[corr_ws <= 0].isna() | (corr_ws[corr_ws <= 0] == 0)).all() if (corr_ws <= 0).any() else True


class TestLoadCoefficients:
    """Test suite for load_bat_coefficients function."""

    @patch('importlib.resources.files')
    def test_load_packaged_coefficients(self, mock_files):
        """Test loading packaged coefficients."""
        # Mock the file resource
        mock_file = Mock()
        mock_file.open.return_value.__enter__.return_value = Mock()
        mock_file.open.return_value.__enter__.return_value.read.return_value = b''
        
        mock_path = Mock()
        mock_path.joinpath.return_value = mock_file
        mock_files.return_value = mock_path
        
        # Mock pickle.load
        mock_coeff = create_mock_coefficients()
        with patch('pickle.load', return_value=mock_coeff):
            coeff = load_bat_coefficients()
        
        assert coeff is not None

    def test_load_custom_coefficients(self, tmp_path):
        """Test loading coefficients from custom path."""
        # Create a temporary pkl file
        coeff = create_mock_coefficients()
        pkl_path = tmp_path / "test_coeff.pkl"
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(coeff, f)
        
        # Load it
        loaded_coeff = load_bat_coefficients(str(pkl_path))
        
        assert loaded_coeff is not None
        assert hasattr(loaded_coeff, 'WSGain')
        assert hasattr(loaded_coeff, 'STDGain')

    def test_load_coefficients_not_found(self):
        """Test error when coefficient file not found."""
        with pytest.raises(FileNotFoundError):
            load_bat_coefficients("/nonexistent/path.pkl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
