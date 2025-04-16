from tact.core import TIAdjustmentClass, RegressionResults
import pandas as pd
from typing import Dict, Any
from tact.utils.get_all_regressions import get_all_regressions
import json

class BaselineResults(TIAdjustmentClass):
    """Site Specific Simple Adjustment Method using regression"""
    
    def _validate_data(self, data: pd.DataFrame, config_path: str) -> None:
        """Validate that required columns are present in the data"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            mapping = config['input_data_column_mapping']
        
        requirements = self.get_requirements()
        missing_cols = []
        
        for internal_path in requirements["required_columns"]:
            actual_col = mapping
            for key in internal_path.split('.'):
                actual_col = actual_col[key]
            
            if actual_col not in data.columns:
                missing_cols.append(f"{internal_path} (mapped to {actual_col})")
                
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        if len(data) < requirements["min_samples"]:
            raise ValueError(f"Data must contain at least {requirements['min_samples']} samples")
    
    def adjust(self, data: pd.DataFrame, config_path: str) -> RegressionResults:
        
        self._validate_data(data, config_path)
        reg_results = get_all_regressions(data, config_path, title='Baseline Results')
        
        return RegressionResults(metrics=reg_results)
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "required_columns": [
                "reference.wind_speed",
                "rsd.height_1.wind_speed",
                "anemometer_2.primary.wind_speed",
                "reference.turbulence_intensity",
                "rsd.height_1.turbulence_intensity",
                "anemometer_2.primary.turbulence_intensity",
                "reference.standard_deviation",
                "rsd.height_1.standard_deviation",
                "anemometer_2.primary.standard_deviation"
            ],
            "min_samples": 2
        }