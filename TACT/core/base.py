# tact/core/base.py
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import json


@dataclass
class RegressionResults:
    metrics: List[Dict[str, Union[str, float]]]


@dataclass
class AdjustmentResult:
    adjusted_data: Optional[pd.DataFrame]
    metrics: RegressionResults


class TIAdjustmentClass(ABC):
    """Base class for all TI adjustment methods"""

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Return method requirements"""
        pass

    @abstractmethod
    def adjust(self, data: pd.DataFrame) -> AdjustmentResult:
        """Perform the adjustment"""
        pass


class AdjustmentMethod(ABC):
    """Base class for all adjustment methods"""

    @abstractmethod
    def required_model_parameters(self) -> Dict[str, Any]:
        """Returns a dictionary of required parameters and their types"""
        pass

    @abstractmethod
    def required_data_columns(self) -> Dict[str, List]:
        """Returns a dictionary of columns required columns in the data dataframe"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Verify all required parameters are included"""
        
        required = self.required_model_parameters()
        for param, param_type in required.items():
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")

            # For string type, we can use isinstance
            if param_type is str:
                if not isinstance(parameters[param], str):
                    raise ValueError(f"Parameter {param} must be a string")

            # For other types, you'll need to add specific checks
            # For now, we'll just check if the parameter exists
            elif parameters[param] is None:
                raise ValueError(f"Parameter {param} cannot be None")

        return True
    
    def validate_data(self, data: pd.DataFrame, config_path: str) -> None:
        """Validate that required columns are present in the data"""
        with open(config_path, "r") as f:
            config = json.load(f)
            mapping = config["input_data_column_mapping"]

        required_columns = self.required_data_columns()
        missing_cols = []

        for internal_path in required_columns:
            actual_col = mapping
            for key in internal_path.split("."):
                actual_col = actual_col[key]

            if actual_col not in data.columns:
                missing_cols.append(f"{internal_path} (mapped to {actual_col})")

        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    @abstractmethod
    def adjust(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Performs the adjustment and returns results"""
        pass
