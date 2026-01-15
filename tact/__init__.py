 # tact/__init__.py
from typing import Dict, Any
import pandas as pd
from .factory import AdjustmentFactory
from tact.core.registry import AdjustmentRegistry

class TACT:
    def __init__(self):
        self.factory = AdjustmentFactory()
    
    def adjust(self, 
              data: pd.DataFrame, 
              method: str, 
              parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            adjustment = self.factory.create_adjustment(method)
            return adjustment.adjust(data, parameters)
        except Exception as e:
            raise ValueError(f"Error performing {method} adjustment: {str(e)}")
    
    def list_available_methods(self) -> list:
        return list(AdjustmentRegistry._methods.keys())
    
    def get_method_parameters(self, method: str) -> Dict[str, Any]:
        adjustment = self.factory.create_adjustment(method)
        return adjustment.get_required_parameters()