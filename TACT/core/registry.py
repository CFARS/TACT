from typing import Dict, Type, Callable, Optional
from .base import TIAdjustmentClass, AdjustmentMethod

# tact/core/registry.py
class MethodRegistry:
    """Registry for TI adjustment methods"""
    
    def __init__(self):
        self._methods: Dict[str, Type[TIAdjustmentClass]] = {}
    
    def register(self, name: str) -> Callable:
        def decorator(method_class: Type[TIAdjustmentClass]) -> Type[TIAdjustmentClass]:
            self._methods[name] = method_class
            return method_class
        return decorator
    
    def get_method(self, name: str) -> Optional[Type[TIAdjustmentClass]]:
        return self._methods.get(name)
    
    
    
class AdjustmentRegistry:
    _methods: Dict[str, Type[AdjustmentMethod]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(method_class: Type[AdjustmentMethod]):
            cls._methods[name] = method_class
            return method_class
        return decorator
    
    @classmethod
    def get_method(cls, name: str) -> Type[AdjustmentMethod]:
        if name not in cls._methods:
            raise ValueError(f"Unknown adjustment method: {name}")
        return cls._methods[name]