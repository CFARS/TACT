# tact/core/__init__.py

from .base import (
    TIAdjustmentClass,  # Base abstract class for all adjustment methods
    AdjustmentResult,    # Data class for method results
    RegressionResults,
)

from .registry import (
    MethodRegistry,      # Registry class for managing adjustment methods
)

# Version info
__version__ = "0.1.0"

# Expose a global registry instance that can be imported and used throughout the package
default_registry = MethodRegistry()

# Define public API
__all__ = [
    "TIAdjustmentClass",
    "AdjustmentResult",
    "RegressionResults",
    "MethodRegistry",
    "default_registry",
]