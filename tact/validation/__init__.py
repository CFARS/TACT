"""Validation module for TACT adjustments."""

from .dnv_rp0661 import validate_dnv_rp0661
from .iea_task52_kpis import validate_iea_task52_kpis

__all__ = ["validate_dnv_rp0661", "validate_iea_task52_kpis"]
