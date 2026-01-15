# tact/factory.py
from tact.core.registry import AdjustmentRegistry

class AdjustmentFactory:
    @staticmethod
    def create_adjustment(method_name: str):
        method_class = AdjustmentRegistry.get_method(method_name)
        return method_class()