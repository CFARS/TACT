# Base Classes

Core abstract classes and interfaces that define the structure of TACT's adjustment methods.

## Adjustment Method

::: tact.core.base.AdjustmentMethod
    options:
        show_root_heading: true
        show_source: true
        members:
            - adjust
            - validate_parameters
            - validate_data
            - get_required_parameters

## Registry

::: tact.core.registry.AdjustmentRegistry
    options:
        show_root_heading: true
        show_source: true
        members:
            - register
            - get_method 