# Site-Specific Simple + Filter (SS-SF)

The SS-SF adjustment method applies a site-specific correction to turbulence intensity measurements using a simple linear regression model with filtering.

## Implementation

::: tact.adjustments.SSSF
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - adjust
            - required_model_parameters
            - required_data_columns
            - perform_SS_SF_adjustment_ported 