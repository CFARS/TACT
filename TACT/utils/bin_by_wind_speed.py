import pandas as pd
import json

def bin_by_wind_speed(inputdata, parameters):
    
    """
    Add wind speed bins to the input data
    
    4/18/2025 -CJP
    """
    
    with open(parameters["config_path"], 'r') as f:
            config = json.load(f)
            column_map = config['input_data_column_mapping']
    
    print(column_map)
    # Convert wind speed column to numeric type
    wind_speed_col = column_map["reference"]["wind_speed"]
    inputdata[wind_speed_col] = pd.to_numeric(inputdata[wind_speed_col], errors='coerce')
    
    inputdata["bins"] = inputdata[wind_speed_col].round(0)  # this acts as bin because the bin defination is between the two half integer values
    bins_p5_interval = pd.interval_range(
        start=0.25, end=20, freq=0.5, closed="left"
    )  # this is creating a interval range of .5 starting at .25
    out = pd.cut(x=inputdata[wind_speed_col], bins=bins_p5_interval)

    # create bin p5 category for each observation
    inputdata["bins_p5"] = out.apply(
        lambda x: x.mid
    )  # the middle of the interval is used as a catagorical label

    inputdata = inputdata[
        inputdata[column_map["reference"]["turbulence_intensity"]] != 0
    ]  # we can only analyze where the ref_TI is not 0

    return inputdata