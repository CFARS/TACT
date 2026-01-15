import json

def calculate_ti(inputdata, parameters):
    
    """
    Calculate raw TI from input data
    
    Written 4/18/2025 -CJP
    """
    
    with open(parameters["config_path"], 'r') as f:
        config = json.load(f)
        column_map = config['input_data_column_mapping']
            
    TI_computed = inputdata[column_map["rsd"]["primary"]["standard_deviation"]]/inputdata[column_map["rsd"]["primary"]["wind_speed"]]
    Rep_TI_computed = TI_computed + 1.28 * inputdata[column_map["rsd"]["primary"]["standard_deviation"]]
    # inputdata = inputdata.rename(columns={'RSD_TI':'RSD_TI_instrument'})
    # inputdata = inputdata.rename(columns={'RSD_RepTI':'RSD_RepTI_instrument'})
    inputdata['RSD_TI'] = TI_computed
    inputdata['RSD_Rep_TI'] = Rep_TI_computed
    return inputdata