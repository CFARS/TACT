import pandas as pd
import json
from tact.classes.Adjustments import Adjustments

def get_all_regressions(inputdata: pd.DataFrame, config_path: str, title: str = None) -> pd.DataFrame:
    # Load the column mapping from config
    with open(config_path, 'r') as f:
        config = json.load(f)
        mapping = config['input_data_column_mapping']
    
    # Define pairs using the JSON structure
    pairList = [
        ['reference.wind_speed', 'rsd.primary.wind_speed'],
        ['reference.wind_speed', 'anemometer_2.primary.wind_speed'],
        ['reference.turbulence_intensity', 'rsd.primary.turbulence_intensity'],
        ['reference.turbulence_intensity', 'anemometer_2.primary.turbulence_intensity'],
        ['reference.standard_deviation', 'rsd.primary.standard_deviation'],
        ['reference.standard_deviation', 'anemometer_2.primary.standard_deviation']
    ]
    
    lenFlag = len(inputdata) < 2
    columns = [title, 'm', 'c', 'rsquared', 'mean difference', 'mse', 'rmse']
    results = pd.DataFrame(columns=columns)
    
    for ref_path, target_path in pairList:
        # Get the actual column names from our mapping
        ref_col = mapping
        for key in ref_path.split('.'):
            ref_col = ref_col[key]
            
        target_col = mapping
        for key in target_path.split('.'):
            target_col = target_col[key]
        
        # Create a descriptive name for the result
        ref_name = ref_path.split('.')[-1]
        target_name = target_path.split('.')[0]
        res_name = f"{ref_name}_regression_{target_name}"
        
        # Use legacy adjustment class to calculate the regression
        _adjuster = Adjustments(inputdata)
        
        if target_col in inputdata.columns and not lenFlag:
            results_regr = [res_name] + _adjuster.get_regression(inputdata[ref_col], inputdata[target_col])
            # results_regr = [res_name, slope, intercept, r2, mean_diff, mse, rmse]
        else:
            results_regr = [res_name, 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']
                
        _results = pd.DataFrame(columns=columns, data=[results_regr])
        results = pd.concat([results, _results], ignore_index=True, axis=0, join='outer')
    
    return results
