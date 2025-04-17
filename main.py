from tact import TACT
from tact.methods.baseline import BaselineResults
from tact.methods.SSSF import SSSF
import pandas as pd

def main():
    #create TACT instance
    tact = TACT()

    # Load data
    data = pd.read_csv('tact/example/data/tact-test-data.csv')
    
    method = 'ss-sf'
    # method = 'baseline'
    
    # Define parameters for baseline adjustment
    parameters = {
        'config_path': 'tact/example/config.json',
        'split': True
    }
    
    results = tact.adjust(
        data=data,
        method=method,
        parameters=parameters
    )

    # Access results
    adjusted_data = results['adjusted_data']
    reg_results = results['reg_results']
    
    print(adjusted_data)

    # Save results
    adjusted_data.to_csv(f'tact/example/output/{method}_adjusted_data.csv')
    reg_results.to_csv(f'tact/example/output/{method}_reg_results.csv')


if __name__ == "__main__":  
    main()