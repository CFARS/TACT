from tact import TACT
from tact.methods.baseline import BaselineResults
from tact.methods.SSSF import SSSF
import pandas as pd

def main():
    #create TACT instance
    tact = TACT()

    # Load data
    data = pd.read_csv('tact/example/data/tact-test-data.csv')
    
    # Define parameters for baseline adjustment
    parameters = {
        'config_path': 'tact/example/config.json',
        'split': True
    }

    # Perform baseline adjustment
    results = tact.adjust(
        data=data,
        method='baseline',
        parameters=parameters
    )
    
    # results = tact.adjust(
    #     data=data,
    #     method='ss-sf',
    #     parameters=parameters
    # )

    # Access results
    adjusted_data = results['adjusted_data']
    metrics = results['metrics']

    # Save results
    adjusted_data.to_csv('adjusted_results.csv')
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()