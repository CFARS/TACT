from tact import TACT
from tact.adjustments.baseline import BaselineResults
from tact.adjustments.SSSF import SSSF
from tact.utils.setup_processors import setup_processors
from tact.utils.process_statistics import process_statistics
from tact.utils.save_results import save_results
from tact.utils.load_data import load_data


def main():
    
    # Configuration
    config = {
        "data_path": "tact/example/data/tact-test-data.csv",
        "config_path": "tact/example/config.json",
        "output_dir": "tact/example/output",
        "method": "ss-sf",  # or 'baseline'
        "parameters": {"split": True}
    }
    
    # Initialize TACT
    tact = TACT()
    
    # Load and process data
    data = load_data(config["data_path"])
    
    # Setup processors
    binning_processor, ti_data_processor, stats_processor = setup_processors(config["config_path"])
    
    # Process data
    data = binning_processor.process(data)
    data = ti_data_processor.process(data)
    
    # Adjust data
    results = tact.adjust(
        data=data, 
        method=config["method"], 
        parameters={**config["parameters"], "config_path": config["config_path"]}
    )
    
    # Process statistics
    means = process_statistics(results["adjusted_data"], stats_processor)
    
    # Save results
    save_results(
        means=means,
        adjusted_data=results["adjusted_data"],
        reg_results=results["reg_results"],
        method=config["method"],
        output_dir=config["output_dir"]
    )
  
if __name__ == "__main__":
    main()
