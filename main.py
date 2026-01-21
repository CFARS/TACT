from tact import TACT
from tact.adjustments.baseline import BaselineResults
from tact.adjustments.SSSF import SSSF
from tact.adjustments.SSWS import SSWS
from tact.adjustments.SSWSStd import SSWSStd
from tact.adjustments.bat import BATAdjustment
from tact.utils.setup_processors import setup_processors
from tact.utils.process_statistics import process_statistics
from tact.utils.save_results import save_results
from tact.utils.load_data import load_data
from tact.validation import validate_dnv_rp0661, validate_iea_task52_kpis
from tact.visualization import plot_dnv_validation, plot_iea_task52_kpis
import json


def main():
    
    # Configuration
    config = {
        "data_path": "tact/example/data/tact-test-data.csv",
        "config_path": "tact/example/config.json",
        "output_dir": "tact/example/output",
        # "method": "sswsstd",  # options: 'ss-sf', 'ssws', 'sswsstd', 'baseline', 'bat'
        "method": "bat",  # BAT method - requires Generic_BAT35_wk6.pkl in tact/assets/bat/
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

    # Load config to get column names
    with open(config["config_path"], 'r') as f:
        column_config = json.load(f)

    col_map = column_config["input_data_column_mapping"]
    ref_ti_col = col_map["reference"]["turbulence_intensity"]
    ref_ws_col = col_map["reference"]["wind_speed"]
    ref_sd_col = col_map["reference"]["standard_deviation"]
    rsd_ti_col = col_map["rsd"]["primary"]["turbulence_intensity"]
    rsd_sd_col = col_map["rsd"]["primary"]["standard_deviation"]

    # DNV RP-0661 Validation
    # Try with minimum TI filter to exclude unreliable low TI measurements
    min_ti_filter = None  # Filter out low TI - set to None to disable, or try 0.03, 0.05, 0.08

    print(f"\nRunning DNV RP-0661 validation with min_ti_threshold = {min_ti_filter}")
    validation_results = validate_dnv_rp0661(
        adjusted_data=results["adjusted_data"],
        reference_col=ref_ti_col,
        adjusted_col="adjTI_RSD_TI",
        wind_speed_col=ref_ws_col,
        bin_col="bins",
        use_test_only=True,
        criteria_type="LV",  # Load Verification criteria (most common)
        min_ti_threshold=min_ti_filter
    )

    # IEA Task 52 KPIs Validation
    print(f"\nRunning IEA Task 52 KPIs validation...")
    iea_validation_results = validate_iea_task52_kpis(
        data=results["adjusted_data"],
        ws_col=ref_ws_col,
        rsd_sd_col=rsd_sd_col,
        ref_sd_col=ref_sd_col,
        rsd_ti_col="adjTI_RSD_TI",  # Use adjusted TI for RSD
        ref_ti_col=ref_ti_col,
        m_values=[4, 9, 14],
        use_test_only=True
    )

    # Process statistics
    means = process_statistics(results["adjusted_data"], stats_processor)

    # Create plots subdirectory
    import os
    plots_dir = os.path.join(config["output_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate DNV validation plots
    plot_dnv_validation(
        validation_results=validation_results,
        adjusted_data=results["adjusted_data"],
        reference_col=ref_ti_col,
        unadjusted_col=rsd_ti_col,
        adjusted_col="adjTI_RSD_TI",
        wind_speed_col=ref_ws_col,
        bin_col="bins",
        criteria_type="LV",
        method_name=config["method"],
        output_dir=plots_dir
    )

    # Generate IEA Task 52 KPIs plots
    plot_iea_task52_kpis(
        validation_results=iea_validation_results,
        m_values=[4, 9, 14],
        title_prefix=f"{config['method']} - IEA Task 52 KPIs",
        save_path=os.path.join(plots_dir, f"{config['method']}_iea_task52_kpis.png")
    )

    # Save results
    save_results(
        means=means,
        adjusted_data=results["adjusted_data"],
        reg_results=results.get("reg_results"),  # Some methods (like BAT) don't return reg_results
        validation_results=validation_results,
        iea_validation_results=iea_validation_results,
        method=config["method"],
        output_dir=config["output_dir"]
    )
  
if __name__ == "__main__":
    main()
