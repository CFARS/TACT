import numpy as np
import pandas as pd


def perform_G_SFc_adjustment(inputdata):
    """
    simple filtered regression results from phase 2 averages used
    """

    results = pd.DataFrame(
        columns=[
            "sensor",
            "height",
            "adjustment",
            "m",
            "c",
            "rsquared",
            "difference",
            "mse",
            "rmse",
        ]
    )
    m = 0.7086
    c = 0.0225

    if inputdata.empty or len(inputdata) < 2:
        results = post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
        if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
            )
        if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
            )
        if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
            )
        if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
            )
        inputdata = False
    else:
        full = pd.DataFrame()
        full["Ref_TI"] = inputdata["Ref_TI"]
        full["RSD_TI"] = inputdata["RSD_TI"]
        full = full.dropna()
        if len(full) < 2:
            results = post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
        else:
            RSD_TI = inputdata["RSD_TI"].copy()
            RSD_TI = (float(m) * RSD_TI) + float(c)
            inputdata["adjTI_RSD_TI"] = RSD_TI
            results = post_adjustment_stats(
                inputdata, results, "Ref_TI", "adjTI_RSD_TI"
            )
        if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            full = pd.DataFrame()
            full["Ane_TI_Ht1"] = inputdata["Ane_TI_Ht1"]
            full["RSD_TI_Ht1"] = inputdata["RSD_TI_Ht1"]
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
            else:
                RSD_TI = inputdata["RSD_TI_Ht1"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata["adjTI_RSD_TI_Ht1"] = RSD_TI
                results = post_adjustment_stats(
                    inputdata, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
        if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            full = pd.DataFrame()
            full["Ane_TI_Ht2"] = inputdata["Ane_TI_Ht2"]
            full["RSD_TI_Ht2"] = inputdata["RSD_TI_Ht2"]
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
            else:
                RSD_TI = inputdata["RSD_TI_Ht2"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata["adjTI_RSD_TI_Ht2"] = RSD_TI
                results = post_adjustment_stats(
                    inputdata, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
        if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            full = pd.DataFrame()
            full["Ane_TI_Ht3"] = inputdata["Ane_TI_Ht3"]
            full["RSD_TI_Ht3"] = inputdata["RSD_TI_Ht3"]
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
            else:
                RSD_TI = inputdata["RSD_TI_Ht3"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata["adjTI_RSD_TI_Ht3"] = RSD_TI
                results = post_adjustment_stats(
                    inputdata, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
        if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            full = pd.DataFrame()
            full["Ane_TI_Ht4"] = inputdata["Ane_TI_Ht4"]
            full["RSD_TI_Ht4"] = inputdata["RSD_TI_Ht4"]
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
            else:
                RSD_TI = inputdata["RSD_TI_Ht4"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata["adjTI_RSD_TI_Ht4"] = RSD_TI
                results = post_adjustment_stats(
                    inputdata, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    results["adjustment"] = ["G-SFa"] * len(results)
    results = results.drop(columns=["sensor", "height"])
    return inputdata, results, m, c
