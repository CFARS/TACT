import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments, empirical_stdAdjustment


def perform_G_Sa_adjustment(inputdata, override, RSDtype):
    """simple filtered regression results from phase2 averages with simple regression from this data"""
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
    inputdata_train = inputdata[inputdata["split"] == True].copy()
    inputdata_test = inputdata[inputdata["split"] == False].copy()
    adj = Adjustments()

    if override:
        m_ph2 = override[0]
        c_ph2 = override[1]
    else:
        # set up which coefficients to use from phase 2 for testing
        if "Wind" in RSDtype["Selection"]:
            m_ph2 = 0.70695
            c_ph2 = 0.02289
        elif "ZX" in RSDtype["Selection"]:
            m_ph2 = 0.68647
            c_ph2 = 0.03901
        elif "Triton" in RSDtype["Selection"]:
            m_ph2 = 0.36532
            c_ph2 = 0.08662
        else:
            print("Warning: Did not apply regression results from phase 2")
            inputdata = pd.DataFrame()

    if inputdata.empty or len(inputdata) < 2:
        results = adj.post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
        if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
            )
        if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
            )
        if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
            )
        if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
            )
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        full = pd.DataFrame()
        full["Ref_TI"] = inputdata_test["Ref_TI"]
        full["RSD_TI"] = inputdata_test["RSD_TI"]
        full = full.dropna()
        if len(full) < 2:
            results = adj.post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
            m = np.NaN
            c = np.NaN
        else:
            model = adj.get_regression(
                inputdata_train["RSD_TI"], inputdata_train["Ref_TI"]
            )
            m = (model[0] + m_ph2) / 2
            c = (model[1] + c_ph2) / 2
            RSD_TI = inputdata_test["RSD_TI"].copy()
            RSD_TI = (m * RSD_TI) + c
            inputdata_test["adjTI_RSD_TI"] = RSD_TI
            inputdata_test["adjRepTI_RSD_RepTI"] = (
                RSD_TI + 1.28 * inputdata_test["RSD_SD"]
            )
            results = adj.post_adjustment_stats(
                inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
            )
        if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht1"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht1"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_TI"], inputdata_train["Ref_TI"]
                )
                m = (model[0] + m_ph2) / 2
                c = (model[1] + c_ph2) / 2
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI_Ht1"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht1"]
                )
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
        if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht2"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht2"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_TI_Ht2"], inputdata_train["Ane_TI_Ht2"]
                )
                m = (model[0] + m_ph2) / 2
                c = (model[1] + c_ph2) / 2
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI_Ht2"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht2"]
                )
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
        if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht3"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht3"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_TI_Ht3"], inputdata_train["Ane_TI_Ht3"]
                )
                m = (model[0] + m_ph2) / 2
                c = (model[1] + c_ph2) / 2
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI_Ht3"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht3"]
                )
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
        if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht4"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht4"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_TI_Ht4"], inputdata_train["Ane_TI_Ht4"]
                )
                m = (model[0] + m_ph2) / 2
                c = (model[1] + c_ph2) / 2
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI_Ht4"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht4"]
                )
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    results["adjustment"] = ["G-Sa"] * len(results)
    results = results.drop(columns=["sensor", "height"])

    return inputdata_test, results, m, c
