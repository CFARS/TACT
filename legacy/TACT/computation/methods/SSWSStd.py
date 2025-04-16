import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments, empirical_stdAdjustment


def perform_SS_WS_Std_adjustment(inputdata):
    """
    correct ws and std before computing TI
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
    inputdata_train = inputdata[inputdata["split"] == True].copy()
    inputdata_test = inputdata[inputdata["split"] == False].copy()
    adj = Adjustments()

    if inputdata.empty or len(inputdata) < 2:
        results = adj.post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
        if "Ane_WS_Ht1" in inputdata.columns and "RSD_WS_Ht1" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
            )
        if "Ane_WS_Ht2" in inputdata.columns and "RSD_WS_Ht2" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
            )
        if "Ane_WS_Ht3" in inputdata.columns and "RSD_WS_Ht3" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
            )
        if "Ane_WS_Ht4" in inputdata.columns and "RSD_WS_Ht4" in inputdata.columns:
            results = adj.post_adjustment_stats(
                [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
            )
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        full = pd.DataFrame()
        full["Ref_WS"] = inputdata_test["Ref_WS"]
        full["RSD_WS"] = inputdata_test["RSD_WS"]
        full["Ref_SD"] = inputdata_test["Ref_SD"]
        full["RSD_Sd"] = inputdata_test["RSD_SD"]
        full = full.dropna()
        if len(full) < 2:
            results = adj.post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
            m = np.NaN
            c = np.NaN
        else:
            model = adj.get_regression(
                inputdata_train["RSD_WS"], inputdata_train["Ref_WS"]
            )
            model_std = adj.get_regression(
                inputdata_train["RSD_SD"], inputdata_train["Ref_SD"]
            )
            m = model[0]
            c = model[1]
            m_std = model_std[0]
            c_std = model_std[1]
            RSD_WS = inputdata_test["RSD_WS"]
            RSD_SD = inputdata_test["RSD_SD"]
            RSD_adjWS = (model[0] * RSD_WS) + model[1]
            RSD_adjSD = (model_std[0] * RSD_SD) + model_std[1]
            inputdata_test["RSD_adjWS"] = RSD_adjWS
            inputdata_test["RSD_adjSD"] = RSD_adjSD
            RSD_TI = inputdata_test["RSD_adjSD"] / inputdata_test["RSD_adjWS"]
            inputdata_test["adjTI_RSD_TI"] = RSD_TI
            results = adj.post_adjustment_stats(
                inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
            )
        if (
            "Ane_WS_Ht1" in inputdata.columns
            and "RSD_WS_Ht1" in inputdata.columns
            and "RSD_SD_Ht1" in inputdata.columns
            and "Ane_SD_Ht1" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht1"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht1"]
            full["Ref_SD"] = inputdata_test["Ane_SD_Ht1"]
            full["RSD_SD"] = inputdata_test["RSD_SD_Ht1"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_WS_Ht1"], inputdata_train["Ane_WS_Ht1"]
                )
                model_std = adj.get_regression(
                    inputdata_train["RSD_SD_Ht1"], inputdata_train["Ane_SD_Ht1"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht1"]
                RSD_SD = inputdata_test["RSD_SD_Ht1"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                RSD_adjSD = (model_std[0] * RSD_SD) + model_std[1]
                inputdata_test["RSD_adjWS_Ht1"] = RSD_adjWS
                inputdata_test["RSD_adjSD_Ht1"] = RSD_adjSD
                RSD_TI = (
                    inputdata_test["RSD_adjSD_Ht1"] / inputdata_test["RSD_adjWS_Ht1"]
                )
                inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
        if (
            "Ane_WS_Ht2" in inputdata.columns
            and "RSD_WS_Ht2" in inputdata.columns
            and "RSD_SD_Ht2" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht2"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht2"]
            full["Ref_SD"] = inputdata_test["Ane_SD_Ht2"]
            full["RSD_SD"] = inputdata_test["RSD_SD_Ht2"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_WS_Ht2"], inputdata_train["Ane_WS_Ht2"]
                )
                model_std = adj.get_regression(
                    inputdata_train["RSD_SD_Ht2"], inputdata_train["Ane_SD_Ht2"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht2"]
                RSD_SD = inputdata_test["RSD_SD_Ht2"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                RSD_adjSD = (model_std[0] * RSD_SD) + model_std[1]
                inputdata_test["RSD_adjWS_Ht2"] = RSD_adjWS
                inputdata_test["RSD_adjSD_Ht2"] = RSD_adjSD
                RSD_TI = (
                    inputdata_test["RSD_adjSD_Ht2"] / inputdata_test["RSD_adjWS_Ht2"]
                )
                inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
        if (
            "Ane_WS_Ht3" in inputdata.columns
            and "RSD_WS_Ht3" in inputdata.columns
            and "RSD_SD_Ht3" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht3"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht3"]
            full["Ref_SD"] = inputdata_test["Ane_SD_Ht3"]
            full["RSD_SD"] = inputdata_test["RSD_SD_Ht3"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_WS_Ht3"], inputdata_train["Ane_WS_Ht3"]
                )
                model_std = adj.get_regression(
                    inputdata_train["RSD_SD_Ht3"], inputdata_train["Ane_SD_Ht3"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht3"]
                RSD_SD = inputdata_test["RSD_SD_Ht3"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                RSD_adjSD = (model_std[0] * RSD_SD) + model_std[1]
                inputdata_test["RSD_adjWS_Ht3"] = RSD_adjWS
                inputdata_test["RSD_adjSD_Ht3"] = RSD_adjSD
                RSD_TI = (
                    inputdata_test["RSD_adjSD_Ht3"] / inputdata_test["RSD_adjWS_Ht3"]
                )
                inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
        if (
            "Ane_WS_Ht4" in inputdata.columns
            and "RSD_WS_Ht4" in inputdata.columns
            and "RSD_SD_Ht4" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht4"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht4"]
            full["Ref_SD"] = inputdata_test["Ane_SD_Ht4"]
            full["RSD_SD"] = inputdata_test["RSD_SD_Ht4"]
            full = full.dropna()
            if len(full) < 2:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = adj.get_regression(
                    inputdata_train["RSD_WS_Ht4"], inputdata_train["Ane_WS_Ht4"]
                )
                model_std = adj.get_regression(
                    inputdata_train["RSD_SD_Ht4"], inputdata_train["Ane_SD_Ht4"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht4"]
                RSD_SD = inputdata_test["RSD_SD_Ht4"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                RSD_adjSD = (model_std[0] * RSD_SD) + model_std[1]
                inputdata_test["RSD_adjWS_Ht4"] = RSD_adjWS
                inputdata_test["RSD_adjSD_Ht4"] = RSD_adjSD
                RSD_TI = (
                    inputdata_test["RSD_adjSD_Ht4"] / inputdata_test["RSD_adjWS_Ht4"]
                )
                inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    results["adjustment"] = ["SS-WS-Std"] * len(results)
    results = results.drop(columns=["sensor", "height"])

    return inputdata_test, results, m, c
