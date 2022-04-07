import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments, empirical_stdAdjustment
from TACT.computation.post import post_adjustment_stats


def perform_SS_WS_adjustment(inputdata):
    """
    correct ws before computing TI
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
    _adjuster_SS_WS = Adjustments()

    if inputdata.empty or len(inputdata) < 2:
        results = post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
        if "Ane_WS_Ht1" in inputdata.columns and "RSD_WS_Ht1" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "ane_ti_ht1", "adjTI_rsd_ti_ht1"
            )
        if "ane_ws_ht2" in inputdata.columns and "RSD_WS_Ht2" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
            )
        if "Ane_WS_Ht3" in inputdata.columns and "RSD_WS_Ht3" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
            )
        if "Ane_WS_Ht4" in inputdata.columns and "RSD_WS_Ht4" in inputdata.columns:
            results = post_adjustment_stats(
                [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
            )
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        full = pd.DataFrame()
        full["Ref_WS"] = inputdata_test["Ref_WS"]
        full["RSD_WS"] = inputdata_test["RSD_WS"]
        full = full.dropna()
        if len(full) < 2:
            results = post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
            m = np.NaN
            c = np.NaN
        else:
            model = _adjuster_SS_WS.get_regression(
                inputdata_train["RSD_WS"], inputdata_train["Ref_WS"]
            )
            m = model[0]
            c = model[1]
            RSD_WS = inputdata_test["RSD_WS"]
            RSD_SD = inputdata_test["RSD_SD"]
            RSD_adjWS = (model[0] * RSD_WS) + model[1]
            inputdata_test["RSD_adjWS"] = RSD_adjWS
            RSD_TI = RSD_SD / inputdata_test["RSD_adjWS"]
            inputdata_test["adjTI_RSD_TI"] = RSD_TI
            results = post_adjustment_stats(
                inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
            )
        if (
            "Ane_WS_Ht1" in inputdata.columns
            and "RSD_WS_Ht1" in inputdata.columns
            and "RSD_SD_Ht1" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht1"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht1"]
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(
                    inputdata_train["RSD_WS_Ht1"], inputdata_train["Ane_WS_Ht1"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht1"]

                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                inputdata_test["RSD_adjWS_Ht1"] = RSD_adjWS
                RSD_TI = inputdata_test["RSD_SD_Ht1"] / inputdata_test["RSD_adjWS_Ht1"]
                inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                results = post_adjustment_stats(
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
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(
                    inputdata_train["RSD_WS_Ht2"], inputdata_train["Ane_WS_Ht2"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht2"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                inputdata_test["RSD_adjWS_Ht2"] = RSD_adjWS
                RSD_TI = inputdata_test["RSD_SD_Ht2"] / inputdata_test["RSD_adjWS_Ht2"]
                inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                results = post_adjustment_stats(
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
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(
                    inputdata_train["RSD_WS_Ht3"], inputdata_train["Ane_WS_Ht3"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht3"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                inputdata_test["RSD_adjWS_Ht3"] = RSD_adjWS
                RSD_TI = inputdata_test["RSD_SD_Ht3"] / inputdata_test["RSD_adjWS_Ht3"]
                inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                results = post_adjustment_stats(
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
            full = full.dropna()
            if len(full) < 2:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(
                    inputdata_train["RSD_WS_Ht4"], inputdata_train["Ane_WS_Ht4"]
                )
                RSD_WS = inputdata_test["RSD_WS_Ht4"]
                RSD_adjWS = (model[0] * RSD_WS) + model[1]
                inputdata_test["RSD_adjWS_Ht4"] = RSD_adjWS
                RSD_TI = inputdata_test["RSD_SD_Ht4"] / inputdata_test["RSD_adjWS_Ht4"]
                inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                results = post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    results["adjustment"] = ["SS-WS"] * len(results)
    results = results.drop(columns=["sensor", "height"])

    return inputdata_test, results, m, c
