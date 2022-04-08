import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments, empirical_stdAdjustment
from TACT.computation.ml import machine_learning_TI


def perform_SS_LTERRA_S_ML_adjustment(
    inputdata, all_trainX_cols, all_trainY_cols, all_testX_cols, all_testY_cols
):

    inputdata_test_result = pd.DataFrame()

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
        all_train = pd.DataFrame()
        all_train["y_train"] = inputdata_train["Ref_TI"].copy()
        all_train["x_train_TI"] = inputdata_train["RSD_TI"].copy()
        all_train["x_train_TKE"] = inputdata_train["RSD_LidarTKE"].copy()
        all_train["x_train_WS"] = inputdata_train["RSD_WS"].copy()
        all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
        all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
        all_test = pd.DataFrame()
        all_test["y_test"] = inputdata_test["Ref_TI"].copy()
        all_test["x_test_TI"] = inputdata_test["RSD_TI"].copy()
        all_test["x_test_TKE"] = inputdata_test["RSD_LidarTKE"].copy()
        all_test["x_test_WS"] = inputdata_test["RSD_WS"].copy()
        all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
        all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
        all_test["TI_test"] = inputdata_test["RSD_TI"].copy()
        all_test["RSD_SD"] = inputdata_test["RSD_SD"].copy()
        all_train = all_train.dropna()
        all_test = all_test.dropna()

        if len(all_train) < 5 and len(all_test) < 5:
            results = adj.post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
            m = np.NaN
            c = np.NaN
        else:
            m = np.NaN
            c = np.NaN
            TI_pred_RF = machine_learning_TI(
                all_train[all_trainX_cols],
                all_train[all_trainY_cols],
                all_test[all_testX_cols],
                all_test[all_testY_cols],
                "RF",
                all_test["TI_test"],
            )
            all_test["adjTI_RSD_TI"] = TI_pred_RF
            all_test["adjRepTI_RSD_RepTI"] = TI_pred_RF + 1.28 * all_test["RSD_SD"]
            all_test["Ref_TI"] = all_test["y_test"]
            inputdata_test_result = pd.merge(inputdata_test, all_test, how="left")
            results = adj.post_adjustment_stats(
                inputdata_test_result, results, "Ref_TI", "adjTI_RSD_TI"
            )

        if (
            "Ane_TI_Ht1" in inputdata.columns
            and "RSD_TI_Ht1" in inputdata.columns
            and "RSD_SD_Ht1" in inputdata.columns
        ):
            all_train = pd.DataFrame()
            all_train["y_train"] = inputdata_train["Ane_TI_Ht1"].copy()
            all_train["x_train_TI"] = inputdata_train["RSD_TI_Ht1"].copy()
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht1_LidarTKE"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht1"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht1"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht1"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht1_LidarTKE"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht1"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht1"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht1"].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()

            if len(all_train) < 5 and len(all_test) < 5:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[all_trainX_cols],
                    all_train[all_trainY_cols],
                    all_test[all_testX_cols],
                    all_test[all_testY_cols],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht1"] = TI_pred_RF
                all_test["Ane_TI_Ht1"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = adj.post_adjustment_stats(
                    inputdata_test_result, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )

        if (
            "Ane_TI_Ht2" in inputdata.columns
            and "RSD_TI_Ht2" in inputdata.columns
            and "RSD_SD_Ht2" in inputdata.columns
        ):
            all_train = pd.DataFrame()
            all_train["y_train"] = inputdata_train["Ane_TI_Ht2"].copy()
            all_train["x_train_TI"] = inputdata_train["RSD_TI_Ht2"].copy()
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht2_LidarTKE"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht2"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht2"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht2"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht2_LidarTKE"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht2"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht2"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht2"].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[all_trainX_cols],
                    all_train[all_trainY_cols],
                    all_test[all_testX_cols],
                    all_test[all_testY_cols],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht2"] = TI_pred_RF
                all_test["Ane_TI_Ht2"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = adj.post_adjustment_stats(
                    inputdata_test_result, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
        if (
            "Ane_TI_Ht3" in inputdata.columns
            and "RSD_TI_Ht3" in inputdata.columns
            and "RSD_SD_Ht3" in inputdata.columns
        ):
            all_train = pd.DataFrame()
            all_train["y_train"] = inputdata_train["Ane_TI_Ht3"].copy()
            all_train["x_train_TI"] = inputdata_train["RSD_TI_Ht3"].copy()
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht3_LidarTKE"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht3"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht3"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht3"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht3_LidarTKE"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht3"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht3"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht3"].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[all_trainX_cols],
                    all_train[all_trainY_cols],
                    all_test[all_testX_cols],
                    all_test[all_testY_cols],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht3"] = TI_pred_RF
                all_test["Ane_TI_Ht3"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = adj.post_adjustment_stats(
                    inputdata_test_result, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
        if (
            "Ane_TI_Ht4" in inputdata.columns
            and "RSD_TI_Ht4" in inputdata.columns
            and "RSD_Sd_Ht4" in inputdata.columns
        ):
            all_train = pd.DataFrame()
            all_train["y_train"] = inputdata_train["Ane_TI_Ht4"].copy()
            all_train["x_train_TI"] = inputdata_train["RSD_TI_Ht4"].copy()
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht4_LidarTKE"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht4"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht4"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht4"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht4_LidarTKE"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht4"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht4"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht4"].copy()
            all_train = all_train.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[all_trainX_cols],
                    all_train[all_trainY_cols],
                    all_test[all_testX_cols],
                    all_test[all_testY_cols],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht4"] = TI_pred_RF
                all_test["Ane_TI_Ht4"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = adj.post_adjustment_stats(
                    inputdata_test_result, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    if inputdata_test_result.empty:
        inputdata_test_result = inputdata_test

    return inputdata_test_result, results, m, c
