import numpy as np
import pandas as pd


def perform_SS_NN_adjustment(inputdata):

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
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        all_train = pd.DataFrame()
        all_train["y_train"] = inputdata_train["Ref_TI"].copy()
        all_train["x_train_TI"] = inputdata_train["RSD_TI"].copy()
        all_train["x_train_TKE"] = inputdata_train["RSD_LidarTKE_class"].copy()
        all_train["x_train_WS"] = inputdata_train["RSD_WS"].copy()
        all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
        all_train["x_train_Hour"] = inputdata_train["Hour"].copy()

        all_train["x_train_TEMP"] = inputdata_train["Temp"].copy()
        all_train["x_train_HUM"] = inputdata_train["Humidity"].copy()
        all_train["x_train_SD"] = inputdata_train["SD"].copy()
        all_train["x_train_Tshift1"] = inputdata_train["x_train_Tshift1"].copy()
        all_train["x_train_Tshift2"] = inputdata_train["x_train_Tshift3"].copy()
        all_train["x_train_Tshift3"] = inputdata_train["x_train_Tshift3"].copy()

        all_test = pd.DataFrame()
        all_test["y_test"] = inputdata_test["Ref_TI"].copy()
        all_test["x_test_TI"] = inputdata_test["RSD_TI"].copy()
        all_test["x_test_TKE"] = inputdata_test["RSD_LidarTKE_class"].copy()
        all_test["x_test_WS"] = inputdata_test["RSD_WS"].copy()
        all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
        all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
        all_test["TI_test"] = inputdata_test["RSD_TI"].copy()
        all_test["RSD_SD"] = inputdata_test["RSD_SD"].copy()
        all_train = all_train.dropna()
        all_test = all_test.dropna()
        if len(all_train) < 5 and len(all_test) < 5:
            results = post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
            m = np.NaN
            c = np.NaN
        else:
            m = np.NaN
            c = np.NaN
            TI_pred_RF = machine_learning_TI(
                all_train[
                    [
                        "x_train_TI",
                        "x_train_TKE",
                        "x_train_WS",
                        "x_train_DIR",
                        "x_train_Hour",
                    ]
                ],
                all_train["y_train"],
                all_test[
                    [
                        "x_test_TI",
                        "x_test_TKE",
                        "x_test_WS",
                        "x_test_DIR",
                        "x_test_Hour",
                    ]
                ],
                all_test["y_test"],
                "RF",
                all_test["TI_test"],
            )
            all_test["adjTI_RSD_TI"] = TI_pred_RF
            all_test["Ref_TI"] = all_test["y_test"]
            inputdata_test_result = pd.merge(inputdata_test, all_test, how="left")
            results = post_adjustment_stats(
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
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht1_LidarTKE_class"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht1"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht1"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht1"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht1_LidarTKE_class"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht1"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht1"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht1"].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[
                        [
                            "x_train_TI",
                            "x_train_TKE",
                            "x_train_WS",
                            "x_train_DIR",
                            "x_train_Hour",
                        ]
                    ],
                    all_train["y_train"],
                    all_test[
                        [
                            "x_test_TI",
                            "x_test_TKE",
                            "x_test_WS",
                            "x_test_DIR",
                            "x_test_Hour",
                        ]
                    ],
                    all_test["y_test"],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht1"] = TI_pred_RF
                all_test["Ane_TI_Ht1"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = post_adjustment_stats(
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
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht2_LidarTKE_class"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht2"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht2"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht2"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht2_LidarTKE_class"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht2"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht2"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht2"].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[
                        [
                            "x_train_TI",
                            "x_train_TKE",
                            "x_train_WS",
                            "x_train_DIR",
                            "x_train_Hour",
                        ]
                    ],
                    all_train["y_train"],
                    all_test[
                        [
                            "x_test_TI",
                            "x_test_TKE",
                            "x_test_WS",
                            "x_test_DIR",
                            "x_test_Hour",
                        ]
                    ],
                    all_test["y_test"],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht2"] = TI_pred_RF
                all_test["Ane_TI_Ht2"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = post_adjustment_stats(
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
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht3_LidarTKE_class"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht3"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht3"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht3"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht3_LidarTKE_class"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht3"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht3"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht3"].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[
                        [
                            "x_train_TI",
                            "x_train_TKE",
                            "x_train_WS",
                            "x_train_DIR",
                            "x_train_Hour",
                        ]
                    ],
                    all_train["y_train"],
                    all_test[
                        [
                            "x_test_TI",
                            "x_test_TKE",
                            "x_test_WS",
                            "x_test_DIR",
                            "x_test_Hour",
                        ]
                    ],
                    all_test["y_test"],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht3"] = TI_pred_RF
                all_test["Ane_TI_Ht3"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = post_adjustment_stats(
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
            all_train["x_train_TKE"] = inputdata_train["RSD_Ht4_LidarTKE_class"].copy()
            all_train["x_train_WS"] = inputdata_train["RSD_WS_Ht4"].copy()
            all_train["x_train_DIR"] = inputdata_train["RSD_Direction"].copy()
            all_train["x_train_Hour"] = inputdata_train["Hour"].copy()
            all_test = pd.DataFrame()
            all_test["y_test"] = inputdata_test["Ane_TI_Ht4"].copy()
            all_test["x_test_TI"] = inputdata_test["RSD_TI_Ht4"].copy()
            all_test["x_test_TKE"] = inputdata_test["RSD_Ht4_LidarTKE_class"].copy()
            all_test["x_test_WS"] = inputdata_test["RSD_WS_Ht4"].copy()
            all_test["x_test_DIR"] = inputdata_test["RSD_Direction"].copy()
            all_test["x_test_Hour"] = inputdata_test["Hour"].copy()
            all_test["TI_test"] = inputdata_test["RSD_TI_Ht4"].copy()
            all_test["RSD_SD"] = inputdata_test["RSD_SD_Ht4"].copy()
            all_train = all_train.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(
                    all_train[
                        [
                            "x_train_TI",
                            "x_train_TKE",
                            "x_train_WS",
                            "x_train_DIR",
                            "x_train_Hour",
                        ]
                    ],
                    all_train["y_train"],
                    all_test[
                        [
                            "x_test_TI",
                            "x_test_TKE",
                            "x_test_WS",
                            "x_test_DIR",
                            "x_test_Hour",
                        ]
                    ],
                    all_test["y_test"],
                    "RF",
                    all_test["TI_test"],
                )
                all_test["adjTI_RSD_TI_Ht4"] = TI_pred_RF
                all_test["Ane_TI_Ht4"] = all_test["y_test"]
                inputdata_test_result = pd.merge(
                    inputdata_test_result, all_test, how="left"
                )
                results = post_adjustment_stats(
                    inputdata_test_result, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    if inputdata_test_result.empty:
        inputdata_test_result = inputdata_test

    return inputdata_test_result, results, m, c
