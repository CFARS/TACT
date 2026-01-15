import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments, post_adjustment_stats, empirical_stdAdjustment
from TACT.computation.calculations import get_regression


def perform_SS_SF_adjustment(inputdata):
        """
        Adjusts 10-minute averaged TI data with a linear slope and offset calibration method
        with a filter applied 
        derived from the test data 

        Parameters
        ----------
        inputdata : dataframe 

        Returns 
        -------
        method_results : dictionary
            inputdata_adj : dataframe
            results : dataframe
            m :  numeric
            c : numeric

        Notes
        -----
        Note: Representative TI computed with original RSD_SD

        References
        ----------
        To do: ADD REFERENCE
        
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

        if inputdata.empty or len(inputdata) < 2:
            results = post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
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
            filtered_Ref_TI = inputdata_train["Ref_TI"][inputdata_train["RSD_TI"] < 0.3]
            filtered_RSD_TI = inputdata_train["RSD_TI"][inputdata_train["RSD_TI"] < 0.3]
            full = pd.DataFrame()
            full["filt_Ref_TI"] = filtered_Ref_TI
            full["filt_RSD_TI"] = filtered_RSD_TI
            full = full.dropna()

            if len(full) < 2:
                results = post_adjustment_stats(
                    [None],
                    results,
                    "Ref_TI",
                    "adjTI_RSD_TI",
                )
                m = np.NaN
                c = np.NaN
            else:
                model = get_regression(filtered_RSD_TI, filtered_Ref_TI)
                m = model[0]
                c = model[1]
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (float(model[0]) * RSD_TI) + float(model[1])
                inputdata_test["adjTI_RSD_TI"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD"]
                )
                results = post_adjustment_stats(
                    inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
                )

            if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
                filtered_Ref_TI = inputdata_train["Ane_TI_Ht1"][
                    inputdata_train["Ane_TI_Ht1"] < 0.3
                ]
                filtered_RSD_TI = inputdata_train["RSD_TI_Ht1"][
                    inputdata_train["RSD_TI_Ht1"] < 0.3
                ]
                full = pd.DataFrame()
                full["filt_Ref_TI"] = filtered_Ref_TI
                full["filt_RSD_TI"] = filtered_RSD_TI
                full = full.dropna()
                if len(full) < 2:
                    results = post_adjustment_stats(
                        [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                    )
                else:
                    model = get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht1"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht1"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht1"]
                    )
                    results = post_adjustment_stats(
                        inputdata, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                    )

            if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
                filtered_Ref_TI = inputdata_train["Ane_TI_Ht2"][
                    inputdata_train["Ane_TI_Ht2"] < 0.3
                ]
                filtered_RSD_TI = inputdata_train["RSD_TI_Ht2"][
                    inputdata_train["RSD_TI_Ht2"] < 0.3
                ]
                full = pd.DataFrame()
                full["filt_Ref_TI"] = filtered_Ref_TI
                full["filt_RSD_TI"] = filtered_RSD_TI
                full = full.dropna()
                if len(full) < 2:
                    results = post_adjustment_stats(
                        [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                    )
                else:
                    model = get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht2"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht2"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht2"]
                    )
                    results = post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                    )

            if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
                filtered_Ref_TI = inputdata_train["Ane_TI_Ht3"][
                    inputdata_train["Ane_TI_Ht3"] < 0.3
                ]
                filtered_RSD_TI = inputdata_train["RSD_TI_Ht3"][
                    inputdata_train["RSD_TI_Ht3"] < 0.3
                ]
                full = pd.DataFrame()
                full["filt_Ref_TI"] = filtered_Ref_TI
                full["filt_RSD_TI"] = filtered_RSD_TI
                full = full.dropna()

                if len(full) < 2:
                    results = post_adjustment_stats(
                        [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                    )
                else:
                    model = get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht3"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht3"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht3"]
                    )
                    results = post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                    )

            if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
                filtered_Ref_TI = inputdata_train["Ane_TI_Ht4"][
                    inputdata_train["Ane_TI_Ht4"] < 0.3
                ]
                filtered_RSD_TI = inputdata_train["RSD_TI_Ht4"][
                    inputdata_train["RSD_TI_Ht4"] < 0.3
                ]
                full = pd.DataFrame()
                full["filt_Ref_TI"] = filtered_Ref_TI
                full["filt_RSD_TI"] = filtered_RSD_TI
                full = full.dropna()

                if len(full) < 2:
                    results = post_adjustment_stats(
                        [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )
                else:
                    model = get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht4"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht4"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht4"]
                    )
                    results = post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )

        results["adjustment"] = ["SS-SF"] * len(results)
        results = results.drop(columns=["sensor", "height"])

        method_results ={}
        method_results['inputdata_adj'] = inputdata_test
        method_results['results'] = results
        method_results['m'] = m
        method_results['c'] = c
        return method_results