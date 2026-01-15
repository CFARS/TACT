import pandas as pd
import numpy as np

def perform_SS_SF_adjustment(self, inputdata):

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
            results = self.post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
            if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
                results = self.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
            if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
                results = self.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
            if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
                results = self.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
            if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
                results = self.post_adjustment_stats(
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
                results = self.post_adjustment_stats(
                    [None],
                    results,
                    "Ref_TI",
                    "adjTI_RSD_TI",
                )
                m = np.NaN
                c = np.NaN
            else:
                model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
                m = model[0]
                c = model[1]
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (float(model[0]) * RSD_TI) + float(model[1])
                inputdata_test["adjTI_RSD_TI"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD"]
                )
                results = self.post_adjustment_stats(
                    inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
                )

            # if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            #     filtered_Ref_TI = inputdata_train["Ane_TI_Ht1"][
            #         inputdata_train["Ane_TI_Ht1"] < 0.3
            #     ]
            #     filtered_RSD_TI = inputdata_train["RSD_TI_Ht1"][
            #         inputdata_train["RSD_TI_Ht1"] < 0.3
            #     ]
            #     full = pd.DataFrame()
            #     full["filt_Ref_TI"] = filtered_Ref_TI
            #     full["filt_RSD_TI"] = filtered_RSD_TI
            #     full = full.dropna()
            #     if len(full) < 2:
            #         results = self.post_adjustment_stats(
            #             [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
            #         )
            #     else:
            #         model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
            #         RSD_TI = inputdata_test["RSD_TI_Ht1"].copy()
            #         RSD_TI = (model[0] * RSD_TI) + model[1]
            #         inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
            #         inputdata_test["adjRepTI_RSD_RepTI_Ht1"] = (
            #             RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht1"]
            #         )
            #         results = self.post_adjustment_stats(
            #             inputdata, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
            #         )

            # if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            #     filtered_Ref_TI = inputdata_train["Ane_TI_Ht2"][
            #         inputdata_train["Ane_TI_Ht2"] < 0.3
            #     ]
            #     filtered_RSD_TI = inputdata_train["RSD_TI_Ht2"][
            #         inputdata_train["RSD_TI_Ht2"] < 0.3
            #     ]
            #     full = pd.DataFrame()
            #     full["filt_Ref_TI"] = filtered_Ref_TI
            #     full["filt_RSD_TI"] = filtered_RSD_TI
            #     full = full.dropna()
            #     if len(full) < 2:
            #         results = self.post_adjustment_stats(
            #             [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
            #         )
            #     else:
            #         model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
            #         RSD_TI = inputdata_test["RSD_TI_Ht2"].copy()
            #         RSD_TI = (model[0] * RSD_TI) + model[1]
            #         inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
            #         inputdata_test["adjRepTI_RSD_RepTI_Ht2"] = (
            #             RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht2"]
            #         )
            #         results = self.post_adjustment_stats(
            #             inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
            #         )

            # if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            #     filtered_Ref_TI = inputdata_train["Ane_TI_Ht3"][
            #         inputdata_train["Ane_TI_Ht3"] < 0.3
            #     ]
            #     filtered_RSD_TI = inputdata_train["RSD_TI_Ht3"][
            #         inputdata_train["RSD_TI_Ht3"] < 0.3
            #     ]
            #     full = pd.DataFrame()
            #     full["filt_Ref_TI"] = filtered_Ref_TI
            #     full["filt_RSD_TI"] = filtered_RSD_TI
            #     full = full.dropna()

            #     if len(full) < 2:
            #         results = self.post_adjustment_stats(
            #             [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
            #         )
            #     else:
            #         model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
            #         RSD_TI = inputdata_test["RSD_TI_Ht3"].copy()
            #         RSD_TI = (model[0] * RSD_TI) + model[1]
            #         inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
            #         inputdata_test["adjRepTI_RSD_RepTI_Ht3"] = (
            #             RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht3"]
            #         )
            #         results = self.post_adjustment_stats(
            #             inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
            #         )

            # if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            #     filtered_Ref_TI = inputdata_train["Ane_TI_Ht4"][
            #         inputdata_train["Ane_TI_Ht4"] < 0.3
            #     ]
            #     filtered_RSD_TI = inputdata_train["RSD_TI_Ht4"][
            #         inputdata_train["RSD_TI_Ht4"] < 0.3
            #     ]
            #     full = pd.DataFrame()
            #     full["filt_Ref_TI"] = filtered_Ref_TI
            #     full["filt_RSD_TI"] = filtered_RSD_TI
            #     full = full.dropna()

            #     if len(full) < 2:
            #         results = self.post_adjustment_stats(
            #             [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
            #         )
            #     else:
            #         model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
            #         RSD_TI = inputdata_test["RSD_TI_Ht4"].copy()
            #         RSD_TI = (model[0] * RSD_TI) + model[1]
            #         inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
            #         inputdata_test["adjRepTI_RSD_RepTI_Ht4"] = (
            #             RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht4"]
            #         )
            #         results = self.post_adjustment_stats(
            #             inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
            #         )

        results["adjustment"] = ["SS-SF"] * len(results)
        results = results.drop(columns=["sensor", "height"])

        return inputdata_test, results, m, c