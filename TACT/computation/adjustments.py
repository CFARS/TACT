try:
    from TACT import logger
except ImportError:
    pass
import pandas as pd
import sys
import matplotlib.pyplot as plt

plt.ioff()  # setting to non-interactive
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class Adjustments:

    """
    document parameters
    """

    def __init__(self, raw_data="", adjustments_list="", baseResultsLists=""):
        self.raw_data = raw_data
        self.adjusted_data = pd.DataFrame()
        self.results_stats = (
            []
        )  # make this a dictionary of results with adjustment_list items as keys

    def get_regression(self, x, y):
        """
        Compute linear regression of data -> need to deprecate this function for get_modelRegression..
        """
        df = pd.DataFrame()
        df["x"] = x
        df["y"] = y
        df = df.dropna()

        feature_name = "x"
        target_name = "y"

        data, target = df[[feature_name]], df[target_name]

        if len(df) > 1:

            x = df["x"].astype(float)
            y = df["y"].astype(float)

            lm = LinearRegression()
            lm.fit(data, target)
            predict = lm.predict(data)

            result = [lm.coef_[0], lm.intercept_]  # slope and intercept?
            result.append(lm.score(data, target))  # r score?
            result.append(abs((x - y).mean()))  # mean diff?

            mse = mean_squared_error(target, predict, multioutput="raw_values")
            rmse = np.sqrt(mse)
            result.append(mse[0])
            result.append(rmse[0])

        else:
            result = [None, None, None, None, None, None]
            result = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        # results order: m, c, r2, mean difference, mse, rmse

        # logger.debug(result)

        return result

    def post_adjustment_stats(self, inputdata, results, ref_col, TI_col):

        if isinstance(inputdata, pd.DataFrame):
            fillEmpty = False
            if ref_col in inputdata.columns and TI_col in inputdata.columns:
                model_adjTI = self.get_regression(inputdata[ref_col], inputdata[TI_col])
                name1 = "TI_regression_" + TI_col + "_" + ref_col
                results.loc[name1, ["m"]] = model_adjTI[0]
                results.loc[name1, ["c"]] = model_adjTI[1]
                results.loc[name1, ["rsquared"]] = model_adjTI[2]
                results.loc[name1, ["difference"]] = model_adjTI[3]
                results.loc[name1, ["mse"]] = model_adjTI[4]
                results.loc[name1, ["rmse"]] = model_adjTI[5]
            else:
                fillEmpty = True
        else:
            fillEmpty = True
        if fillEmpty:
            name1 = "TI_regression_" + TI_col + "_" + ref_col
            results.loc[name1, ["m"]] = "NaN"
            results.loc[name1, ["c"]] = "NaN"
            results.loc[name1, ["rsquared"]] = "NaN"
            results.loc[name1, ["difference"]] = "NaN"
            results.loc[name1, ["mse"]] = "NaN"
            results.loc[name1, ["rmse"]] = "NaN"
        return results

    def perform_SS_S_adjustment(self, inputdata):
        """
        Note: Representative TI computed with original RSD_SD
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
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ref_TI"]
            full["RSD_TI"] = inputdata_test["RSD_TI"]
            full = full.dropna()
            if len(full) < 2:
                results = self.post_adjustment_stats(
                    [None], results, "Ref_TI", "adjTI_RSD_TI"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = self.get_regression(
                    inputdata_train["RSD_TI"], inputdata_train["Ref_TI"]
                )
                m = model[0]
                c = model[1]
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (model[0] * RSD_TI) + model[1]
                inputdata_test["adjTI_RSD_TI"] = RSD_TI
                results = self.post_adjustment_stats(
                    inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
                )
            if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
                full = pd.DataFrame()
                full["Ref_TI"] = inputdata_test["Ane_TI_Ht1"]
                full["RSD_TI"] = inputdata_test["RSD_TI_Ht1"]
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_TI"], inputdata_train["Ref_TI"]
                    )
                    RSD_TI = inputdata_test["RSD_TI_Ht1"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                    )

            if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
                full = pd.DataFrame()
                full["Ref_TI"] = inputdata_test["Ane_TI_Ht2"]
                full["RSD_TI"] = inputdata_test["RSD_TI_Ht2"]
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_TI_Ht2"], inputdata_train["Ane_TI_Ht2"]
                    )
                    RSD_TI = inputdata_test["RSD_TI_Ht2"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                    )

            if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
                full = pd.DataFrame()
                full["Ref_TI"] = inputdata_test["Ane_TI_Ht3"]
                full["RSD_TI"] = inputdata_test["RSD_TI_Ht3"]
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_TI_Ht3"], inputdata_train["Ane_TI_Ht3"]
                    )
                    RSD_TI = inputdata_test["RSD_TI_Ht3"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                    )

            if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
                full = pd.DataFrame()
                full["Ref_TI"] = inputdata_test["Ane_TI_Ht4"]
                full["RSD_TI"] = inputdata_test["RSD_TI_Ht4"]
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_TI_Ht4"], inputdata_train["Ane_TI_Ht4"]
                    )
                    RSD_TI = inputdata_test["RSD_TI_Ht4"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )

        results["adjustment"] = ["SS-S"] * len(results)
        results = results.drop(columns=["sensor", "height"])

        return inputdata_test, results, m, c

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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                    )
                else:
                    model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht1"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht1"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht1"]
                    )
                    results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                    )
                else:
                    model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht2"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht2"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht2"]
                    )
                    results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                    )
                else:
                    model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht3"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht3"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht3"]
                    )
                    results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )
                else:
                    model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
                    RSD_TI = inputdata_test["RSD_TI_Ht4"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht4"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht4"]
                    )
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )

        results["adjustment"] = ["SS-SF"] * len(results)
        results = results.drop(columns=["sensor", "height"])

        return inputdata_test, results, m, c


def empirical_stdAdjustment(
    inputdata,
    results,
    Ref_TI_col,
    RSD_TI_col,
    Ref_SD_col,
    RSD_SD_col,
    Ref_WS_col,
    RSD_WS_col,
):
    """
    set adjustment values
    """
    inputdata_test = inputdata.copy()
    adj = Adjustments()

    # get col names
    name_ref = Ref_TI_col.split("_TI")
    name_rsd = RSD_TI_col.split("_TI")
    name = RSD_TI_col.split("_TI")
    adjTI_name = str("adjTI_" + RSD_TI_col)

    if len(inputdata) < 2:
        results = adj.post_adjustment_stats([None], results, Ref_TI_col, adjTI_name)
        m = np.NaN
        c = np.NaN
    else:
        # add the new columns, initialized by uncorrected Data
        tmp = str("adj" + RSD_SD_col)
        inputdata_test[tmp] = inputdata_test[RSD_SD_col].copy()
        inputdata_test[str("adjTI_" + RSD_TI_col)] = inputdata_test[RSD_TI_col].copy()

        inputdata_test.loc[
            ((inputdata[Ref_WS_col] >= 4) & (inputdata_test[Ref_WS_col] < 8)), tmp
        ] = ((1.116763 * inputdata_test[tmp]) + 0.024685) - (
            ((1.116763 * inputdata_test[tmp]) + 0.024685) * 0.00029
        )
        inputdata_test.loc[
            ((inputdata[Ref_WS_col] >= 4) & (inputdata_test[Ref_WS_col] < 8)),
            adjTI_name,
        ] = (
            inputdata_test[tmp] / inputdata_test[RSD_WS_col]
        )

        inputdata_test.loc[
            ((inputdata[Ref_WS_col] >= 8) & (inputdata_test[Ref_WS_col] < 12)), tmp
        ] = ((1.064564 * inputdata_test[tmp]) + 0.040596) - (
            ((1.064564 * inputdata_test[tmp]) + 0.040596) * -0.00161
        )
        inputdata_test.loc[
            ((inputdata[Ref_WS_col] >= 8) & (inputdata_test[Ref_WS_col] < 12)),
            adjTI_name,
        ] = (
            inputdata_test[tmp] / inputdata_test[RSD_WS_col]
        )

        inputdata_test.loc[
            ((inputdata[Ref_WS_col] >= 12) & (inputdata_test[Ref_WS_col] < 16)), tmp
        ] = ((0.97865 * inputdata_test[tmp]) + 0.124371) - (
            ((0.97865 * inputdata_test[tmp]) + 0.124371) * -0.00093
        )
        inputdata_test.loc[
            ((inputdata[Ref_WS_col] >= 12) & (inputdata_test[Ref_WS_col] < 16)),
            adjTI_name,
        ] = (
            inputdata_test[tmp] / inputdata_test[RSD_WS_col]
        )

        results = adj.post_adjustment_stats(inputdata_test, results, Ref_TI_col, adjTI_name)

    return inputdata_test, results
