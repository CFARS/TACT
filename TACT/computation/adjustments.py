try:
    from TACT import logger
except ImportError:
    pass
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()  # setting to non-interactive
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# from TACT.computation.methods.GSa import perform_G_Sa_adjustment
# from TACT.computation.methods.GSFc import perform_G_SFc_adjustment
# from TACT.computation.methods.SSLTERRAML import perform_SS_LTERRA_ML_adjustment
# from TACT.computation.methods.SSLTERRASML import perform_SS_LTERRA_S_ML_adjustment
# from TACT.computation.methods.SSNN import perform_SS_NN_adjustment
# from TACT.computation.methods.SSSS import perform_SS_SS_adjustment
# from TACT.computation.methods.SSWS import perform_SS_WS_adjustment
# from TACT.computation.methods.SSWSStd import perform_SS_WS_Std_adjustment


class Adjustments:

    """
    class to hold adjusted data and results

    Attributes
    ----------
    raw_data: 
    adjusted_data: 
    results_stats: 

    """

    def __init__(self, raw_data="", adjustments_list="", baseResultsLists=""):
        logger.debug("Generating Adjustments object")
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

    def get_modelRegression(self, inputdata, column1, column2, fit_intercept=True):
        """
        Parameters
        ----------
        inputdata : dataframe
        column1 : string
            column name for x-variable
        column2 : string
            column name for y-variable
        columnNameOut : string
            column name for predicted value

        Returns
        -------
        dict
            output of regression
        """
        x = inputdata[column1].values.astype(float)
        y = inputdata[column2].values.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)
        regr = linear_model.LinearRegression(fit_intercept=fit_intercept)
        regr.fit(x, y)
        slope = regr.coef_[0][0]
        intercept = regr.intercept_[0]
        predict = regr.predict(x)
        y = y.astype(np.float)
        r = np.corrcoef(x, y)[0, 1]
        r2 = r2_score(y, predict)  # coefficient of determination, explained variance
        mse = mean_squared_error(y, predict, multioutput="raw_values")[0]
        rmse = np.sqrt(mse)
        difference = abs((x - y).mean())
        resultsDict = {
            "c": intercept,
            "m": slope,
            "r": r,
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "predicted": predict,
            "difference": difference,
        }
        result = [slope, intercept, r2, difference, mse, rmse]

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
        Adjusts 10-minute averaged TI with a linear slope and offset calibration method 
        derived from the test data 

        Parameters
        ----------
        inputdata : dataframe 

        Returns 
        -------
        inputdata_adj : dataframe
        results : dataframe
        m :  numeric
            slope
        c : numeric
            intercept

        Notes
        -----
        Note: Representative TI computed with original RSD_SD

        References
        ----------
        To do: FIND/GET REFERENCE!
        
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
        """
        Adjusts 10-minute averaged TI data with a linear slope and offset calibration method
        with a filter applied 
        derived from the test data 

        Parameters
        ----------
        inputdata : dataframe 

        Returns 
        -------
        inputdata_adj : dataframe
        results : dataframe
        m :  numeric
        c : numeric

        Notes
        -----
        Note: Representative TI computed with original RSD_SD

        References
        ----------
        To do: FIND/GET REFERENCE!
        
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

    def perform_SS_WS_adjustment(self, inputdata):
        """
        Adjusts 10-minute averaged wind speed data with a linear slope and offset calibration method 
        derived from the test data 

        Parameters
        ----------
        inputdata : dataframe 

        Returns 
        -------
        inputdata_adj : dataframe
        results : dataframe
        m :  numeric
        c : numeric

        Notes
        -----
        Note: Representative TI computed with original RSD_SD

        References
        ----------
        To do: FIND/GET REFERENCE!
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
            if "Ane_WS_Ht1" in inputdata.columns and "RSD_WS_Ht1" in inputdata.columns:
                results = self.post_adjustment_stats(
                    [None], results, "ane_ti_ht1", "adjTI_rsd_ti_ht1"
                )
            if "ane_ws_ht2" in inputdata.columns and "RSD_WS_Ht2" in inputdata.columns:
                results = self.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
            if "Ane_WS_Ht3" in inputdata.columns and "RSD_WS_Ht3" in inputdata.columns:
                results = self.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
            if "Ane_WS_Ht4" in inputdata.columns and "RSD_WS_Ht4" in inputdata.columns:
                results = self.post_adjustment_stats(
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
                results = self.post_adjustment_stats(
                    [None], results, "Ref_TI", "adjTI_RSD_TI"
                )
                m = np.NaN
                c = np.NaN
            else:
                model = self.get_regression(
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
                results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_WS_Ht1"], inputdata_train["Ane_WS_Ht1"]
                    )
                    RSD_WS = inputdata_test["RSD_WS_Ht1"]

                    RSD_adjWS = (model[0] * RSD_WS) + model[1]
                    inputdata_test["RSD_adjWS_Ht1"] = RSD_adjWS
                    RSD_TI = (
                        inputdata_test["RSD_SD_Ht1"] / inputdata_test["RSD_adjWS_Ht1"]
                    )
                    inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                    results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_WS_Ht2"], inputdata_train["Ane_WS_Ht2"]
                    )
                    RSD_WS = inputdata_test["RSD_WS_Ht2"]
                    RSD_adjWS = (model[0] * RSD_WS) + model[1]
                    inputdata_test["RSD_adjWS_Ht2"] = RSD_adjWS
                    RSD_TI = (
                        inputdata_test["RSD_SD_Ht2"] / inputdata_test["RSD_adjWS_Ht2"]
                    )
                    inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                    results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_WS_Ht3"], inputdata_train["Ane_WS_Ht3"]
                    )
                    RSD_WS = inputdata_test["RSD_WS_Ht3"]
                    RSD_adjWS = (model[0] * RSD_WS) + model[1]
                    inputdata_test["RSD_adjWS_Ht3"] = RSD_adjWS
                    RSD_TI = (
                        inputdata_test["RSD_SD_Ht3"] / inputdata_test["RSD_adjWS_Ht3"]
                    )
                    inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                    results = self.post_adjustment_stats(
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
                    results = self.post_adjustment_stats(
                        [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(
                        inputdata_train["RSD_WS_Ht4"], inputdata_train["Ane_WS_Ht4"]
                    )
                    RSD_WS = inputdata_test["RSD_WS_Ht4"]
                    RSD_adjWS = (model[0] * RSD_WS) + model[1]
                    inputdata_test["RSD_adjWS_Ht4"] = RSD_adjWS
                    RSD_TI = (
                        inputdata_test["RSD_SD_Ht4"] / inputdata_test["RSD_adjWS_Ht4"]
                    )
                    inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )

        results["adjustment"] = ["SS-WS"] * len(results)
        results = results.drop(columns=["sensor", "height"])

        return inputdata_test, results, m, c

    def perform_G_Sa_adjustment(self, inputdata, override, RSDtype):
        """
        simple filtered regression results from phase2 averages with simple regression from this data 

        Parameters
        ----------
        inputdata : dataframe 

        Returns 
        -------
        inputdata_adj : dataframe
        results : dataframe
        m :  numeric
        c : numeric

        Notes
        -----
        Note: Representative TI computed with original RSD_SD

        References
        ----------
        To do: FIND/GET REFERENCE!
        
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
                m = (model[0] + m_ph2) / 2
                c = (model[1] + c_ph2) / 2
                RSD_TI = inputdata_test["RSD_TI"].copy()
                RSD_TI = (m * RSD_TI) + c
                inputdata_test["adjTI_RSD_TI"] = RSD_TI
                inputdata_test["adjRepTI_RSD_RepTI"] = (
                    RSD_TI + 1.28 * inputdata_test["RSD_SD"]
                )
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
                    m = (model[0] + m_ph2) / 2
                    c = (model[1] + c_ph2) / 2
                    RSD_TI = inputdata_test["RSD_TI"].copy()
                    RSD_TI = (m * RSD_TI) + c
                    inputdata_test["adjTI_RSD_TI_Ht1"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht1"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht1"]
                    )
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
                    m = (model[0] + m_ph2) / 2
                    c = (model[1] + c_ph2) / 2
                    RSD_TI = inputdata_test["RSD_TI"].copy()
                    RSD_TI = (m * RSD_TI) + c
                    inputdata_test["adjTI_RSD_TI_Ht2"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht2"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht2"]
                    )
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
                    m = (model[0] + m_ph2) / 2
                    c = (model[1] + c_ph2) / 2
                    RSD_TI = inputdata_test["RSD_TI"].copy()
                    RSD_TI = (m * RSD_TI) + c
                    inputdata_test["adjTI_RSD_TI_Ht3"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht3"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht3"]
                    )
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
                    m = (model[0] + m_ph2) / 2
                    c = (model[1] + c_ph2) / 2
                    RSD_TI = inputdata_test["RSD_TI"].copy()
                    RSD_TI = (m * RSD_TI) + c
                    inputdata_test["adjTI_RSD_TI_Ht4"] = RSD_TI
                    inputdata_test["adjRepTI_RSD_RepTI_Ht4"] = (
                        RSD_TI + 1.28 * inputdata_test["RSD_SD_Ht4"]
                    )
                    results = self.post_adjustment_stats(
                        inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                    )

        results["adjustment"] = ["G-Sa"] * len(results)
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

        results = adj.post_adjustment_stats(
            inputdata_test, results, Ref_TI_col, adjTI_name
        )

    return inputdata_test, results


def train_test_split(trainPercent, inputdata, stepOverride=False):
    """
    train is 'split' == True
    """
    import copy
    import numpy as np

    _inputdata = pd.DataFrame(
        columns=inputdata.columns, data=copy.deepcopy(inputdata.values)
    )

    if stepOverride:
        msk = [False] * len(inputdata)
        _inputdata["split"] = msk
        _inputdata.loc[stepOverride[0] : stepOverride[1], "split"] = True

    else:
        msk = np.random.rand(len(_inputdata)) < float(trainPercent / 100)
        train = _inputdata[msk]
        test = _inputdata[~msk]
        _inputdata["split"] = msk

    return _inputdata


def quick_metrics(inputdata, config, results_df, lm_adj_dict, testID):
    """"""
    from TACT.computation.match import perform_match, perform_match_input

    _adjuster = Adjustments(raw_data=inputdata)

    inputdata_train = inputdata[inputdata["split"] == True].copy()
    inputdata_test = inputdata[inputdata["split"] == False].copy()

    # baseline results
    results_ = get_all_regressions(inputdata_test, title="baselines")
    results_RSD_Ref = results_.loc[
        results_["baselines"].isin(["TI_regression_Ref_RSD"])
    ].reset_index()
    results_Ane2_Ref = results_.loc[
        results_["baselines"].isin(["TI_regression_Ref_Ane2"])
    ].reset_index()
    results_RSD_Ref_SD = results_.loc[
        results_["baselines"].isin(["SD_regression_Ref_RSD"])
    ].reset_index()
    results_Ane2_Ref_SD = results_.loc[
        results_["baselines"].isin(["SD_regression_Ref_Ane2"])
    ].reset_index()
    results_RSD_Ref_WS = results_.loc[
        results_["baselines"].isin(["WS_regression_Ref_RSD"])
    ].reset_index()
    results_Ane2_Ref_WS = results_.loc[
        results_["baselines"].isin(["WS_regression_Ref_Ane2"])
    ].reset_index()
    results_RSD_Ref.loc[0, "testID"] = [testID]
    results_Ane2_Ref.loc[0, "testID"] = [testID]
    results_RSD_Ref_SD.loc[0, "testID"] = [testID]
    results_Ane2_Ref_SD.loc[0, "testID"] = [testID]
    results_RSD_Ref_WS.loc[0, "testID"] = [testID]
    results_Ane2_Ref_WS.loc[0, "testID"] = [testID]
    results_df = pd.concat(
        [
            results_df,
            results_RSD_Ref,
            results_Ane2_Ref,
            results_RSD_Ref_SD,
            results_Ane2_Ref_SD,
            results_RSD_Ref_WS,
            results_Ane2_Ref_WS,
        ],
        axis=0,
    )

    # Run a few adjustments with this timing test aswell
    inputdata_adj, lm_adj, m, c = _adjuster.perform_SS_S_adjustment(inputdata.copy())
    lm_adj_dict[str(str(testID) + " :SS_S")] = lm_adj
    inputdata_adj, lm_adj, m, c = _adjuster.perform_SS_SF_adjustment(inputdata.copy())
    lm_adj_dict[str(str(testID) + " :SS_SF")] = lm_adj
    inputdata_adj, lm_adj, m, c = _adjuster.perform_SS_WS_adjustment(inputdata.copy())
    lm_adj_dict[str(str(testID) + " :SS_WS-Std")] = lm_adj
    inputdata_adj, lm_adj = perform_match(inputdata.copy())
    lm_adj_dict[str(str(testID) + " :Match")] = lm_adj
    inputdata_adj, lm_adj = perform_match_input(inputdata.copy())
    lm_adj_dict[str(str(testID) + " :SS_Match_erforminput")] = lm_adj
    override = False
    inputdata_adj, lm_adj, m, c = _adjuster.perform_G_Sa_adjustment(
        inputdata.copy(), override, config.RSDtype
    )
    lm_adj_dict[str(str(testID) + " :SS_G_SFa")] = lm_adj

    return results_df, lm_adj_dict


def get_all_regressions(inputdata, title=None):
    """Create dataframe of all regression statistics for all instrument height comparisons
    
    Parameters:
    -----------
    reader : pandas dataframe
        all data to analyze for regression statistics
    title : string
        string to label the results dataframe
    
    Returns:
    --------
    daily_sr : pandas dataframe
        regression statistics output for each compared instrument/height

    References:
    -----------

    """

    pairList = [
        ["Ref_WS", "RSD_WS"],
        ["Ref_WS", "Ane2_WS"],
        ["Ref_TI", "RSD_TI"],
        ["Ref_TI", "Ane2_TI"],
        ["Ref_SD", "RSD_SD"],
        ["Ref_SD", "Ane2_SD"],
    ]

    lenFlag = False
    if len(inputdata) < 2:
        lenFlag = True

    columns = [title, "m", "c", "rsquared", "mean difference", "mse", "rmse"]
    results = pd.DataFrame(columns=columns)

    logger.debug(f"getting regr for {title}")

    for p in pairList:

        res_name = str(
            p[0].split("_")[1]
            + "_regression_"
            + p[0].split("_")[0]
            + "_"
            + p[1].split("_")[0]
        )

        if p[1] in inputdata.columns and lenFlag == False:
            _adjuster = Adjustments(inputdata)
            results_regr = [res_name] + _adjuster.get_regression(
                inputdata[p[0]], inputdata[p[1]]
            )

        else:
            results_regr = [res_name, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]

        _results = pd.DataFrame(columns=columns, data=[results_regr])
        results = pd.concat(
            [results, _results], ignore_index=True, axis=0, join="outer"
        )

    # labels not required
    labelsExtra = [
        "RSD_SD_Ht1",
        "RSD_TI_Ht1",
        "RSD_WS_Ht1",
        "RSD_SD_Ht2",
        "RSD_TI_Ht2",
        "RSD_WS_Ht2",
        "RSD_SD_Ht3",
        "RSD_TI_Ht3",
        "RSD_WS_Ht3",
        "RSD_WS_Ht4",
        "RSD_SD_Ht4",
        "RSD_TI_Ht4",
    ]
    labelsRef = ["Ref_WS", "Ref_TI", "Ref_SD"]
    labelsAne = [
        "Ane_SD_Ht1",
        "Ane_TI_Ht1",
        "Ane_WS_Ht1",
        "Ane_SD_Ht2",
        "Ane_TI_Ht2",
        "Ane_WS_Ht2",
        "Ane_SD_Ht3",
        "Ane_TI_Ht3",
        "Ane_WS_Ht3",
        "Ane_WS_Ht4",
        "Ane_SD_Ht4",
        "Ane_TI_Ht4",
    ]

    for l in labelsExtra:

        parts = l.split("_")
        reg_type = list(set(parts).intersection(["WS", "TI", "SD"]))

        if "RSD" in l:
            ht_type = parts[2]
            ref_type = [s for s in labelsAne if reg_type[0] in s]
            ref_type = [s for s in ref_type if ht_type in s]

        res_name = str(reg_type[0] + "_regression_" + parts[0])

        if "Ht" in parts[2]:
            res_name = (
                res_name
                + parts[2]
                + "_"
                + ref_type[0].split("_")[0]
                + ref_type[0].split("_")[2]
            )

        else:
            res_name = res_name + "_Ref"

        logger.debug(res_name)

        if l in inputdata.columns and lenFlag == False:
            _adjuster = Adjustments(inputdata)
            res = [res_name] + _adjuster.get_regression(
                inputdata[ref_type[0]], inputdata[l]
            )

        else:
            res = [res_name, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]

        logger.debug(res)

        _results = pd.DataFrame(columns=columns, data=[res])
        results = pd.concat(
            [results, _results], ignore_index=True, axis=0, join="outer"
        )

    return results
