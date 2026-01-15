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

from TACT.computation.calculations import get_regression

class Adjustments:

    """
    class to hold adjusted data and results

    Attributes
    ----------
    raw_data: pandas dataframe
    adjusted_data: dictionary of adjusted data based on list of methods
    results_stats: dictionary of results based on list of methods
    """

    def __init__(self, raw_data="", adjusted_data_list=""):
        logger.debug("Generating Adjustments object")
        self.raw_data = raw_data
        self.adjusted_data = {}
        for i in adjusted_data_list.keys(): 
            self.adjusted_data[i] = None
        self.results_stats = {}
        for i in adjusted_data_list.keys(): 
            self.results_stats[i] = None

    def method_record(self, data, config, method, inputdata_adj): 
        from TACT.writers.labels import populate_resultsLists, populate_resultsLists_stability
        from TACT.computation.TI import record_TIadj

        lm_adj = {}
        lm_adj['sensor'] = config.model
        lm_adj["height"] = config.height
        lm_adj["adjustment"] = method

        self.results_stats[method] = populate_resultsLists(
            "",
            method,
            lm_adj,
            inputdata_adj,
            data.timestamps,
            method,
            )

        self.adjusted_data[method] = record_TIadj(
            method,
            inputdata_adj,
            data.timestamps,
            method,
            emptyclassFlag=False,
            )

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

def post_adjustment_stats(inputdata, results, ref_col, TI_col):

        if isinstance(inputdata, pd.DataFrame):
            fillEmpty = False
            if ref_col in inputdata.columns and TI_col in inputdata.columns:
                model_adjTI = get_regression(inputdata[ref_col], inputdata[TI_col])
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