import numpy as np
import pandas as pd
try:
    from TACT import logger
except ImportError:
    pass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
            results_regr = [res_name] + get_regression(
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
            res = [res_name] + get_regression(
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


def get_regression(x, y):
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

def log_of_ratio(x, xref):
    """Calculate natural logarithm of ratio between two values... useful for power law extrapolation

    Parameters
    ----------
    x : float
        numerator inside log
    xref : float
        denominator inside log

    Returns
    -------
    float
        log(x / xref)
    """
    x_new = np.log(x / xref)
    return x_new


def power_law(uref, h, href, shear):
    """Extrapolate wind speed (or other) according to power law.

    NOTE: see  https://en.wikipedia.org/wiki/Wind_profile_power_law

    Parameters
    ----------
    uref : float
        wind speed at reference height (same units as extrapolated wind speed, u)
    h : float
        height of extrapolated wind speed (same units as href)
    href : float
        reference height (same units as h)
    shear : float
        shear exponent alpha (1/7 in neutral stability) (unitless)

    Returns
    -------
    float
        extrapolated wind speed (same units as uref)
    """
    u = np.array(uref) * np.array(h / href) ** np.array(shear)
    return u
