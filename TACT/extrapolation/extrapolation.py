try:
    from TACT import logger
except ImportError:
    pass
import argparse
from future.utils import itervalues, iteritems
import numpy as np
import os
import pandas as pd
import re
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sys
from .calculations import log_of_ratio, power_law
from TACT.computation.post_adjustment import post_adjustment_stats
from TACT.computation.TI_computations import (
    get_TI_MBE_Diff_j,
    get_TI_Diff_r,
    get_TI_bybin,
    get_TI_byTIrefbin,
)


def get_extrap_col_and_ht(height, num, primary_height, sensor="Ane", var="WS"):
    """Determine name of column and height to use for extrapolation purposes

    Parameters
    ----------
    height : int or float
        comparison height
    num : int or str
        number/label of comparison height
    primary_height : int or float
        primary comparison height
    sensor : str
        type of sensor ("Ane", "RSD")
    var : str
        variable to be extracted ("WS", "SD", "TI")

    Returns
    -------
    string
        column name of data to be extracted from inputdata
    float
        height to be extracted for extrapolation
    """
    col_name = sensor + "_" + str(var) + "_Ht" + str(num)
    if "primary" in col_name:
        if sensor == "Ane":
            col_name = "Ref_" + str(var)
        elif sensor == "RSD":
            col_name = "RSD_" + str(var)
        height = float(primary_height)
    else:
        height = float(height)

    return col_name, height


def get_modelRegression_extrap(inputdata, column1, column2, fit_intercept=True):
    """

    Parameters
    ----------
    inputdata : DataFrame
    column1 : str
        column name for x-variable
    column2 : str
        column name for y-variable
    columnNameOut : str
        column name for predicted value

    Returns
    -------
    dict
        regression
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
    slope = regr.coef_[0]
    intercept = regr.intercept_
    predict = regr.predict(x)
    y = y.astype(np.float)
    r = np.corrcoef(x, y)[0, 1]
    r2 = r2_score(x, predict)  # coefficient of determination, explained variance
    mse = mean_squared_error(x, predict, multioutput="raw_values")
    rmse = np.sqrt(mse)
    difference = abs((x - y).mean())
    results = {
        "c": intercept,
        "m": slope,
        "r": r,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "predicted": predict,
        "difference": difference,
    }

    return results


def get_shear_exponent(inputdata, extrap_metadata, height):
    """Calculate shear exponent for TI extrapolation from two anemometers at different heights

    Parameters
    ----------
    inputdata : DataFrame
    extrap_metadata : DataFrame
        metadata required for TI extrapolation
    height : float
        Primary comparison height

    Returns
    -------
    float
        shear exponent calculated from anemometer data alpha = log(v2/v1)/ log(z2/z1)
    """
    # get columns
    inputs = extrap_metadata.loc[extrap_metadata["type"] == "input", ["height", "num"]]

    # Combine all anemometers into one table
    df = pd.DataFrame(None)
    for ix, irow in inputs.iloc[:-1, :].iterrows():
        col_i, ht_i = get_extrap_col_and_ht(irow["height"], irow["num"], height)
        for jx, jrow in inputs.loc[ix + 1 :, :].iterrows():
            col_j, ht_j = get_extrap_col_and_ht(jrow["height"], jrow["num"], height)
            tmp = pd.DataFrame(None)
            baseName = str(
                str(col_i) + "_" + str(col_j) + "_" + str(ht_i) + "_" + str(ht_j)
            )
            tmp[str(baseName + "_y")] = log_of_ratio(
                inputdata[col_i].values.astype(float),
                inputdata[col_j].values.astype(float),
            )
            tmp[str(baseName + "_x")] = log_of_ratio(ht_i, ht_j)
            tmp[str(baseName + "_alpha")] = (
                tmp[str(baseName + "_y")] / tmp[str(baseName + "_x")]
            )
            df = pd.concat((df, tmp), axis=1)
    df = df.reset_index(drop=True)

    # Calculate shear exponent
    alphaCols = [s for s in df.columns.to_list() if "alpha" in s]
    splitList = [s.split("_") for s in alphaCols]
    ht_diffs = []
    for l in splitList:
        htList = [item for item in l if "." in item]
        ht_diffs.append(float(htList[1]) - float(htList[0]))
    maxdif_idx = ht_diffs.index(max(ht_diffs))
    shearTimeseries = df[alphaCols[maxdif_idx]]
    # reg_extrap = get_modelRegression_extrap(df, 'x', 'y', fit_intercept=False)
    # shear = reg_extrap['m']

    return shearTimeseries


def perform_TI_extrapolation(inputdata, extrap_metadata, extrapolation_type, height):
    """Perform the TI extrapolation on anemometer data.

    Parameters
    ----------
    inputdata : DataFrame
    extrap_metadata : DataFrame
        metadata required for TI extrapolation
    extrapolation_type: str
        ("truth") decide what type of extrapolation to perform
    height : float
        Primary comparison height (m)

    Returns
    -------
    DataFrame
        input data (dataframe) with additional columns v2 = v1(z2/z1)^alpha
    """

    # Calculate shear exponent
    shearTimeseries = get_shear_exponent(inputdata, extrap_metadata, height)

    # TI columns and heights
    row = extrap_metadata.loc[extrap_metadata["type"] == "extrap", :].squeeze()
    col_extrap_ane, ht_extrap = get_extrap_col_and_ht(
        row["height"], row["num"], height, sensor="Ane", var="TI"
    )
    col_extrap_RSD, ht_extrap = get_extrap_col_and_ht(
        row["height"], row["num"], height, sensor="RSD", var="TI"
    )
    col_extrap_RSD_SD, ht_extrap = get_extrap_col_and_ht(
        row["height"], row["num"], height, sensor="RSD", var="SD"
    )
    col_extrap_ane_SD, ht_extrap = get_extrap_col_and_ht(
        row["height"], row["num"], height, sensor="Ane", var="SD"
    )

    # Select reference height just below extrapolation height
    hts = extrap_metadata.loc[extrap_metadata["height"] < ht_extrap, :]
    ref = hts.loc[hts["height"] == max(hts["height"]), :].squeeze()

    col_ref, ht_ref = get_extrap_col_and_ht(ref["height"], ref["num"], height, "Ane")
    col_ref_sd, ht_ref = get_extrap_col_and_ht(
        ref["height"], ref["num"], height, "Ane", var="SD"
    )

    # Extrapolate wind speed and st. dev. and calculate extrapolated TI
    WS_ane_extrap = power_law(inputdata[col_ref], ht_extrap, ht_ref, shearTimeseries)
    SD_ane_extrap = power_law(
        inputdata[col_ref_sd], ht_extrap, ht_ref, -shearTimeseries
    )
    TI_ane_extrap = SD_ane_extrap / WS_ane_extrap

    # Extract available TI values
    TI_RSD = inputdata[col_extrap_RSD].values
    SD_RSD = inputdata[col_extrap_RSD_SD].values
    if extrapolation_type == "truth":
        TI_ane_truth = inputdata[col_extrap_ane].values

    # Insert new columns into DataFrame
    inputdata["TI_RSD"] = TI_RSD
    inputdata["TI_ane_extrap"] = TI_ane_extrap
    if extrapolation_type == "truth":
        inputdata["TI_ane_truth"] = TI_ane_truth

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

    results = post_adjustment_stats(inputdata, results, "TI_RSD", "TI_ane_extrap")
    restults = post_adjustment_stats(inputdata, results, "TI_ane_truth", "TI_RSD")
    if extrapolation_type == "truth":
        results = post_adjustment_stats(
            inputdata, results, "TI_ane_truth", "TI_ane_extrap"
        )

    return inputdata, results, shearTimeseries


def change_extrap_names(TI_list, rename):
    """Rename columns and rows for tables created for TI extrapolation
    Parameters
    ----------
    TI_list : list
        list of DataFrames like TI_MBE_j_
    rename : dict
        dictionary to map existing names to new names, supplied to pd.DataFrame.rename()

    Returns
    -------
    list
        same list (TI_list) with all appropriate columns and rows renamed
    """
    for t, tab in enumerate(TI_list):
        if not isinstance(tab, list):
            continue
        for b, bins in enumerate(tab):
            if not isinstance(bins, pd.DataFrame):
                continue
            TI_list[t][b] = bins.rename(columns=rename, index=rename)

    return TI_list


def extrap_configResult(
    extrapolation_type, inputdataEXTRAP, resLists, method, lm_adj, appendString=""
):
    """Temporarily fudge the names of variables so they fit with the standard functions

    Parameters
    ----------
    extrapolation_type : str
        ("simple", "truth")
    inputdataEXTRAP : DataFrame
    resLists : list
    method : str
    lm_adj : DataFrame
    appendString : str

    Returns
    -------
    DataFrame
    List

    """
    if "corrWS_RSD_TI" in inputdataEXTRAP:
        inputdataEXTRAP = inputdataEXTRAP.drop(columns=["corrWS_RSD_TI"])
    if "Ane2_TI" in inputdataEXTRAP:
        inputdataEXTRAP = inputdataEXTRAP.drop(columns=["Ane2_TI"])
    if extrapolation_type == "truth":
        inputdataEXTRAP["RSD_TI"] = inputdataEXTRAP["TI_ane_truth"]
    else:
        inputdataEXTRAP = inputdataEXTRAP.drop(columns=["RSD_TI"])
    inputdataEXTRAP["Ref_TI"] = inputdataEXTRAP["TI_ane_extrap"]
    inputdataEXTRAP["corrTI_RSD_TI"] = inputdataEXTRAP["TI_RSD"]

    # Run through the standard functions
    try:
        (
            TI_MBE_j_,
            TI_Diff_j_,
            TI_RMSE_j_,
            RepTI_MBE_j_,
            RepTI_Diff_j_,
            RepTI_RMSE_j_,
        ) = get_TI_MBE_Diff_j(inputdataEXTRAP)
        TI_Diff_r_, RepTI_Diff_r_ = get_TI_Diff_r(inputdataEXTRAP)
        rep_TI_results_1mps, rep_TI_results_05mps = get_representative_TI(
            inputdataEXTRAP
        )
        TIbybin = get_TI_bybin(inputdataEXTRAP)
        TIbyRefbin = get_TI_byTIrefbin(inputdataEXTRAP)
        total_stats, belownominal_stats, abovenominal_stats = get_description_stats(
            inputdataEXTRAP
        )
        Distribution_stats, sampleTests = Dist_stats(
            inputdataEXTRAP, Timestamps, adjustment_name
        )
        resDict = True

    except:
        resDict = False
        resLists[str("TI_MBEList_" + appendString)].append(None)
        resLists[str("TI_DiffList_" + appendString)].append(None)
        resLists[str("TI_DiffRefBinsList_" + appendString)].append(None)
        resLists[str("TI_RMSEList_" + appendString)].append(None)
        resLists[str("RepTI_MBEList_" + appendString)].append(None)
        resLists[str("RepTI_DiffList_" + appendString)].append(None)
        resLists[str("RepTI_DiffRefBinsList_" + appendString)].append(None)
        resLists[str("RepTI_RMSEList_" + appendString)].append(None)
        resLists[str("rep_TI_results_1mps_List_" + appendString)].append(None)
        resLists[str("rep_TI_results_05mps_List_" + appendString)].append(None)
        resLists[str("TIBinList_" + appendString)].append(None)
        resLists[str("TIRefBinList_" + appendString)].append(None)
        resLists[str("total_StatsList_" + appendString)].append(None)
        resLists[str("belownominal_statsList_" + appendString)].append(None)
        resLists[str("abovenominal_statsList_" + appendString)].append(None)
        resLists[str("lm_adjList_" + appendString)].append(lm_adj)
        resLists[str("adjustmentTagList_" + appendString)].append(method)
        resLists[str("Distribution_statsList_" + appendString)].append(None)
        resLists[str("sampleTestsLists_" + appendString)].append(None)

    if resDict:
        resDict = {}
        # Rename labels after forcing them through the standard functions
        rename = {
            "TI_diff_RSD_Ref": "TI_diff_AneTruth_AneExtrap",
            "TI_diff_adjTI_RSD_Ref": "TI_diff_RSD_AneExtrap",
            "TI_error_RSD_Ref": "TI_error_AneTruth_AneExtrap",
            "TI_error_adjTI_RSD_Ref": "TI_error_RSD_AneExtrap",
            "TI_RMSE_RSD_Ref": "TI_RMSE_AneTruth_AneExtrap",
            "TI_RMSE_adjTI_RSD_Ref": "TI_RMSE_RSD_AneExtrap",
            "Ref_TI": "AneExtrap_TI",
            "RSD_TI": "AneTruth_TI",
            "corrTI_RSD_TI": "RSD_TI",
            "RSD_Ref": "AneTruth_AneExtrap",
            "CorrTI_RSD_Ref": "RSD_AneExtrap",
            "RepTI_diff_RSD_Ref": "RepTI_diff_AneTruth_AneExtrap",
            "RepTI_diff_adjRepTI_RSD_Ref": "RepTI_diff_RSD_AneExtrap",
            "RepTI_error_RSD_Ref": "RepTI_error_AneTruth_AneExtrap",
            "RepTI_error_adjRepTI_RSD_Ref": "RepTI_error_RSD_AneExtrap",
            "RepTI_RMSE_RSD_Ref": "RepTI_RMSE_AneTruth_AneExtrap",
            "RepTI_RMSE_adjRepTI_RSD_Ref": "RepTI_RMSE_RSD_AneExtrap",
        }
        resDict["TI_MBE_j_"] = change_extrap_names(TI_MBE_j_, rename)
        resDict["TI_Diff_j_"] = change_extrap_names(TI_Diff_j_, rename)
        resDict["TI_Diff_r_"] = change_extrap_names(TI_Diff_r_, rename)
        resDict["TI_RMSE_j_"] = change_extrap_names(TI_RMSE_j_, rename)
        resDict["RepTI_MBE_j_"] = change_extrap_names(RepTI_MBE_j_, rename)
        resDict["RepTI_Diff_j_"] = change_extrap_names(RepTI_Diff_j_, rename)
        resDict["RepTI_Diff_r_"] = change_extrap_names(RepTI_Diff_r_, rename)
        resDict["RepTI_RMSE_j_"] = change_extrap_names(RepTI_RMSE_j_, rename)
        resDict["rep_TI_results_1mps"] = change_extrap_names(
            [[rep_TI_results_1mps]], rename
        )[0][0]
        resDict["rep_TI_results_05mps"] = change_extrap_names(
            [[rep_TI_results_05mps]], rename
        )[0][0]
        resDict["TIbybin"] = change_extrap_names(TIbybin, rename)
        resDict["TIbyRefbin"] = change_extrap_names(TIbyRefbin, rename)
        resDict["total_stats"] = tuple(
            change_extrap_names([list(total_stats)], rename)[0]
        )
        resDict["belownominal_stats"] = tuple(
            change_extrap_names([list(belownominal_stats)], rename)[0]
        )
        resDict["abovenominal_stats"] = tuple(
            change_extrap_names([list(abovenominal_stats)], rename)[0]
        )
        resDict["Distribution_stats"] = change_extrap_names(Distribution_stats, rename)
        resDict["sampleTests"] = change_extrap_names(sampleTests, rename)

        resLists[str("TI_MBEList_" + appendString)].append(resDict["TI_MBE_j_"])
        resLists[str("TI_DiffList_" + appendString)].append(resDict["TI_Diff_j_"])
        resLists[str("TI_DiffRefBinsList_" + appendString)].append(
            resDict["TI_Diff_r_"]
        )
        resLists[str("TI_RMSEList_" + appendString)].append(resDict["TI_RMSE_j_"])
        resLists[str("RepTI_MBEList_" + appendString)].append(resDict["RepTI_MBE_j_"])
        resLists[str("RepTI_DiffList_" + appendString)].append(resDict["RepTI_Diff_j_"])
        resLists[str("RepTI_DiffRefBinsList_" + appendString)].append(
            resDict["RepTI_Diff_r_"]
        )
        resLists[str("RepTI_RMSEList_" + appendString)].append(resDict["RepTI_RMSE_j_"])
        resLists[str("rep_TI_results_1mps_List_" + appendString)].append(
            resDict["rep_TI_results_1mps"]
        )
        resLists[str("rep_TI_results_05mps_List_" + appendString)].append(
            resDict["rep_TI_results_05mps"]
        )
        resLists[str("TIBinList_" + appendString)].append(resDict["TIbybin"])
        resLists[str("TIRefBinList_" + appendString)].append(resDict["TIbyRefbin"])
        resLists[str("total_StatsList_" + appendString)].append(resDict["total_stats"])
        resLists[str("belownominal_statsList_" + appendString)].append(
            resDict["belownominal_stats"]
        )
        resLists[str("abovenominal_statsList_" + appendString)].append(
            resDict["abovenominal_stats"]
        )
        resLists[str("lm_adjList_" + appendString)].append(lm_adj)
        resLists[str("adjustmentTagList_" + appendString)].append(method)
        resLists[str("Distribution_statsList_" + appendString)].append(
            resDict["Distribution_stats"]
        )
        resLists[str("sampleTestsLists_" + appendString)].append(resDict["sampleTests"])

    return inputdataEXTRAP, resLists
