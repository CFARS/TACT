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
from TACT.extrapolation.calculations import log_of_ratio, power_law
from TACT.computation.adjustments import Adjustments
from TACT.computation.post_adjustment import post_adjustment_stats


def get_representative_TI(inputdata):
    # get the representaive TI, there is an error here. Std is std of WS not TI so the calculation is currently in error
    if "Ane2_TI" in inputdata.columns:
        representative_TI_bins = (
            inputdata[["RSD_TI", "Ref_TI", "Ane2_TI", "bins"]]
            .groupby(by=["bins"])
            .agg(["mean", "std", lambda x: x.mean() + 1.28 * x.std()])
        )
        representative_TI_bins.columns = [
            "RSD_TI_mean",
            "RSD_TI_std",
            "RSD_TI_rep",
            "Ref_TI_mean",
            "Ref_TI_std",
            "Ref_TI_rep",
            "Ane2_TI_mean",
            "Ane2_TI_std",
            "Ane2_TI_rep",
        ]

        representative_TI_binsp5 = (
            inputdata[["RSD_TI", "Ref_TI", "Ane2_TI", "bins_p5"]]
            .groupby(by=["bins_p5"])
            .agg(["mean", "std", lambda x: x.mean() + 1.28 * x.std()])
        )
        representative_TI_binsp5.columns = [
            "RSD_TI_mean",
            "RSD_TI_std",
            "RSD_TI_rep",
            "Ref_TI_mean",
            "Ref_TI_std",
            "Ref_TI_rep",
            "Ane2_TI_mean",
            "Ane2_TI_std",
            "Ane2_TI_rep",
        ]
    else:
        representative_TI_bins = (
            inputdata[["RSD_TI", "Ref_TI", "bins"]]
            .groupby(by=["bins"])
            .agg(["mean", "std", lambda x: x.mean() + 1.28 * x.std()])
        )
        representative_TI_bins.columns = [
            "RSD_TI_mean",
            "RSD_TI_std",
            "RSD_TI_rep",
            "Ref_TI_mean",
            "Ref_TI_std",
            "Ref_TI_rep",
        ]

        representative_TI_binsp5 = (
            inputdata[["RSD_TI", "Ref_TI", "bins_p5"]]
            .groupby(by=["bins_p5"])
            .agg(["mean", "std", lambda x: x.mean() + 1.28 * x.std()])
        )
        representative_TI_binsp5.columns = [
            "RSD_TI_mean",
            "RSD_TI_std",
            "RSD_TI_rep",
            "Ref_TI_mean",
            "Ref_TI_std",
            "Ref_TI_rep",
        ]

    return representative_TI_bins, representative_TI_binsp5


def get_count_per_WSbin(inputdata, column):
    # Count per wind speed bin
    inputdata = inputdata[
        (inputdata["bins_p5"].astype(float) > 1.5)
        & (inputdata["bins_p5"].astype(float) < 21)
    ]
    resultsstats_bin = inputdata[[column, "bins"]].groupby(by="bins").agg(["count"])
    resultsstats_bin_p5 = (
        inputdata[[column, "bins_p5"]].groupby(by="bins_p5").agg(["count"])
    )
    resultsstats_bin = pd.DataFrame(resultsstats_bin.unstack()).T
    resultsstats_bin.index = [column]
    resultsstats_bin_p5 = pd.DataFrame(resultsstats_bin_p5.unstack()).T
    resultsstats_bin_p5.index = [column]
    return resultsstats_bin, resultsstats_bin_p5


def get_stats_per_WSbin(inputdata, column):
    # this will be used as a base function for all frequency agg caliculaitons for each bin to get the stats per wind speed bins
    inputdata = inputdata[
        (inputdata["bins_p5"].astype(float) > 1.5)
        & (inputdata["bins_p5"].astype(float) < 21)
    ]
    resultsstats_bin = (
        inputdata[[column, "bins"]].groupby(by="bins").agg(["mean", "std"])
    )  # get mean and standard deviation of values in the 1mps bins
    resultsstats_bin_p5 = (
        inputdata[[column, "bins_p5"]].groupby(by="bins_p5").agg(["mean", "std"])
    )  # get mean and standard deviation of values in the 05mps bins
    resultsstats_bin = pd.DataFrame(resultsstats_bin.unstack()).T
    resultsstats_bin.index = [column]
    resultsstats_bin_p5 = pd.DataFrame(resultsstats_bin_p5.unstack()).T
    resultsstats_bin_p5.index = [column]
    return resultsstats_bin, resultsstats_bin_p5


def get_stats_per_TIbin(inputdata, column):
    # this will be used as a base function for all frequency agg caliculaitons for each bin to get the stats per refereence TI bins
    inputdata = inputdata[
        (inputdata["RefTI_bins"].astype(float) > 0.00)
        & (inputdata["RefTI_bins"].astype(float) < 1.0)
    ]
    resultsstats_RefTI_bin = (
        inputdata[[column, "RefTI_bins"]].groupby(by="RefTI_bins").agg(["mean", "std"])
    )  # get mean and standard deviation of values in the 05mps bins
    resultsstats_RefTI_bin = pd.DataFrame(resultsstats_RefTI_bin.unstack()).T
    resultsstats_RefTI_bin.index = [column]

    return resultsstats_RefTI_bin


def get_RMSE_per_WSbin(inputdata, column):
    """
    get RMSE with no fit model, just based on residual being the reference
    """
    squared_TI_Diff_j_RSD_Ref, squared_TI_Diff_jp5_RSD_Ref = get_stats_per_WSbin(
        inputdata, column
    )
    TI_RMSE_j = squared_TI_Diff_j_RSD_Ref ** (0.5)
    TI_RMSE_jp5 = squared_TI_Diff_jp5_RSD_Ref ** (0.5)
    TI_RMSE_j = TI_RMSE_j[column].drop(columns=["std"])
    TI_RMSE_jp5 = TI_RMSE_jp5[column].drop(columns=["std"])

    idx = TI_RMSE_j.index
    old = idx[0]
    idx_str = idx[0].replace("SquaredDiff", "RMSE")
    TI_RMSE_j = TI_RMSE_j.rename(index={old: idx_str})
    idxp5 = TI_RMSE_jp5.index
    oldp5 = idxp5[0]
    idxp5_str = idxp5[0].replace("SquaredDiff", "RMSE")
    TI_RMSE_jp5 = TI_RMSE_jp5.rename(index={oldp5: idxp5_str})

    return TI_RMSE_j, TI_RMSE_jp5


def get_TI_MBE_Diff_j(inputdata):

    TI_MBE_j_ = []
    TI_Diff_j_ = []
    TI_RMSE_j_ = []

    RepTI_MBE_j_ = []
    RepTI_Diff_j_ = []
    RepTI_RMSE_j_ = []

    # get the bin wise stats for DIFFERENCE and ERROR and RMSE between RSD and Ref TI (UNCORRECTED)
    if "RSD_TI" in inputdata.columns:
        inputdata["RSD_TI"] = inputdata["RSD_TI"].astype(float)
        inputdata["Ref_TI"] = inputdata["Ref_TI"].astype(float)
        inputdata["TI_diff_RSD_Ref"] = (
            inputdata["RSD_TI"] - inputdata["Ref_TI"]
        )  # caliculating the diff in ti for each timestamp
        inputdata["TI_error_RSD_Ref"] = (
            inputdata["TI_diff_RSD_Ref"] / inputdata["Ref_TI"]
        )  # calculating the error for each timestamp (diff normalized to ref_TI)
        inputdata["TI_SquaredDiff_RSD_Ref"] = (
            inputdata["TI_diff_RSD_Ref"] * inputdata["TI_diff_RSD_Ref"]
        )  # calculating squared diff each Timestamp
        TI_MBE_j_RSD_Ref, TI_MBE_jp5_RSD_Ref = get_stats_per_WSbin(
            inputdata, "TI_error_RSD_Ref"
        )
        TI_Diff_j_RSD_Ref, TI_Diff_jp5_RSD_Ref = get_stats_per_WSbin(
            inputdata, "TI_diff_RSD_Ref"
        )
        TI_RMSE_j_RSD_Ref, TI_RMSE_jp5_RSD_Ref = get_RMSE_per_WSbin(
            inputdata, "TI_SquaredDiff_RSD_Ref"
        )

        TI_MBE_j_.append([TI_MBE_j_RSD_Ref, TI_MBE_jp5_RSD_Ref])
        TI_Diff_j_.append([TI_Diff_j_RSD_Ref, TI_Diff_jp5_RSD_Ref])
        TI_RMSE_j_.append([TI_RMSE_j_RSD_Ref, TI_RMSE_jp5_RSD_Ref])
    else:
        print("Warning: No RSD TI. Cannot compute error stats for this category")

    # get the bin wise stats for DIFFERENCE and ERROR and RMSE between RSD and Ref TI (CORRECTED)
    if "corrTI_RSD_TI" in inputdata.columns:
        inputdata["TI_diff_adjTI_RSD_Ref"] = (
            inputdata["corrTI_RSD_TI"] - inputdata["Ref_TI"]
        )  # caliculating the diff in ti for each timestamp
        inputdata["TI_error_adjTI_RSD_Ref"] = (
            inputdata["TI_diff_adjTI_RSD_Ref"] / inputdata["Ref_TI"]
        )  # calculating the error for each timestamp (diff normalized to ref_TI)
        inputdata["TI_SquaredDiff_adjTI_RSD_Ref"] = (
            inputdata["TI_diff_adjTI_RSD_Ref"] * inputdata["TI_diff_adjTI_RSD_Ref"]
        )  # calculating squared diff each Timestamp
        TI_MBE_j_adjTI_RSD_Ref, TI_MBE_jp5_adjTI_RSD_Ref = get_stats_per_WSbin(
            inputdata, "TI_error_adjTI_RSD_Ref"
        )
        TI_Diff_j_adjTI_RSD_Ref, TI_Diff_jp5_adjTI_RSD_Ref = get_stats_per_WSbin(
            inputdata, "TI_diff_adjTI_RSD_Ref"
        )
        TI_RMSE_j_adjTI_RSD_Ref, TI_RMSE_jp5_adjTI_RSD_Ref = get_RMSE_per_WSbin(
            inputdata, "TI_SquaredDiff_adjTI_RSD_Ref"
        )

        TI_MBE_j_.append([TI_MBE_j_adjTI_RSD_Ref, TI_MBE_jp5_adjTI_RSD_Ref])
        TI_Diff_j_.append([TI_Diff_j_adjTI_RSD_Ref, TI_Diff_jp5_adjTI_RSD_Ref])
        TI_RMSE_j_.append([TI_RMSE_j_adjTI_RSD_Ref, TI_RMSE_jp5_adjTI_RSD_Ref])
    else:
        print(
            "Warning: No corrected RSD TI. Cannot compute error stats for this category"
        )

    # get the bin wise stats for DIFFERENCE and ERROR and RMSE between redundant anemometer and Ref TI
    if "Ane2_TI" in inputdata.columns:
        inputdata["Ane2_TI"] = inputdata["Ane2_TI"].astype(float)
        inputdata["TI_diff_Ane2_Ref"] = inputdata["Ane2_TI"] - inputdata["Ref_TI"]
        inputdata["TI_error_Ane2_Ref"] = (
            inputdata["TI_diff_Ane2_Ref"] / inputdata["Ref_TI"]
        )
        inputdata["TI_SquaredDiff_Ane2_Ref"] = (
            inputdata["TI_diff_Ane2_Ref"] * inputdata["TI_diff_Ane2_Ref"]
        )
        TI_MBE_j_Ane2_Ref, TI_MBE_jp5_Ane2_Ref = get_stats_per_WSbin(
            inputdata, "TI_error_Ane2_Ref"
        )
        TI_Diff_j_Ane2_Ref, TI_Diff_jp5_Ane2_Ref = get_stats_per_WSbin(
            inputdata, "TI_diff_Ane2_Ref"
        )
        TI_RMSE_j_Ane2_ref, TI_RMSE_jp5_Ane2_ref = get_RMSE_per_WSbin(
            inputdata, "TI_SquaredDiff_Ane2_Ref"
        )
        TI_MBE_j_.append([TI_MBE_j_Ane2_Ref, TI_MBE_jp5_Ane2_Ref])
        TI_Diff_j_.append([TI_Diff_j_Ane2_Ref, TI_Diff_jp5_Ane2_Ref])
        TI_RMSE_j_.append([TI_RMSE_j_Ane2_ref, TI_RMSE_jp5_Ane2_ref])
    else:
        print("Warning: No Ane2 TI. Cannot compute error stats for this category")

    return TI_MBE_j_, TI_Diff_j_, TI_RMSE_j_, RepTI_MBE_j_, RepTI_Diff_j_, RepTI_RMSE_j_


def get_TI_Diff_r(inputdata):
    """
    get TI abs difference by reference TI bin
    """

    TI_Diff_r_ = []
    RepTI_Diff_r_ = []

    # get the bin wise stats for DIFFERENCE between RSD and Ref TI (UNCORRECTED)
    if "RSD_TI" in inputdata.columns:
        inputdata["RSD_TI"] = inputdata["RSD_TI"].astype(float)
        inputdata["Ref_TI"] = inputdata["Ref_TI"].astype(float)
        inputdata["TI_diff_RSD_Ref"] = (
            inputdata["RSD_TI"] - inputdata["Ref_TI"]
        )  # caliculating the diff in ti for each timestamp
        TI_Diff_r_RSD_Ref = get_stats_per_TIbin(inputdata, "TI_diff_RSD_Ref")
        TI_Diff_r_.append([TI_Diff_r_RSD_Ref])
    else:
        print("Warning: No RSD TI. Cannot compute error stats for this category")

    # get the bin wise stats for DIFFERENCE between RSD and Ref TI (CORRECTED)
    if "corrTI_RSD_TI" in inputdata.columns:
        inputdata["TI_error_adjTI_RSD_Ref"] = (
            inputdata["TI_diff_adjTI_RSD_Ref"] / inputdata["Ref_TI"]
        )  # calculating the error for each timestamp (diff normalized to ref_TI)
        TI_Diff_r_adjTI_RSD_Ref = get_stats_per_TIbin(
            inputdata, "TI_diff_adjTI_RSD_Ref"
        )
        TI_Diff_r_.append([TI_Diff_r_adjTI_RSD_Ref])
    else:
        print(
            "Warning: No corrected RSD TI. Cannot compute error stats for this category"
        )

    # get the bin wise stats for DIFFERENCE and ERROR and RMSE between redundant anemometer and Ref TI
    if "Ane2_TI" in inputdata.columns:
        inputdata["Ane2_TI"] = inputdata["Ane2_TI"].astype(float)
        inputdata["TI_diff_Ane2_Ref"] = inputdata["Ane2_TI"] - inputdata["Ref_TI"]
        TI_Diff_r_Ane2_Ref = get_stats_per_TIbin(inputdata, "TI_diff_Ane2_Ref")
        TI_Diff_r_.append([TI_Diff_r_Ane2_Ref])
    else:
        print("Warning: No Ane2 TI. Cannot compute error stats for this category")

    return TI_Diff_r_, RepTI_Diff_r_


def get_TI_bybin(inputdata):
    results = []

    if "RSD_TI" in inputdata.columns:
        RSD_TI_j, RSD_TI_jp5 = get_stats_per_WSbin(inputdata, "RSD_TI")
        results.append([RSD_TI_j, RSD_TI_jp5])
    else:
        results.append(["NaN", "NaN"])

    Ref_TI_j, Ref_TI_jp5 = get_stats_per_WSbin(inputdata, "Ref_TI")
    results.append([Ref_TI_j, Ref_TI_jp5])

    if (
        "corrTI_RSD_TI" in inputdata.columns
    ):  # this is checking if corrected TI windspeed is present in the input data and using that for getting the results.
        corrTI_RSD_TI_j, corrTI_RSD_TI_jp5 = get_stats_per_WSbin(
            inputdata, "corrTI_RSD_TI"
        )
        results.append([corrTI_RSD_TI_j, corrTI_RSD_TI_jp5])
    else:
        results.append(pd.DataFrame(["NaN", "NaN"]))

    # get the bin wise stats for both diff and error between RSD corrected for ws and Ref
    if "Ane2_TI" in inputdata.columns:
        Ane2_TI_j, Ane2_TI_jp5 = get_stats_per_WSbin(inputdata, "Ane2_TI")
        results.append([Ane2_TI_j, Ane2_TI_jp5])
    else:
        results.append(pd.DataFrame(["NaN", "NaN"]))

    return results


def get_TI_byTIrefbin(inputdata):
    results = []

    if "RSD_TI" in inputdata.columns:
        RSD_TI_r = get_stats_per_TIbin(inputdata, "RSD_TI")
        results.append([RSD_TI_r])
    else:
        results.append(["NaN"])

    if (
        "corrTI_RSD_TI" in inputdata.columns
    ):  # this is checking if corrected TI is present
        corrTI_RSD_TI_r = get_stats_per_TIbin(inputdata, "corrTI_RSD_TI")
        results.append([corrTI_RSD_TI_r])
    else:
        results.append(pd.DataFrame(["NaN"]))

    # get the bin wise stats for both diff and error between RSD corrected for ws and Ref
    if "Ane2_TI" in inputdata.columns:
        Ane2_TI_r = get_stats_per_TIbin(inputdata, "Ane2_TI")
        results.append([Ane2_TI_r])
    else:
        results.append(pd.DataFrame(["NaN"]))

    return results


def get_stats_inBin(inputdata_m, start, end):
    # this was discussed in the meeting , but the results template didn't ask for this.
    inputdata = inputdata_m.loc[
        (inputdata_m["Ref_WS"] > start) & (inputdata_m["Ref_WS"] <= end)
    ].copy()
    _adjuster_stats = Adjustments()

    if "RSD_TI" in inputdata.columns:
        inputdata["TI_diff_RSD_Ref"] = (
            inputdata["RSD_TI"] - inputdata["Ref_TI"]
        )  # caliculating the diff in ti for each timestamp
        inputdata["TI_error_RSD_Ref"] = (
            inputdata["TI_diff_RSD_Ref"] / inputdata["Ref_TI"]
        )  # calculating the error for each timestamp

    if "RSD_TI" in inputdata.columns:
        TI_error_RSD_Ref_Avg = inputdata["TI_error_RSD_Ref"].mean()
        TI_error_RSD_Ref_Std = inputdata["TI_error_RSD_Ref"].std()
        TI_diff_RSD_Ref_Avg = inputdata["TI_diff_RSD_Ref"].mean()
        TI_diff_RSD_Ref_Std = inputdata["TI_diff_RSD_Ref"].std()
    else:
        TI_error_RSD_Ref_Avg = None
        TI_error_RSD_Ref_Std = None
        TI_diff_RSD_Ref_Avg = None
        TI_diff_RSD_Ref_Std = None

    # RSD V Reference
    if "RSD_TI" in inputdata.columns:
        modelResults = _adjuster_stats.get_regression(
            inputdata["Ref_TI"], inputdata["RSD_TI"]
        )
        rmse = modelResults[5]
        slope = modelResults[0]
        offset = modelResults[1]
        r2 = modelResults[2]
    else:
        rmse = None
        slope = None
        offset = None
        r2 = None

    results = pd.DataFrame(
        [
            TI_error_RSD_Ref_Avg,
            TI_error_RSD_Ref_Std,
            TI_diff_RSD_Ref_Avg,
            TI_diff_RSD_Ref_Std,
            slope,
            offset,
            rmse,
            r2,
        ],
        columns=["RSD_Ref"],
    )

    if (
        "corrTI_RSD_TI" in inputdata.columns
    ):  # this is checking if corrected TI windspeed is present in the input data and using that for getting the results.
        # Cor RSD vs Reg RSD
        inputdata["TI_diff_adjTI_RSD_Ref"] = (
            inputdata["corrTI_RSD_TI"] - inputdata["Ref_TI"]
        )  # caliculating the diff in ti for each timestamp
        inputdata["TI_error_adjTI_RSD_Ref"] = (
            inputdata["TI_diff_adjTI_RSD_Ref"] / inputdata["Ref_TI"]
        )  # calculating the error for each timestamp
        TI_error_adjTI_RSD_Ref_Avg = inputdata["TI_error_adjTI_RSD_Ref"].mean()
        TI_error_adjTI_RSD_Ref_Std = inputdata["TI_error_adjTI_RSD_Ref"].std()
        TI_diff_adjTI_RSD_Ref_Avg = inputdata["TI_diff_adjTI_RSD_Ref"].mean()
        TI_diff_adjTI_RSD_Ref_Std = inputdata["TI_diff_adjTI_RSD_Ref"].std()

        modelResults = _adjuster_stats.get_regression(
            inputdata["corrTI_RSD_TI"], inputdata["Ref_TI"]
        )
        rmse = modelResults[5]
        slope = modelResults[0]
        offset = modelResults[1]
        r2 = modelResults[2]

        results["CorrTI_RSD_Ref"] = [
            TI_error_adjTI_RSD_Ref_Avg,
            TI_error_adjTI_RSD_Ref_Std,
            TI_diff_adjTI_RSD_Ref_Avg,
            TI_diff_adjTI_RSD_Ref_Std,
            slope,
            offset,
            rmse,
            r2,
        ]
    else:
        results["CorrTI_RSD_Ref"] = [
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
            "NaN",
        ]

    # anem 2 vs ref
    if "Ane2_TI" in inputdata.columns:
        inputdata["TI_diff_Ane2_Ref"] = (
            inputdata["Ane2_TI"] - inputdata["Ref_TI"]
        )  # caliculating the diff in ti for each timestamp
        inputdata["TI_error_Ane2_Ref"] = (
            inputdata["TI_diff_Ane2_Ref"] / inputdata["Ref_TI"]
        )  # calculating the error for each timestamp
        TI_error_Ane2_Ref_Avg = inputdata["TI_error_Ane2_Ref"].mean()
        TI_error_Ane2_Ref_Std = inputdata["TI_error_Ane2_Ref"].std()
        TI_diff_Ane2_Ref_Avg = inputdata["TI_diff_Ane2_Ref"].mean()
        TI_diff_Ane2_Ref_Std = inputdata["TI_diff_Ane2_Ref"].std()

        modelResults = _adjuster_stats.get_regression(
            inputdata["Ane2_TI"], inputdata["Ref_TI"]
        )
        rmse = modelResults[5]
        slope = modelResults[0]
        offset = modelResults[1]
        r2 = modelResults[2]

        results["Ane2_Ref"] = [
            TI_error_Ane2_Ref_Avg,
            TI_error_Ane2_Ref_Std,
            TI_diff_Ane2_Ref_Avg,
            TI_diff_Ane2_Ref_Std,
            slope,
            offset,
            rmse,
            r2,
        ]
    else:
        results["Ane2_Ref"] = ["NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]

    results.index = [
        "TI_error_mean",
        "TI_error_std",
        "TI_diff_mean",
        "TI_diff_std",
        "Slope",
        "Offset",
        "RMSE",
        "R-squared",
    ]

    return results.T  # T(ranspose) so that reporting looks good.


def get_description_stats(inputdata):
    totalstats = get_stats_inBin(inputdata, 1.75, 20)
    belownominal = get_stats_inBin(inputdata, 1.75, 11.5)
    abovenominal = get_stats_inBin(inputdata, 10, 20)
    return totalstats, belownominal, abovenominal


def get_distribution_test_results(inputdata_adj, ref_col, test_col, subset=False):
    """
    performs statistical tests on results. Kolmogorov-Smirnov test. The K-S statistical test is a nonparametric
    test used to quantify the distance between the empirical distribution functions of two samples. It is
    sensitive to differences in both location and shape of the empirical cumulative distribution functions of the two
    samples, and thus acts as a stand-alone detection of statistical difference.
    """
    # K-S test to compare samples from two different sensors
    import numpy as np
    from scipy import stats

    if ref_col in inputdata_adj.columns and test_col in inputdata_adj.columns:
        if isinstance(subset, pd.DataFrame):
            a = np.array(inputdata_adj[ref_col])
            b = np.array(subset[test_col])
        else:
            a = np.array(inputdata_adj[ref_col])
            b = np.array(inputdata_adj[test_col])
        distribution_test_results = stats.ks_2samp(a, b)
    else:
        distribution_test_results = stats.ks_2samp([np.NaN, np.NaN], [np.NaN, np.NaN])

    return distribution_test_results


class StatResult:
    pass


def Dist_stats(inputdata_adj, Timestamps, adjustmentName):
    """
    test all relevant chunks of data
    """

    distribution_test_results = pd.DataFrame()
    sampleWindow_test_results = pd.DataFrame()
    sampleWindow_test_results_new = pd.DataFrame()

    # full test data
    names = []
    KStest_stat = []
    p_value = []

    # for subsets
    idx = []
    T = []
    p_value_T = []
    ref_list = []
    test_list = []

    pairs = [
        ["Ref_WS", "Ane2_WS"],
        ["Ref_WS", "RSD_WS"],
        ["Ref_SD", "Ane2_SD"],
        ["Ref_SD", "RSD_SD"],
        ["Ref_TI", "Ane2_TI"],
        ["Ref_TI", "RSD_TI"],
        ["Ane_WS_Ht1", "RSD_WS_Ht1"],
        ["Ane_WS_Ht2", "RSD_WS_Ht2"],
        ["Ane_WS_Ht3", "RSD_WS_Ht3"],
        ["Ane_WS_Ht4", "RSD_WS_Ht4"],
        ["Ane_SD_Ht1", "RSD_SD_Ht1"],
        ["Ane_SD_Ht2", "RSD_SD_Ht2"],
        ["Ane_SD_Ht3", "RSD_SD_Ht3"],
        ["Ane_SD_Ht4", "RSD_SD_Ht4"],
        ["Ane_TI_Ht1", "RSD_TI_Ht1"],
        ["Ane_TI_Ht2", "RSD_TI_Ht2"],
        ["Ane_TI_Ht3", "RSD_TI_Ht3"],
        ["Ane_TI_Ht4", "RSD_TI_Ht4"],
        ["Ref_RepTI", "Ane2_RepTI"],
        ["Ref_RepTI", "RSD_RepTI"],
        ["Ane_RepTI_Ht1", "RSD_RepTI_Ht1"],
        ["Ane_RepTI_Ht2", "RSD_RepTI_Ht2"],
        ["Ane_RepTI_Ht3", "RSD_RepTI_Ht3"],
        ["Ane_RepTI_Ht4", "RSD_RepTI_Ht4"],
        ["Ref_TI", "corrTI_RSD_TI"],
        ["Ref_RepTI", "corrRepTI_RSD_RepTI"],
        ["Ane_TI_Ht1", "corrTI_RSD_TI_Ht1"],
        ["Ane_RepTI_Ht1", "corrRepTI_RSD_RepTI_Ht1"],
        ["Ane_TI_Ht2", "corrTI_RSD_TI_Ht2"],
        ["Ane_RepTI_Ht2", "corrRepTI_RSD_RepTI_Ht2"],
        ["Ane_TI_Ht3", "corrTI_RSD_TI_Ht3"],
        ["Ane_RepTI_Ht3", "corrRepTI_RSD_RepTI_Ht3"],
        ["Ane_TI_Ht4", "corrTI_RSD_TI_Ht4"],
        ["Ane_RepTI_Ht4", "corrRepTI_RSD_RepTI_Ht4"],
    ]

    b1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    Nsamples_90days = 12960
    inputdata_adj = inputdata_adj.reset_index()
    t_length = len(inputdata_adj) - Nsamples_90days

    for p in pairs:
        ref = p[0]
        test = p[1]
        if ref == "Ref_WS" or ref == "Ref_SD" or ref == "Ref_TI":
            idx.append("0:end")
            T.append("all_data")
            chunk = inputdata_adj
            results = get_distribution_test_results(
                inputdata_adj, ref, test, subset=chunk
            )
            p_value_T.append(results.pvalue)
            if len(inputdata_adj) > Nsamples_90days:
                for i in range(0, t_length, 60):  # shift by 60
                    nn = str(str(i) + "_" + str(12960 + i))
                    tt = (
                        list(Timestamps)[i]
                        + "_to_"
                        + list(Timestamps)[Nsamples_90days + i]
                    )
                    idx.append(nn)
                    T.append(tt)
                    chunk = inputdata_adj[i : 12960 + i]
                    results = get_distribution_test_results(
                        inputdata_adj, ref, test, subset=chunk
                    )
                    p_value_T.append(results.pvalue)
            sampleWindow_test_results_new["idx"] = idx
            sampleWindow_test_results_new[str("chunk" + "_" + ref + "_" + test)] = T
            sampleWindow_test_results_new[
                str("p_score" + "_" + ref + "_" + test)
            ] = p_value_T
            sampleWindow_test_results = pd.concat(
                [sampleWindow_test_results, sampleWindow_test_results_new], axis=1
            )
            sampleWindow_test_results_new = pd.DataFrame()
            ref_list.append(ref)
            ref_list.append(ref)
            ref_list.append(ref)
            test_list.append(test)
            test_list.append(test)
            test_list.append(test)

        idx = []
        T = []
        p_value_T = []

        if ref in inputdata_adj.columns and test in inputdata_adj.columns:
            results = get_distribution_test_results(inputdata_adj, ref, test)
            names.append(str(p[0] + "_VS_" + p[1]))
            KStest_stat.append(results.statistic)
            p_value.append(results.pvalue)
            for bin in b1:
                binsubset = inputdata_adj[inputdata_adj["bins"] == bin]
                if len(binsubset) == 0:
                    names.append(str(p[0] + "_VS_" + p[1]))
                    KStest_stat.append(None)
                    p_value.append(None)
                else:
                    results = get_distribution_test_results(binsubset, ref, test)
                    names.append(str("bin_" + str(bin) + "_" + p[0] + "_VS_" + p[1]))
                    KStest_stat.append(results.statistic)
                    p_value.append(results.pvalue)
        else:
            names.append(str(p[0] + "_VS_" + p[1]))
            KStest_stat.append(None)
            p_value.append(None)
    distribution_test_results[str("Test Name" + "_" + adjustmentName)] = names
    distribution_test_results[
        str("KS test statistics" + "_" + "all_data" + "_" + adjustmentName)
    ] = KStest_stat
    distribution_test_results[
        str("p_value" + "_" + "all_data" + "_" + adjustmentName)
    ] = p_value

    if len(sampleWindow_test_results) > 1:
        pick = [
            c for c in sampleWindow_test_results.columns.to_list() if "p_score" in c
        ]
        plist = sampleWindow_test_results[pick]
        plist = plist[1:]
        pick2 = [c for c in sampleWindow_test_results.columns.to_list() if "idx" in c]
        idxlist = sampleWindow_test_results["idx"]
        cols_idx = idxlist.columns.to_list()
        cols_idx[0] = "idx_all"
        idxlist.columns = cols_idx
        cols_plist = plist.columns.to_list()
        idx_data = list(idxlist["idx_all"])[1:]
        min_idx = []
        median_idx = []
        max_idx = []
        for c in cols_plist:
            ix = cols_plist.index(c)
            r = ref_list[ix]
            t = test_list[ix]
            cols_plist = plist.columns.to_list()
            cp = cols_plist[ix]
            minx = plist[cp].idxmin()
            if len(plist) % 2 == 0:
                medVal = plist[cp][:-1].median()
            else:
                medVal = plist[cp].median()
            try:
                medx = list(plist[cp]).index(medVal)
            except:
                temp = list(plist[cp])
                del temp[-1]
                import statistics

                medVal = statistics.median(temp)
                medx = list(plist[cp]).index(medVal[0])
            maxx = plist[cp].idxmax()
            minInt = idx_data[minx]
            medInt = idx_data[medx]
            maxInt = idx_data[maxx]
            min_idx.append([minInt])
            median_idx.append([medInt])
            max_idx.append([maxInt])
            # fig = plt.figure()
            # plt.plot(idx_data[1:],plist[cp])
            # plotName = str(r + '_vs_' + t +  '.png')
            # fig.savefig(plotName)

    return distribution_test_results, sampleWindow_test_results


def get_representative_TI_15mps(inputdata):
    # this is the represetative TI, this is currently only done at a 1m/s bins not sure if this needs to be on .5m/s
    # TODO: find out if this needs to be on the 1m/s bin or the .5m/s bin
    inputdata_TI15 = inputdata[inputdata["bins"] == 15]
    listofcols = ["Ref_TI"]
    if "RSD_TI" in inputdata.columns:
        listofcols.append("RSD_TI")
    if "Ane2_TI" in inputdata.columns:
        listofcols.append("Ane2_TI")
    if "corrTI_RSD_WS" in inputdata.columns:
        listofcols.append("corrTI_RSD_WS")
    results = inputdata_TI15[listofcols].describe()
    results.loc["Rep_TI", :] = results.loc["mean"] + 1.28 * results.loc["std"]
    results = results.loc[["mean", "std", "Rep_TI"], :].T
    results.columns = ["mean_15mps", "std_15mps", "Rep_TI"]
    return results
