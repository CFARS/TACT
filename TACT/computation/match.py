from tkinter import W
import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments


def perform_match_input(inputdata):
    """
    correct the TI inputs separately
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
    else:
        full = pd.DataFrame()
        full["Ref_WS"] = inputdata_test["Ref_WS"]
        full["RSD_WS"] = inputdata_test["RSD_WS"]
        full = full.dropna()
        if len(full) < 10:
            results = adj.post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
            m = np.NaN
            c = np.NaN
        else:
            WS_output = hist_match(inputdata_train, inputdata_test, "Ref_WS", "RSD_WS")
            SD_output = hist_match(inputdata_train, inputdata_test, "Ref_SD", "RSD_SD")
            inputdata_test["adjTI_RSD_TI"] = SD_output / WS_output
            results = adj.post_adjustment_stats(
                inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
            )
        if (
            "Ane_WS_Ht1" in inputdata.columns
            and "Ane_SD_Ht1" in inputdata.columns
            and "RSD_WS_Ht1" in inputdata.columns
            and "RSD_SD_Ht1" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht1"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht1"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_WS_Ht1", "RSD_WS_Ht1"
                )
                SD_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_SD_Ht1", "RSD_SD_Ht1"
                )
                inputdata_test["adjTI_RSD_TI_Ht1"] = SD_output / WS_output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
        if (
            "Ane_WS_Ht2" in inputdata.columns
            and "Ane_SD_Ht2" in inputdata.columns
            and "RSD_WS_Ht2" in inputdata.columns
            and "RSD_SD_Ht2" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht2"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht2"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_WS_Ht2", "RSD_WS_Ht2"
                )
                SD_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_SD_Ht2", "RSD_SD_Ht2"
                )
                inputdata_test["adjTI_RSD_TI_Ht2"] = SD_output / WS_output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
        if (
            "Ane_WS_Ht3" in inputdata.columns
            and "Ane_SD_Ht3" in inputdata.columns
            and "RSD_WS_Ht3" in inputdata.columns
            and "RSD_SD_Ht3" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht3"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht3"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_WS_Ht3", "RSD_WS_Ht3"
                )
                SD_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_SD_Ht3", "RSD_SD_Ht3"
                )
                inputdata_test["adjTI_RSD_TI_Ht3"] = SD_output / WS_output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
        if (
            "Ane_WS_Ht4" in inputdata.columns
            and "Ane_SD_Ht4" in inputdata.columns
            and "RSD_WS_Ht4" in inputdata.columns
            and "RSD_SD_Ht4" in inputdata.columns
        ):
            full = pd.DataFrame()
            full["Ref_WS"] = inputdata_test["Ane_WS_Ht4"]
            full["RSD_WS"] = inputdata_test["RSD_WS_Ht4"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_WS_Ht4", "RSD_WS_Ht4"
                )
                SD_output = hist_match(
                    inputdata_train, inputdata_test, "Ane_SD_Ht4", "RSD_SD_Ht4"
                )
                inputdata_test["adjTI_RSD_TI_Ht4"] = SD_output / WS_output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    results["adjustment"] = ["SS-Match_input"] * len(results)
    results = results.drop(columns=["sensor", "height"])
    return inputdata_test, results


def perform_match(inputdata):
    # manipulate histogram to match template histogram (ref) - virtually a look-up table.
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
    else:
        full = pd.DataFrame()
        full["Ref_TI"] = inputdata_test["Ref_TI"]
        full["RSD_TI"] = inputdata_test["RSD_TI"]
        full = full.dropna()
        if len(full) < 10:
            results = adj.post_adjustment_stats(
                [None], results, "Ref_TI", "adjTI_RSD_TI"
            )
            m = np.NaN
            c = np.NaN
        else:
            output = hist_match(inputdata_train, inputdata_test, "Ref_TI", "RSD_TI")
            inputdata_test["adjTI_RSD_TI"] = output
            results = adj.post_adjustment_stats(
                inputdata_test, results, "Ref_TI", "adjTI_RSD_TI"
            )
        if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht1"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht1"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(
                    inputdata_train, inputdata_test, "Ane_TI_Ht1", "RSD_TI_Ht1"
                )
                inputdata_test["adjTI_RSD_TI_Ht1"] = output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
                )
        if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht2"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht2"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(
                    inputdata_train, inputdata_test, "Ane_TI_Ht2", "RSD_TI_Ht2"
                )
                inputdata_test["adjTI_RSD_TI_Ht2"] = output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
                )
        if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht3"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht3"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(
                    inputdata_train, inputdata_test, "Ane_TI_Ht3", "RSD_TI_Ht3"
                )
                inputdata_test["adjTI_RSD_TI_Ht3"] = output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
                )
        if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            full = pd.DataFrame()
            full["Ref_TI"] = inputdata_test["Ane_TI_Ht4"]
            full["RSD_TI"] = inputdata_test["RSD_TI_Ht4"]
            full = full.dropna()
            if len(full) < 10:
                results = adj.post_adjustment_stats(
                    [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(
                    inputdata_train, inputdata_test, "Ane_TI_Ht4", "RSD_TI_Ht4"
                )
                inputdata_test["adjTI_RSD_TI_Ht4"] = output
                results = adj.post_adjustment_stats(
                    inputdata_test, results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
                )

    import matplotlib.pyplot as plt

    #    plt.plot(inputdata_test['Ref_TI'])
    #    plt.plot(inputdata_test['RSD_TI'])
    #    plt.plot(inputdata_test['adjTI_RSD_TI'])

    #    plt.scatter(inputdata_test['Ref_TI'], inputdata_test['RSD_TI'], label='RefvsRSD')
    #    plt.scatter(inputdata_test['Ref_TI'], inputdata_test['adjTI_RSD_TI'], label='RefvsCorrectedRSD')
    #    plt.scatter(inputdata_test['Ref_TI'], inputdata_test['Ane2_TI'], label='RefvsRedundant')
    #    plt.legend()
    #    plt.show()
    results["adjustment"] = ["SS-Match"] * len(results)
    results = results.drop(columns=["sensor", "height"])
    return inputdata_test, results


def hist_match(inputdata_train, inputdata_test, refCol, testCol):

    test1 = inputdata_test[refCol].copy
    source = inputdata_test[refCol].copy().dropna()
    template = inputdata_train[testCol].copy().dropna()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)
    n_bins = 200
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    #    import matplotlib.pyplot as plt

    #    plt.plot(s_quantiles, label='source')
    #    plt.plot(t_quantiles, label='template')
    #    plt.legend()
    #    plt.show()
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    output = interp_t_values[bin_idx]

    # test number 2
    imhist_source, bins_source = np.histogram(source, n_bins, density=True)
    cdf_source = imhist_source.cumsum()  # cumulative distribution function
    cdf_source = n_bins * cdf_source / cdf_source[-1]  # normalize

    imhist_template, bins_template = np.histogram(template, n_bins, density=True)
    cdf_template = imhist_template.cumsum()  # cumulative distribution function
    cdf_template = n_bins * cdf_template / cdf_template[-1]  # normalize

    im2 = np.interp(source, bins_template[:-1], cdf_source)
    output = np.interp(im2, cdf_template, bins_template[:-1])

    #    plt.plot(cdf_source,label='source')
    #    plt.plot(cdf_template,label='template')

    #    plt.legend()
    #    plt.show()

    output_df = source
    output_df = output_df.to_frame()
    output_df["output"] = output
    not_outs = output_df.columns.to_list()
    not_outs = [i for i in not_outs if i != "output"]
    output_df = output_df.drop(columns=not_outs)

    res = inputdata_test.join(output_df, how="left")
    output_res = res["output"].values

    return output_res
