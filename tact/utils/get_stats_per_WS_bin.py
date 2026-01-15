import pandas as pd

def get_stats_per_WS_bin(inputdata, column):
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