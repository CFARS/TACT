import pandas as pd
from tact.utils.get_stats_per_WS_bin import get_stats_per_WS_bin


def get_TI_by_bin(inputdata, parameters):
    results = []

    if "RSD_TI" in inputdata.columns:
        RSD_TI_j, RSD_TI_jp5 = get_stats_per_WS_bin(inputdata, "RSD_TI")
        results.append([RSD_TI_j, RSD_TI_jp5])
    else:
        results.append(["NaN", "NaN"])

    Ref_TI_j, Ref_TI_jp5 = get_stats_per_WS_bin(inputdata, "Ref_TI")
    results.append([Ref_TI_j, Ref_TI_jp5])

    if (
        "adjTI_RSD_TI" in inputdata.columns
    ):  # this is checking if corrected TI windspeed is present in the input data and using that for getting the results.
        adjTI_RSD_TI_j, adjTI_RSD_TI_jp5 = get_stats_per_WS_bin(
            inputdata, "adjTI_RSD_TI"
        )
        results.append([adjTI_RSD_TI_j, adjTI_RSD_TI_jp5])
    else:
        results.append(pd.DataFrame(["NaN", "NaN"]))
