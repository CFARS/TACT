try:
    from TACT import logger
except ImportError:
    pass

import numpy as np
import pandas as pd
from TACT.computation.adjustments import Adjustments, empirical_stdAdjustment


def perform_G_C_adjustment(inputdata):
    """
    Note: comprehensive empirical adjustment from a dozen locations. Focuses on std. deviation
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
        inputdata_test = inputdata.copy()
        inputdata = False
    else:
        inputdata_test, results = empirical_stdAdjustment(
            inputdata,
            results,
            "Ref_TI",
            "RSD_TI",
            "Ref_SD",
            "RSD_SD",
            "Ref_WS",
            "RSD_WS",
        )
        if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(
                inputdata_test,
                results,
                "Ane_TI_Ht1",
                "RSD_TI_Ht1",
                "Ane_SD_Ht1",
                "RSD_SD_Ht1",
                "Ane_WS_Ht1",
                "RSD_WS_Ht1",
            )
        if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(
                inputdata_test,
                results,
                "Ane_TI_Ht2",
                "RSD_TI_Ht2",
                "Ane_SD_Ht2",
                "RSD_SD_Ht2",
                "Ane_WS_Ht2",
                "RSD_WS_Ht2",
            )
        if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(
                inputdata_test,
                results,
                "Ane_TI_Ht3",
                "RSD_TI_Ht3",
                "Ane_SD_Ht3",
                "RSD_SD_Ht3",
                "Ane_WS_Ht3",
                "RSD_WS_Ht3",
            )
        if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(
                inputdata_test,
                results,
                "Ane_TI_Ht4",
                "RSD_TI_Ht4",
                "Ane_SD_Ht4",
                "RSD_SD_Ht4",
                "Ane_WS_Ht4",
                "RSD_WS_Ht4",
            )
    results["adjustment"] = ["G-C"] * len(results)
    results = results.drop(columns=["sensor", "height"])
    m = np.NaN
    c = np.NaN

    return inputdata_test, results, m, c
# class Perform_G_C_Adjustment(AdjustmentsParent):
#     """
#     Note: comprehensive empirical adjustment from a dozen locations. Focuses on std. deviation
#     """
#     def __init__(self, inputdata):
#         logger.debug("Initialized Perform_G_C_Adjustment class")
    
#         super().__init__(inputdata)
    
#     results = pd.DataFrame(
#         columns=[
#             "sensor",
#             "height",
#             "adjustment",
#             "m",
#             "c",
#             "rsquared",
#             "difference",
#             "mse",
#             "rmse",
#         ]
#     )
#     adj = Adjustments()

#     if inputdata.empty or len(inputdata) < 2:
#         results = adj.post_adjustment_stats([None], results, "Ref_TI", "adjTI_RSD_TI")
#         if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
#             results = adj.post_adjustment_stats(
#                 [None], results, "Ane_TI_Ht1", "adjTI_RSD_TI_Ht1"
#             )
#         if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
#             results = adj.post_adjustment_stats(
#                 [None], results, "Ane_TI_Ht2", "adjTI_RSD_TI_Ht2"
#             )
#         if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
#             results = adj.post_adjustment_stats(
#                 [None], results, "Ane_TI_Ht3", "adjTI_RSD_TI_Ht3"
#             )
#         if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
#             results = adj.post_adjustment_stats(
#                 [None], results, "Ane_TI_Ht4", "adjTI_RSD_TI_Ht4"
#             )
#         m = np.NaN
#         c = np.NaN
#         inputdata_test = inputdata.copy()
#         inputdata = False
#     else:
#         inputdata_test, results = empirical_stdAdjustment(
#             inputdata,
#             results,
#             "Ref_TI",
#             "RSD_TI",
#             "Ref_SD",
#             "RSD_SD",
#             "Ref_WS",
#             "RSD_WS",
#         )
#         if "Ane_TI_Ht1" in inputdata.columns and "RSD_TI_Ht1" in inputdata.columns:
#             inputdata_test, results = empirical_stdAdjustment(
#                 inputdata_test,
#                 results,
#                 "Ane_TI_Ht1",
#                 "RSD_TI_Ht1",
#                 "Ane_SD_Ht1",
#                 "RSD_SD_Ht1",
#                 "Ane_WS_Ht1",
#                 "RSD_WS_Ht1",
#             )
#         if "Ane_TI_Ht2" in inputdata.columns and "RSD_TI_Ht2" in inputdata.columns:
#             inputdata_test, results = empirical_stdAdjustment(
#                 inputdata_test,
#                 results,
#                 "Ane_TI_Ht2",
#                 "RSD_TI_Ht2",
#                 "Ane_SD_Ht2",
#                 "RSD_SD_Ht2",
#                 "Ane_WS_Ht2",
#                 "RSD_WS_Ht2",
#             )
#         if "Ane_TI_Ht3" in inputdata.columns and "RSD_TI_Ht3" in inputdata.columns:
#             inputdata_test, results = empirical_stdAdjustment(
#                 inputdata_test,
#                 results,
#                 "Ane_TI_Ht3",
#                 "RSD_TI_Ht3",
#                 "Ane_SD_Ht3",
#                 "RSD_SD_Ht3",
#                 "Ane_WS_Ht3",
#                 "RSD_WS_Ht3",
#             )
#         if "Ane_TI_Ht4" in inputdata.columns and "RSD_TI_Ht4" in inputdata.columns:
#             inputdata_test, results = empirical_stdAdjustment(
#                 inputdata_test,
#                 results,
#                 "Ane_TI_Ht4",
#                 "RSD_TI_Ht4",
#                 "Ane_SD_Ht4",
#                 "RSD_SD_Ht4",
#                 "Ane_WS_Ht4",
#                 "RSD_WS_Ht4",
#             )
#     results["adjustment"] = ["G-C"] * len(results)
#     results = results.drop(columns=["sensor", "height"])
#     m = np.NaN
#     c = np.NaN

#     return inputdata_test, results, m, c
