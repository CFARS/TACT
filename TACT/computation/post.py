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
import sys
from .adjustments import Adjustments


def post_adjustment_stats(inputdata, results, ref_col, TI_col):

    if isinstance(inputdata, pd.DataFrame):
        fillEmpty = False
        if ref_col in inputdata.columns and TI_col in inputdata.columns:
            _adjuster_post_adjustment = Adjustments()
            model_adjTI = _adjuster_post_adjustment.get_regression(
                inputdata[ref_col], inputdata[TI_col]
            )
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
