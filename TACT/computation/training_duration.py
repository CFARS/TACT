from TACT.computation.adjustments import train_test_split
from TACT.computation.adjustments import quick_metrics

import numpy as np
import pandas as pd
import sys

def test_train_split_convergence(inputdata, config):
    """incrementally increase the % parameter for test train split to check for convergence

    Parameters
    ----------
    x : type
        description
    y : type
        description

    Returns
    -------
    pandas dataframe

    """
    splitList = np.linspace(0.0, 100.0, num=20, endpoint=False)
    print("Testing model generation time period sensitivity...% of data")
    time_test_A_adjustment_df = {}
    TimeTestA_baseline_df = pd.DataFrame()

    for s in splitList[1:]:

        sys.stdout.write("\r")
        sys.stdout.write(f"{str(s).rjust(10, ' ')} %      ")

        inputdata_test = train_test_split(s, inputdata.copy())
        TimeTestA_baseline_df, time_test_A_adjustment_df = quick_metrics(
            inputdata_test,
            config,
            TimeTestA_baseline_df,
            time_test_A_adjustment_df,
            str(100 - s),
        )

    sys.stdout.flush()
    print()    

    return TimeTestA_baseline_df, time_test_A_adjustment_df

def test_train_split_add_days(inputdata, config): 
    """info"""
    numberofObsinOneDay = 144
    numberofDaysInTest = int(round(len(inputdata) / numberofObsinOneDay))
    print("Testing model generation time period sensitivity...days to train model")
    print("Number of days in the study " + str(numberofDaysInTest))
    time_test_B_adjustment_df = {}
    TimeTestB_baseline_df = pd.DataFrame()

    for i in range(0, numberofDaysInTest):

        sys.stdout.write("\r")
        sys.stdout.write(
            f"{str(i).rjust(10, ' ')} of {str(numberofDaysInTest)} days   "
        )

        windowEnd = (i + 1) * (numberofObsinOneDay)
        inputdata_test = train_test_split(
            i, inputdata.copy(), stepOverride=[0, windowEnd]
        )
        TimeTestB_baseline_df, time_test_B_adjustment_df = quick_metrics(
            inputdata_test,
            config,
            TimeTestB_baseline_df,
            time_test_B_adjustment_df,
            str(numberofDaysInTest - i),
        )

    sys.stdout.flush()
    print()

    return TimeTestB_baseline_df, time_test_B_adjustment_df

def test_train_split_window(inputdata, config): 
    """
    fix this it's broken
    """

    TimeTestC_baseline_df = pd.DataFrame()
    time_test_C_adjustment_df = {}

    #numberofObsinOneDay = 144
    #if len(inputdata) > (
    #    numberofObsinOneDay * 90
    #):  # check to see if experiment is greater than 3 months
    #        "Testing model generation time period sensitivity...6 week window pick"
    #    )
    #    windowStart = 0
    #    windowEnd = numberofObsinOneDay * 42
    #    time_test_C_adjustment_df = {}
    #    TimeTestC_baseline_df = pd.DataFrame()

    #    while windowEnd < len(inputdata):
    #        print(
    #            str(
    #                "After observation #"
    #                + str(windowStart)
    #                + " "
    #                + "Before observation #"
    #                + str(windowEnd)
    #            )
    #        )
    #        windowStart += numberofObsinOneDay * 7
    #        windowEnd = windowStart + (numberofObsinOneDay * 42)
    #        inputdata_test = train_test_split(
    #            i, inputdata.copy(), stepOverride=[windowStart, windowEnd]
    #        )
    #        TimeTestC_baseline_df, time_test_C_adjustment_df = quick_metrics(
    #                inputdata_test,
    #                config,
    #                TimeTestC_baseline_df,
    #                time_test_C_adjustment_df,
    #                str("After_" + str(windowStart) + "_" + "Before_" + str(windowEnd)),
    #        )

    return TimeTestC_baseline_df, time_test_C_adjustment_df

