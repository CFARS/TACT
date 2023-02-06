"""
TACT (Turbulence Intensity Adjustment Comparison Tool) is a project supported by CFARS, 
(The Consortium for the Advancement of Remote Sensing). 
This script can be used with remote sensing data to adjust turbulence observations.
Authors: Nikhil Kondabala, Alexandra Arntsen, Andrew Black, Barrett Goudeau, Nigel Swytink-Binnema, Nicolas Jolin

"""
try:
    from TACT import logger
except ImportError:
    pass

from TACT.computation.adjustments import (
    Adjustments,
    get_all_regressions,
    train_test_split,
    quick_metrics,
)
from TACT.computation.methods.GC import perform_G_C_adjustment
from TACT.computation.methods.GSa import perform_G_Sa_adjustment
from TACT.computation.methods.GSFc import perform_G_SFc_adjustment
from TACT.computation.methods.SSLTERRAML import perform_SS_LTERRA_ML_adjustment
from TACT.computation.methods.SSLTERRASML import perform_SS_LTERRA_S_ML_adjustment
from TACT.computation.methods.SSNN import perform_SS_NN_adjustment
from TACT.computation.methods.SSSS import perform_SS_SS_adjustment
from TACT.computation.methods.SSWS import perform_SS_WS_adjustment
from TACT.computation.methods.SSWSStd import perform_SS_WS_Std_adjustment
from TACT.computation.match import perform_match, perform_match_input
from TACT.computation.stability import (
    calculate_stability_TKE,
    calculate_stability_alpha,
)
from TACT.computation.TI import (
    get_count_per_WSbin,
    record_TIadj,
)
from TACT.computation.training_duration import test_train_split_convergence
#from TACT.computation.training_duration import test_train_split_add_days
from TACT.computation.training_duration import test_train_split_sliding_window
from TACT.extrapolation.extrapolation import (
    perform_TI_extrapolation,
    extrap_configResult,
)
from TACT.readers.config import Config
from TACT.readers.data import Data
from TACT.writers.console import block_print, enable_print
from TACT.writers.files import write_all_resultstofile
from TACT.writers.labels import (
    initialize_resultsLists,
    populate_resultsLists,
    populate_resultsLists_stability,
)

import os
import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception(
            "Tool will not run at this time. You must be using Python 3, as running on Python 2 will encounter errors."
        )
    # ------------------------
    # set up and configuration
    # ------------------------
    """parser get_input_files"""
    config = Config()

    """metadata parser"""
    config.get_site_metadata()
    config.get_filtering_metadata()
    config.get_adjustments_metadata()

    """data object assignments"""
    data = Data(config.input_filename, config.config_file)
    data.get_inputdata()
    data.get_refTI_bins()  # >> to data_file.py
    data.check_for_alphaConfig()

    a = data.a
    lab_a = data.lab_a
    print("%%%%%%%%%%%%%%%%%%%%%%%%% Processing Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # -------------------------------
    # special handling for data types
    # -------------------------------
    stabilityFlag = False
    if config.RSDtype["Selection"][0:4] == "Wind":
        stabilityFlag = True
    if config.RSDtype["Selection"] == "ZX":
        stabilityFlag = True
        TI_computed = data.inputdata["RSD_SD"] / data.inputdata["RSD_WS"]
        RepTI_computed = TI_computed + 1.28 * data.inputdata["RSD_SD"]
        data.inputdata = data.inputdata.rename(columns={"RSD_TI": "RSD_TI_instrument"})
        data.inputdata = data.inputdata.rename(
            columns={"RSD_RepTI": "RSD_RepTI_instrument"}
        )
        data.inputdata["RSD_TI"] = TI_computed
        data.inputdata["RSD_RepTI"] = RepTI_computed
    elif config.RSDtype["Selection"] == "Triton":
        print(
            "RSD type is triton, not that output uncorrected TI is instrument corrected"
        )
    # ------------------------
    # Baseline Results
    # ------------------------
    # Get all regressions available
    reg_results = get_all_regressions(data.inputdata, title="Full comparison")

    (
        stabilityClass_tke,
        stabilityMetric_tke,
        regimeBreakdown_tke,
    ) = calculate_stability_TKE(data, config)
    (
        cup_alphaFlag,
        stabilityClass_ane,
        stabilityMetric_ane,
        regimeBreakdown_ane,
        Ht_1_ane,
        Ht_2_ane,
        stabilityClass_rsd,
        stabilityMetric_rsd,
        regimeBreakdown_rsd,
    ) = calculate_stability_alpha(data, config)

    # ------------------------
    # Time Sensivity Analysis
    # this section needs it's own module, and grooming/work
    # ------------------------
    # TimeTestA = pd.DataFrame()
    # TimeTestB = pd.DataFrame()
    # TimeTestC = pd.DataFrame()

    # To Do: add training_duration.py and put this section there
    #--------------------------------------------------------------
    sys.exit()
    if config.time_test_flag == True:
        # A) increase % of test train split -- check for convergence --- basic metrics recorded baseline but also for every adjustments
        splitList = np.linspace(0.0, 100.0, num=20, endpoint=False)
        print("Testing model generation time period sensitivity...% of data")
        time_test_A_adjustment_df = {}
        TimeTestA_baseline_df = pd.DataFrame()

        for s in splitList[1:]:

            sys.stdout.write("\r")
            sys.stdout.write(f"{str(s).rjust(10, ' ')} %      ")

            inputdata_test = train_test_split(s, data.inputdata.copy())
            TimeTestA_baseline_df, time_test_A_adjustment_df = quick_metrics(
                inputdata_test,
                config,
                TimeTestA_baseline_df,
                time_test_A_adjustment_df,
                str(100 - s),
            )

        sys.stdout.flush()
        print()

        # B) incrementally Add days to training set sequentially -- check for convergence
        numberofObsinOneDay = 144
        numberofDaysInTest = int(round(len(data.inputdata) / numberofObsinOneDay))
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
                i, data.inputdata.copy(), stepOverride=[0, windowEnd]
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

        # C) If experiment is greater than 3 months, slide a 6 week window (1 week step)
        if len(data.inputdata) > (
            numberofObsinOneDay * 90
        ):  # check to see if experiment is greater than 3 months
            print(
                "Testing model generation time period sensitivity...6 week window pick"
            )
            windowStart = 0
            windowEnd = numberofObsinOneDay * 42
            time_test_C_adjustment_df = {}
            TimeTestC_baseline_df = pd.DataFrame()

            while windowEnd < len(data.inputdata):
                print(
                    str(
                        "After observation #"
                        + str(windowStart)
                        + " "
                        + "Before observation #"
                        + str(windowEnd)
                    )
                )
                windowStart += numberofObsinOneDay * 7
                windowEnd = windowStart + (numberofObsinOneDay * 42)
                inputdata_test = train_test_split(
                    i, data.inputdata.copy(), stepOverride=[windowStart, windowEnd]
                )
                TimeTestC_baseline_df, time_test_C_adjustment_df = quick_metrics(
                    inputdata_test,
                    config,
                    TimeTestC_baseline_df,
                    time_test_C_adjustment_df,
                    str("After_" + str(windowStart) + "_" + "Before_" + str(windowEnd)),
                )
    else:
        TimeTestA_baseline_df = pd.DataFrame()
        TimeTestB_baseline_df = pd.DataFrame()
        TimeTestC_baseline_df = pd.DataFrame()
        time_test_A_adjustment_df = {}
        time_test_B_adjustment_df = {}
        time_test_C_adjustment_df = {}

    # -----------------------
    # Test - Train split
    # -----------------------
    # random 80-20 split
    data.inputdata = train_test_split(80.0, data.inputdata.copy())

    inputdata_train = (
        data.inputdata[data.inputdata["split"] == True].copy().join(data.timestamps)
    )
    inputdata_test = (
        data.inputdata[data.inputdata["split"] == False].copy().join(data.timestamps)
    )

    timestamp_train = inputdata_train["Timestamp"]
    timestamp_test = inputdata_test["Timestamp"]

    # -----------------------------
    # stability class subset lists
    # -----------------------------
    # get reg_results by stability class: list of df's for each height
    reg_results_class1 = []
    reg_results_class2 = []
    reg_results_class3 = []
    reg_results_class4 = []
    reg_results_class5 = []

    reg_results_class1_alpha = {}
    reg_results_class2_alpha = {}
    reg_results_class3_alpha = {}
    reg_results_class4_alpha = {}
    reg_results_class5_alpha = {}

    if (
        config.RSDtype["Selection"][0:4] == "Wind"
        or "ZX" in config.RSDtype["Selection"]
    ):

        inputdata_class1 = []
        inputdata_class2 = []
        inputdata_class3 = []
        inputdata_class4 = []
        inputdata_class5 = []
        RSD_h = []

        Alldata_inputdata = data.inputdata.copy()
        for h in stabilityClass_tke.columns.to_list():
            RSD_h.append(h)
            inputdata_class1.append(Alldata_inputdata[Alldata_inputdata[h] == 1])
            inputdata_class2.append(Alldata_inputdata[Alldata_inputdata[h] == 2])
            inputdata_class3.append(Alldata_inputdata[Alldata_inputdata[h] == 3])
            inputdata_class4.append(Alldata_inputdata[Alldata_inputdata[h] == 4])
            inputdata_class5.append(Alldata_inputdata[Alldata_inputdata[h] == 5])

        All_class_data = [
            inputdata_class1,
            inputdata_class2,
            inputdata_class3,
            inputdata_class4,
            inputdata_class5,
        ]
        All_class_data_clean = [
            inputdata_class1,
            inputdata_class2,
            inputdata_class3,
            inputdata_class4,
            inputdata_class5,
        ]

        for h in RSD_h:
            idx = RSD_h.index(h)
            df = inputdata_class1[idx]
            reg_results_class1.append(
                get_all_regressions(df, title=str("TKE_stability_" + h + "class1"))
            )
            df = inputdata_class2[idx]
            reg_results_class2.append(
                get_all_regressions(df, title=str("TKE_stability_" + h + "class2"))
            )
            df = inputdata_class3[idx]
            reg_results_class3.append(
                get_all_regressions(df, title=str("TKE_stability_" + h + "class3"))
            )
            df = inputdata_class4[idx]
            reg_results_class4.append(
                get_all_regressions(df, title=str("TKE_stability_" + h + "class4"))
            )
            df = inputdata_class5[idx]
            reg_results_class5.append(
                get_all_regressions(df, title=str("TKE_stability_" + h + "class5"))
            )

    if data.RSD_alphaFlag:
        del (
            inputdata_class1,
            inputdata_class2,
            inputdata_class3,
            inputdata_class4,
            inputdata_class5,
        )

        Alldata_inputdata = data.inputdata.copy()
        colName = stabilityClass_rsd.name
        Alldata_inputdata[colName] = stabilityClass_rsd.values

        inputdata_class1 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_rsd.name] == 1.0
        ]
        inputdata_class2 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_rsd.name] == 2.0
        ]
        inputdata_class3 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_rsd.name] == 3.0
        ]
        inputdata_class4 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_rsd.name] == 4.0
        ]
        inputdata_class5 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_rsd.name] == 5.0
        ]

        All_class_data_alpha_RSD = [
            inputdata_class1,
            inputdata_class2,
            inputdata_class3,
            inputdata_class4,
            inputdata_class5,
        ]
        All_class_data_alpha_RSD_clean = [
            inputdata_class1.copy(),
            inputdata_class2.copy(),
            inputdata_class3.copy(),
            inputdata_class4.copy(),
            inputdata_class5.copy(),
        ]

        reg_results_class1_alpha["RSD"] = get_all_regressions(
            inputdata_class1, title=str("alpha_stability_RSD" + "class1")
        )
        reg_results_class2_alpha["RSD"] = get_all_regressions(
            inputdata_class2, title=str("alpha_stability_RSD" + "class2")
        )
        reg_results_class3_alpha["RSD"] = get_all_regressions(
            inputdata_class3, title=str("alpha_stability_RSD" + "class3")
        )
        reg_results_class4_alpha["RSD"] = get_all_regressions(
            inputdata_class4, title=str("alpha_stability_RSD" + "class4")
        )
        reg_results_class5_alpha["RSD"] = get_all_regressions(
            inputdata_class5, title=str("alpha_stability_RSD" + "class5")
        )

    if cup_alphaFlag:
        del (
            inputdata_class1,
            inputdata_class2,
            inputdata_class3,
            inputdata_class4,
            inputdata_class5,
        )

        Alldata_inputdata = data.inputdata.copy()
        colName = stabilityClass_ane.name
        Alldata_inputdata[colName] = stabilityClass_ane.values

        inputdata_class1 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_ane.name] == 1.0
        ]
        inputdata_class2 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_ane.name] == 2.0
        ]
        inputdata_class3 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_ane.name] == 3.0
        ]
        inputdata_class4 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_ane.name] == 4.0
        ]
        inputdata_class5 = Alldata_inputdata[
            Alldata_inputdata[stabilityClass_ane.name] == 5.0
        ]

        All_class_data_alpha_Ane = [
            inputdata_class1,
            inputdata_class2,
            inputdata_class3,
            inputdata_class4,
            inputdata_class5,
        ]
        All_class_data_alpha_Ane_clean = [
            inputdata_class1.copy(),
            inputdata_class2.copy(),
            inputdata_class3.copy(),
            inputdata_class4.copy(),
            inputdata_class5.copy(),
        ]

        reg_results_class1_alpha["Ane"] = get_all_regressions(
            inputdata_class1, title=str("alpha_stability_Ane" + "class1")
        )
        reg_results_class2_alpha["Ane"] = get_all_regressions(
            inputdata_class2, title=str("alpha_stability_Ane" + "class2")
        )
        reg_results_class3_alpha["Ane"] = get_all_regressions(
            inputdata_class3, title=str("alpha_stability_Ane" + "class3")
        )
        reg_results_class4_alpha["Ane"] = get_all_regressions(
            inputdata_class4, title=str("alpha_stability_Ane" + "class4")
        )
        reg_results_class5_alpha["Ane"] = get_all_regressions(
            inputdata_class5, title=str("alpha_stability_Ane" + "class5")
        )

    # ------------------------
    # TI Adjustments
    # ------------------------
    from TACT.computation.adjustments import Adjustments

    baseResultsLists = initialize_resultsLists("")

    # get number of observations in each bin
    count_1mps, count_05mps = get_count_per_WSbin(data.inputdata, "RSD_WS")

    inputdata_train = (
        data.inputdata[data.inputdata["split"] == True].copy().join(data.timestamps)
    )
    inputdata_test = (
        data.inputdata[data.inputdata["split"] == False].copy().join(data.timestamps)
    )

    timestamp_train = inputdata_train["Timestamp"]
    timestamp_test = inputdata_test["Timestamp"]

    count_1mps_train, count_05mps_train = get_count_per_WSbin(inputdata_train, "RSD_WS")
    count_1mps_test, count_05mps_test = get_count_per_WSbin(inputdata_test, "RSD_WS")

    if (
        config.RSDtype["Selection"][0:4] == "Wind"
        or "ZX" in config.RSDtype["Selection"]
    ):
        primary_c = [h for h in RSD_h if "Ht" not in h]
        primary_idx = RSD_h.index(primary_c[0])
        ResultsLists_stability = initialize_resultsLists("stability_")
    if cup_alphaFlag:
        ResultsLists_stability_alpha_Ane = initialize_resultsLists(
            "stability_alpha_Ane"
        )
    if data.RSD_alphaFlag:
        ResultsLists_stability_alpha_RSD = initialize_resultsLists(
            "stability_alpha_RSD"
        )

    name_1mps_tke = []
    name_1mps_alpha_Ane = []
    name_1mps_alpha_RSD = []
    name_05mps_tke = []
    name_05mps_alpha_Ane = []
    name_05mps_alpha_RSD = []
    count_1mps_tke = []
    count_1mps_alpha_Ane = []
    count_1mps_alpha_RSD = []
    count_05mps_tke = []
    count_05mps_alpha_Ane = []
    count_05mps_alpha_RSD = []

    for c in range(0, len(All_class_data)):
        name_1mps_tke.append(str("count_1mps_class_" + str(c) + "_tke"))
        name_1mps_alpha_Ane.append(str("count_1mps_class_" + str(c) + "_alpha_Ane"))
        name_1mps_alpha_RSD.append(str("count_1mps_class_" + str(c) + "_alpha_RSD"))
        name_05mps_tke.append(str("count_05mps_class_" + str(c) + "_tke"))
        name_05mps_alpha_Ane.append(str("count_05mps_class_" + str(c) + "_alpha_Ane"))
        name_05mps_alpha_RSD.append(str("count_05mps_class_" + str(c) + "_alpha_RSD"))

        try:
            c_1mps_tke, c_05mps_tke = get_count_per_WSbin(
                All_class_data[c][primary_idx], "RSD_WS"
            )
            count_1mps_tke.append(c_1mps_tke)
            count_05mps_tke.append(c_05mps_tke)
        except:
            count_1mps_tke.append(None)
            count_05mps_tke.append(None)
        try:
            c_1mps_alpha_Ane, c_05mps_alpha_Ane = get_count_per_WSbin(
                All_class_data_alpha_Ane[c], "RSD_WS"
            )
            count_1mps_alpha_Ane.append(c_1mps_alpha_Ane)
            count_05mps_alpha_Ane.append(c_05mps_alpha_Ane)
        except:
            count_1mps_alpha_Ane.append(None)
            count_05mps_alpha_Ane.append(None)
        try:
            c_1mps_alpha_RSD, c_05mps_alpha_RSD = get_count_per_WSbin(
                All_class_data_alpha_RSD[c], "RSD_WS"
            )
            count_1mps_alpha_RSD.append(c_1mps_alpha_RSD)
            count_05mps_alpha_RSD.append(c_05mps_alpha_RSD)
        except:
            count_1mps_alpha_RSD.append(None)
            count_05mps_alpha_RSD.append(None)

    # intialize 10 minute output
    TI_10minuteAdjusted = pd.DataFrame()

    # initialize Adjustments object
    adjuster = Adjustments(
        data.inputdata.copy(), config.adjustments_metadata, baseResultsLists
    )

    for method in config.adjustments_metadata:
        # Checking whether or not to execute each adjustement method available 
        # based on data configuration checks

        # Site Specific Simple Adjustment (SS-S)
        if method != "SS-S":
            pass
        elif method == "SS-S" and config.adjustments_metadata["SS-S"] == False:
            pass
        else:
            print("Applying Adjustment Method: SS-S")
            logger.info("Applying Adjustment Method: SS-S")
            inputdata_adj, results, m, c = adjuster.perform_SS_S_adjustment(
                data.inputdata.copy()
            )
            print("SS-S: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-S"
            adjustment_name = "SS_S"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: SS-S by stability class (TKE)")
                logger.info("Applying Adjustment Method: SS-S by stability class (TKE)")
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = adjuster.perform_SS_S_adjustment(
                        item[primary_idx].copy()
                    )
                    print("SS-S: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-S" + "_TKE_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS-S" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-S by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-S by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                print(str("class " + str(className)))
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = adjuster.perform_SS_S_adjustment(
                        item.copy()
                    )
                    print("SS-S: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str("SS-S" + "_" + "class_" + str(className))
                    adjustment_name = str("SS-S" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print("Applying Adjustment Method: SS-S by stability class Alpha w/cup")
                logger.info(
                    "Applying Adjustment Method: SS-S by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = adjuster.perform_SS_S_adjustment(
                        item.copy()
                    )
                    print("SS-S: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-S" + "_alphaCup_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS-S" + "_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ********************************************** #
        # Site Specific Simple + Filter Adjustment (SS-SF)
        if method != "SS-SF":
            pass
        elif method == "SS-SF" and config.adjustments_metadata["SS-SF"] == False:
            pass
        else:
            print("Applying Adjustment Method: SS-SF")
            logger.info("Applying Adjustment Method: SS-SF")
            # inputdata_adj, lm_adj, m, c = perform_SS_SF_adjustment(data.inputdata.copy())
            inputdata_adj, lm_adj, m, c = adjuster.perform_SS_SF_adjustment(
                data.inputdata.copy()
            )
            print("SS-SF: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-SF"
            adjustment_name = "SS_SF"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if (
                config.RSDtype["Selection"][0:4] == "Wind"
                or "ZX" in config.RSDtype["Selection"]
            ):
                print("Applying Adjustment Method: SS-SF by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: SS-SF by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = adjuster.perform_SS_SF_adjustment(
                        item[primary_idx].copy()
                    )
                    print("SS-SF: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-SF" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_SF" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-SF by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-SF by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = adjuster.perform_SS_SF_adjustment(
                        item.copy()
                    )
                    print("SS-SF: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-SF" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_SF" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-SF by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-SF by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = adjuster.perform_SS_SF_adjustment(
                        item.copy()
                    )
                    print("SS-SF: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-SF" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_SF" + "_alphaCup_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ************************************ #
        # Site Specific Simple Adjustment (SS-SS) combining stability classes adjusted differently
        if method != "SS-SS":
            pass
        elif method == "SS-SS" and config.adjustments_metadata["SS-SS"] == False:
            pass
        elif (
            config.RSDtype["Selection"][0:4] != "Wind"
            and "ZX" not in config.RSDtype["Selection"]
        ):
            pass
        else:
            print("Applying Adjustment Method: SS-SS")
            logger.info("Applying Adjustment Method: SS-SS")
            inputdata_adj, lm_adj, m, c = perform_SS_SS_adjustment(
                data.inputdata.copy(), All_class_data, primary_idx
            )
            print("SS-SS: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-SS"
            adjustment_name = "SS_SS"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print(
                    "Applying Adjustment Method: SS-SS by stability class (TKE). SAME as Baseline"
                )
                logger.info(
                    "Applying Adjustment Method: SS-SS by stability class (TKE). SAME as Baseline"
                )
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    print("SS-SS: y = " + str(m) + " * x + " + str(c))
                    adjustment_name = str("SS_SS" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )
            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-SS by stability class Alpha w/ RSD. SAEM as Baseline"
                )
                logger.info(
                    "Applying Adjustment Method: SS-SS by stability class Alpha w/ RSD. SAEM as Baseline"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    print("SS-SS: y = " + str(m) + "* x +" + str(c))
                    adjustment_name = str("SS_SS" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-SS by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-SS by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    print("SS-SS: y = " + str(m) + "* x +" + str(c))
                    emptyclassFlag = False
                    adjustment_name = str("SS_SS" + "_alphaCup_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ******************************************* #
        # Site Specific WindSpeed Adjustment (SS-WS)
        if method != "SS-WS":
            pass
        elif method == "SS-WS" and config.adjustments_metadata["SS-WS"] == False:
            pass
        else:
            print("Applying Adjustment Method: SS-WS")
            logger.info("Applying Adjustment Method: SS-WS")
            inputdata_adj, lm_adj, m, c = perform_SS_WS_adjustment(
                data.inputdata.copy()
            )
            print("SS-WS: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-WS"
            adjustment_name = "SS_WS"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if (
                config.RSDtype["Selection"][0:4] == "Wind"
                or "ZX" in config.RSDtype["Selection"]
            ):
                print("Applying Adjustment Method: SS-WS by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: SS-WS by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_SS_WS_adjustment(
                        item[primary_idx].copy()
                    )
                    print("SS-WS: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-WS" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_WS" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )
            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-WS by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-WS by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_SS_WS_adjustment(item.copy())
                    print("SS-WS: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-WS" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_WS" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-WS by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-WS by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_SS_WS_adjustment(item.copy())
                    print("SS-WS: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-WS" + "_" + "class_" + str(className)
                    )
                    emptyclassFlag = False
                    adjustment_name = str("SS_WS" + "_alphaCup_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ******************************************* #
        # Site Specific Comprehensive Adjustment (SS-WS-Std)
        if method != "SS-WS-Std":
            pass
        elif (
            method == "SS-WS-Std" and config.adjustments_metadata["SS-WS-Std"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: SS-WS-Std")
            logger.info("Applying Adjustment Method: SS-WS-Std")
            inputdata_adj, lm_adj, m, c = perform_SS_WS_Std_adjustment(
                data.inputdata.copy()
            )
            print("SS-WS-Std: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-WS-Std"
            adjustment_name = "SS_WS_Std"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if (
                config.RSDtype["Selection"][0:4] == "Wind"
                or "ZX" in config.RSDtype["Selection"]
            ):
                print("Applying Adjustment Method: SS-WS-Std by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: SS-WS-Std by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_SS_WS_Std_adjustment(
                        item[primary_idx].copy()
                    )
                    print("SS-WS-Std: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-WS-Std" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_WS_Std" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )
            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-WS-Std by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-WS-Std by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_SS_WS_Std_adjustment(
                        item.copy()
                    )
                    print("SS-WS-Std: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-WS-Std" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_WS_Std" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-WS-Std by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-WS-Std by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_SS_WS_Std_adjustment(
                        item.copy()
                    )
                    print("SS-WS-Std: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-WS-Std" + "_" + "class_" + str(className)
                    )
                    emptyclassFlag = False
                    adjustment_name = str("SS_WS_Std" + "_alphaCup_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # **************************************************************** #
        # Site Specific LTERRA for WC 1HZ Data Adjustment (G-LTERRA_WC_1HZ)
        if method != "SS-LTERRA-WC-1HZ":
            pass
        elif (
            method == "SS-LTERRA-WC-1HZ"
            and config.adjustments_metadata["SS-LTERRA-WC-1HZ"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: SS-LTERRA-WC-1HZ")
            logger.info("Applying Adjustment Method: SS-LTERRA-WC-1HZ")

        # ******************************************************************* #
        # Site Specific LTERRA WC Machine Learning Adjustment (SS-LTERRA-MLa)
        # Random Forest Regression with now ancillary columns
        if method != "SS-LTERRA-MLa":
            pass
        elif (
            method == "SS-LTERRA-MLa"
            and config.adjustments_metadata["SS-LTERRA-MLa"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: SS-LTERRA-MLa")
            logger.info("Applying Adjustment Method: SS-LTERRA-MLa")

            inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_ML_adjustment(
                data.inputdata.copy()
            )
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS_LTERRA_MLa"
            adjustment_name = "SS_LTERRA_MLa"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print(
                    "Applying Adjustment Method: SS-LTERRA MLa by stability class (TKE)"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA MLa by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_ML_adjustment(
                        item[primary_idx].copy()
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS_LTERRA_MLa" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_LTERRA_MLa" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )
            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-LTERRA MLa by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA MLa by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_ML_adjustment(
                        item.copy()
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-LTERRA_MLa" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str(
                        "SS_LTERRA_ML" + "_alphaRSD_" + str(className)
                    )
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLa by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLa by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_ML_adjustment(
                        item.copy()
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS_LTERRA_MLa" + "_" + "class_" + str(className)
                    )
                    emptyclassFlag = False
                    adjustment_name = str(
                        "SS_LTERRA_MLa" + "_alphaCup_" + str(className)
                    )
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ************************************************************************************ #
        # Site Specific LTERRA WC (w/ stability) Machine Learning Adjustment (SS-LTERRA_MLc)
        if method != "SS-LTERRA-MLc":
            pass
        elif (
            method == "SS-LTERRA-MLc"
            and config.adjustments_metadata["SS-LTERRA-MLc"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: SS-LTERRA-MLc")
            logger.info("Applying Adjustment Method: SS-LTERRA-MLc")
            all_trainX_cols = [
                "x_train_TI",
                "x_train_TKE",
                "x_train_WS",
                "x_train_DIR",
                "x_train_Hour",
            ]
            all_trainY_cols = ["y_train"]
            all_testX_cols = [
                "x_test_TI",
                "x_test_TKE",
                "x_test_WS",
                "x_test_DIR",
                "x_test_Hour",
            ]
            all_testY_cols = ["y_test"]

            inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                data.inputdata.copy(),
                all_trainX_cols,
                all_trainY_cols,
                all_testX_cols,
                all_testY_cols,
            )
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS_LTERRA_MLc"
            adjustment_name = "SS_LTERRA_MLc"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )

            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLc by stability class (TKE)"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLc by stability class (TKE)"
                )

                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                        item[primary_idx].copy(),
                        all_trainX_cols,
                        all_trainY_cols,
                        all_testX_cols,
                        all_testY_cols,
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS_LTERRA_MLc" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_LTERRA_MLc" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )
            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLc by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLc by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                        item.copy(),
                        all_trainX_cols,
                        all_trainY_cols,
                        all_testX_cols,
                        all_testY_cols,
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-LTERRA_MLc" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str(
                        "SS_LTERRA_S_ML" + "_alphaRSD_" + str(className)
                    )
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLc by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLc by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                        item.copy(),
                        all_trainX_cols,
                        all_trainY_cols,
                        all_testX_cols,
                        all_testY_cols,
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS_LTERRA_MLc" + "_" + "class_" + str(className)
                    )
                    emptyclassFlag = False
                    adjustment_name = str(
                        "SS_LTERRA_MLc" + "_alphaCup_" + str(className)
                    )
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # *********************** #
        # Site Specific SS-LTERRA-MLb
        if method != "SS-LTERRA-MLb":
            pass
        elif (
            method == "SS-LTERRA-MLb"
            and config.adjustments_metadata["SS-LTERRA-MLb"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: SS-LTERRA-MLb")
            logger.info("Applying Adjustment Method: SS-LTERRA-MLb")
            all_trainX_cols = ["x_train_TI", "x_train_TKE"]
            all_trainY_cols = ["y_train"]
            all_testX_cols = ["x_test_TI", "x_test_TKE"]
            all_testY_cols = ["y_test"]

            inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                data.inputdata.copy(),
                all_trainX_cols,
                all_trainY_cols,
                all_testX_cols,
                all_testY_cols,
            )
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS_LTERRA_MLb"
            adjustment_name = "SS_LTERRA_MLb"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )

            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLb by stability class (TKE)"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLb by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                        item[primary_idx].copy(),
                        all_trainX_cols,
                        all_trainY_cols,
                        all_testX_cols,
                        all_testY_cols,
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS_LTERRA_MLb" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_LTERRA_MLb" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )
            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLb by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLb by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                        item.copy(),
                        all_trainX_cols,
                        all_trainY_cols,
                        all_testX_cols,
                        all_testY_cols,
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-LTERRA_MLb" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str(
                        "SS_LTERRA_MLb" + "_alphaRSD_" + str(className)
                    )
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-LTERRA_MLb by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-LTERRA_MLb by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                        item.copy(),
                        all_trainX_cols,
                        all_trainY_cols,
                        all_testX_cols,
                        all_testY_cols,
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS_LTERRA_MLb" + "_" + "class_" + str(className)
                    )
                    emptyclassFlag = False
                    adjustment_name = str(
                        "SS_LTERRA_MLb" + "_alphaCup_" + str(className)
                    )
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # *********************** #
        # TI Extrapolation (TI-Ext)
        if method != "TI-Extrap":
            pass
        elif (
            method == "TI-Extrap" and config.adjustments_metadata["TI-Extrap"] == False
        ):
            pass
        else:
            print("Found enough data to perform extrapolation comparison")
            block_print()
            # Get extrapolation height
            height_extrap = float(
                config.extrap_metadata["height"][
                    config.extrap_metadata["type"] == "extrap"
                ]
            )
            # Extrapolate
            inputdata_adj, lm_adj, shearTimeseries = perform_TI_extrapolation(
                data.inputdata.copy(),
                config.extrap_metadata,
                config.extrapolation_type,
                config.height,
            )
            adjustment_name = "TI_EXTRAP"
            lm_adj["adjustment"] = adjustment_name

            inputdataEXTRAP = inputdata_adj.copy()
            inputdataEXTRAP, baseResultsLists = extrap_configResult(
                config.extrapolation_type,
                inputdataEXTRAP,
                baseResultsLists,
                method,
                lm_adj,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, shearTimeseries = perform_TI_extrapolation(
                        item[primary_idx].copy(),
                        config.extrap_metadata,
                        config.extrapolation_type,
                        config.height,
                    )
                    lm_adj["adjustment"] = str(
                        "TI_EXT_class1" + "_TKE_" + "class_" + str(className)
                    )
                    inputdataEXTRAP = inputdata_adj.copy()
                    inputdataEXTRAP, ResultsLists_class = extrap_configResult(
                        config.extrapolation_type,
                        inputdataEXTRAP,
                        ResultsLists_class,
                        method,
                        lm_adj,
                        appendString="class_",
                    )
                    className += 1

                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if cup_alphaFlag:
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, shearTimeseries = perform_TI_extrapolation(
                        item.copy(),
                        config.extrap_metadata,
                        config.extrapolation_type,
                        config.height,
                    )
                    lm_adj["adjustment"] = str(
                        "TI_Ane_class1" + "_alphaCup_" + "class_" + str(className)
                    )
                    inputdataEXTRAP = inputdata_adj.copy()
                    inputdataEXTRAP, ResultsLists_class_alpha_Ane = extrap_configResult(
                        config.extrapolation_type,
                        inputdataEXTRAP,
                        ResultsLists_class_alpha_Ane,
                        method,
                        lm_adj,
                        appendString="class_alpha_Ane",
                    )
                    className += 1

                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )
            if data.RSD_alphaFlag:
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, shearTimeseries = perform_TI_extrapolation(
                        item.copy(),
                        config.extrap_metadata,
                        config.extrapolation_type,
                        config.height,
                    )
                    lm_adj["adjustment"] = str(
                        "TI_RSD_class1" + "_alphaRSD_" + "class_" + str(className)
                    )
                    inputdataEXTRAP = inputdata_adj.copy()
                    inputdataEXTRAP, ResultsLists_class_alpha_RSD = extrap_configResult(
                        config.extrapolation_type,
                        inputdataEXTRAP,
                        ResultsLists_class_alpha_RSD,
                        method,
                        lm_adj,
                        appendString="class_alpha_RSD",
                    )
                    className += 1

                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )
            # Add extra info to meta data and reformat
            if config.extrapolation_type == "simple":
                desc = "No truth measurement at extrapolation height"
            else:
                desc = "Truth measurement available at extrapolation height"
            config.extrap_metadata = (
                config.extrap_metadata.append(
                    {"type": np.nan, "height": np.nan, "num": np.nan}, ignore_index=True
                )
                .append(
                    pd.DataFrame(
                        [["extrapolation type", config.extrapolation_type, desc]],
                        columns=config.extrap_metadata.columns,
                    )
                )
                .rename(
                    columns={
                        "type": "Type",
                        "height": "Height (m)",
                        "num": "Comparison Height Number",
                    }
                )
            )
            enable_print()

        # ************************************************** #
        # Histogram Matching
        if method != "SS-Match":
            pass
        elif method == "SS-Match" and config.adjustments_metadata["SS-Match"] == False:
            pass
        else:
            print("Applying Match algorithm: SS-Match")
            logger.info("Applying Match algorithm: SS-Match")
            inputdata_adj, lm_adj = perform_match(data.inputdata.copy())
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-Match"
            adjustment_name = "SS_Match"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: SS-Match by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: SS-Match by stability class (TKE)"
                )
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj = perform_match(item[primary_idx].copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-Match" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_Match" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-Match by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-Match by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj = perform_match(item.copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-Match" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_Match" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-Match by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-Match by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj = perform_match(item.copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-Match" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_Match" + "_alphaCup_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ************************************************** #
        # Histogram Matching Input Corrected
        if method != "SS-Match2":
            pass
        elif (
            method == "SS-Match2" and config.adjustments_metadata["SS-Match2"] == False
        ):
            pass
        else:
            print("Applying input match algorithm: SS-Match2")
            logger.info("Applying input match algorithm: SS-Match2")
            inputdata_adj, lm_adj = perform_match_input(data.inputdata.copy())
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "SS-Match2"
            adjustment_name = "SS_Match2"

            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: SS-Match2 by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: SS-Match2 by stability class (TKE)"
                )
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj = perform_match_input(
                        item[primary_idx].copy()
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-Match2" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_Match2" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-Match2 by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: SS-Match2 by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj = perform_match_input(item.copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-Match2" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_Match2" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: SS-Match2 by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: SS-Match2 by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj = perform_match_input(item.copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "SS-Match2" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("SS_Match2" + "_alphaCup_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )
        # ************************************************** #
        # Global Simple Phase II mean Linear Reressions (G-Sa) + project
        """
            RSD_TI = .984993 * RSD_TI + .087916
        """

        if method != "G-Sa":
            pass
        elif method == "G-Sa" and config.adjustments_metadata["G-Sa"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-Sa")
            logger.info("Applying Adjustment Method: G-Sa")
            override = False
            inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                data.inputdata.copy(), override, config.RSDtype
            )
            print("G-Sa: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "G-Sa"
            adjustment_name = "G_Sa"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: G-Sa by stability class (TKE)")
                logger.info("Applying Adjustment Method: G-Sa by stability class (TKE)")
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                        item[primary_idx].copy(), override, config.RSDtype
                    )
                    print("G-Sa: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-Sa" + "_TKE_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-Sa" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: G-Sa by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: G-Sa by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                        item.copy(), override, config.RSDtype
                    )
                    print("G-Sa: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str("G-Sa" + "_" + "class_" + str(className))
                    adjustment_name = str("G-Sa" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print("Applying Adjustment Method: G-Sa by stability class Alpha w/cup")
                logger.info(
                    "Applying Adjustment Method: G-Sa by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                        item.copy(), override, config.RSDtype
                    )
                    print("G-Sa: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-Sa" + "_alphaCup_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-Sa" + "_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ******************************************************** #
        # Global Simple w/filter Phase II Linear Regressions (G-SFa) + project
        # Check these values, but for WC m = 0.7086 and c = 0.0225
        if method != "G-SFa":
            pass
        elif method == "G-SFa" and config.adjustments_metadata["G-SFa"] == False:
            pass
        elif config.RSDtype["Selection"][0:4] != "Wind":
            pass
        else:
            print("Applying Adjustment Method: G-SFa")
            logger.info("Applying Adjustment Method: G-SFa")
            override = [0.7086, 0.0225]
            inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                data.inputdata.copy(), override, config.RSDtype
            )
            print("G-SFa: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "G-SFa"
            adjustment_name = "G_SFa"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: G-SFa by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: G-SFa by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                        item[primary_idx].copy(), override, config.RSDtype
                    )
                    print("G-SFa: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-SFa" + "_TKE_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-SFa" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: G-SFa by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: G-SFa by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                        item.copy(), override, config.RSDtype
                    )
                    print("G-SFa: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str("G-Sa" + "_" + "class_" + str(className))
                    adjustment_name = str("G-SFa" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: G-SFa by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: G-SFa by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_G_Sa_adjustment(
                        item.copy(), override, config.RSDtype
                    )
                    print("G-SFa: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-SFa" + "_alphaCup_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-SFa" + "_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ************************************************ #
        # Global Standard Deviation and WS adjustment (G-Sc)
        if method != "G-SFc":
            pass
        elif method == "G-SFc" and config.adjustments_metadata["G-SFc"] == False:
            pass
        elif config.RSDtype["Selection"][0:4] != "Wind":
            pass
        else:
            print("Applying Adjustment Method: G-Sc")
            logger.info("Applying Adjustment Method: G-Sc")
            inputdata_adj, lm_adj, m, c = perform_G_SFc_adjustment(
                data.inputdata.copy()
            )
            print("G-SFc: y = " + str(m) + " * x + " + str(c))
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "G-SFc"
            adjustment_name = "G_SFc"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: G-SFa by stability class (TKE)")
                logger.info(
                    "Applying Adjustment Method: G-SFa by stability class (TKE)"
                )
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    inputdata_adj, lm_adj, m, c = perform_G_SFc_adjustment(
                        item[primary_idx].copy()
                    )
                    print("G-SFc: y = " + str(m) + " * x + " + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-SFc" + "_TKE_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-SFc" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print(
                    "Applying Adjustment Method: G-SFc by stability class Alpha w/ RSD"
                )
                logger.info(
                    "Applying Adjustment Method: G-SFc by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_adj, lm_adj, m, c = perform_G_SFc_adjustment(item.copy())
                    print("G-SFc: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-SFc" + "_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-SFc" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print(
                    "Applying Adjustment Method: G-SFc by stability class Alpha w/cup"
                )
                logger.info(
                    "Applying Adjustment Method: G-SFc by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_adj, lm_adj, m, c = perform_G_SFc_adjustment(item.copy())
                    print("G-SFc: y = " + str(m) + "* x +" + str(c))
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-SFc" + "_alphaCup_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-SFc" + "_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ************************ #
        # Global Comprehensive (G-C)
        """
        based on empirical calibrations by EON
        """
        if method != "G-C":
            pass
        elif method == "G-C" and config.adjustments_metadata["G-C"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-C")
            logger.info("Applying Adjustment Method: G-C")
            inputdata_adj, lm_adj, m, c = perform_G_C_adjustment(data.inputdata.copy())
            lm_adj["sensor"] = config.model
            lm_adj["height"] = config.height
            lm_adj["adjustment"] = "G-C"
            adjustment_name = "G_C"
            baseResultsLists = populate_resultsLists(
                baseResultsLists,
                "",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                data.timestamps,
                method,
            )
            TI_10minuteAdjusted = record_TIadj(
                adjustment_name,
                inputdata_adj,
                data.timestamps,
                method,
                TI_10minuteAdjusted,
                emptyclassFlag=False,
            )

            if config.RSDtype["Selection"][0:4] == "Wind":
                print("Applying Adjustment Method: G-C by stability class (TKE)")
                logger.info("Applying Adjustment Method: G-C by stability class (TKE)")
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists("class_")
                className = 1
                for item in All_class_data:
                    print(str("class " + str(className)))
                    inputdata_adj, lm_adj, m, c = perform_G_C_adjustment(
                        item[primary_idx].copy()
                    )
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-C" + "_TKE_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-C" + "_TKE_" + str(className))
                    ResultsLists_class = populate_resultsLists(
                        ResultsLists_class,
                        "class_",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(
                    ResultsLists_stability, ResultsLists_class, ""
                )

            if data.RSD_alphaFlag:
                print("Applying Adjustment Method: G-C by stability class Alpha w/ RSD")
                logger.info(
                    "Applying Adjustment Method: G-C by stability class Alpha w/ RSD"
                )
                ResultsLists_class_alpha_RSD = initialize_resultsLists(
                    "class_alpha_RSD"
                )
                className = 1
                for item in All_class_data_alpha_RSD:
                    print(str("class " + str(className)))
                    inputdata_adj, lm_adj, m, c = perform_G_C_adjustment(item.copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str("G-C" + "_" + "class_" + str(className))
                    adjustment_name = str("G-C" + "_alphaRSD_" + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(
                        ResultsLists_class_alpha_RSD,
                        "class_alpha_RSD",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_RSD,
                    ResultsLists_class_alpha_RSD,
                    "alpha_RSD",
                )

            if cup_alphaFlag:
                print("Applying Adjustment Method: G-C by stability class Alpha w/cup")
                logger.info(
                    "Applying Adjustment Method: G-C by stability class Alpha w/cup"
                )
                ResultsLists_class_alpha_Ane = initialize_resultsLists(
                    "class_alpha_Ane"
                )
                className = 1
                for item in All_class_data_alpha_Ane:
                    print(str("class " + str(className)))
                    inputdata_adj, lm_adj, m, c = perform_G_C_adjustment(item.copy())
                    lm_adj["sensor"] = config.model
                    lm_adj["height"] = config.height
                    lm_adj["adjustment"] = str(
                        "G-C" + "_alphaCup_" + "class_" + str(className)
                    )
                    adjustment_name = str("G-C" + "_" + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(
                        ResultsLists_class_alpha_Ane,
                        "class_alpha_Ane",
                        adjustment_name,
                        lm_adj,
                        inputdata_adj,
                        data.timestamps,
                        method,
                    )
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
                    ResultsLists_stability_alpha_Ane,
                    ResultsLists_class_alpha_Ane,
                    "alpha_Ane",
                )

        # ************************ #
        # Global Comprehensive (G-Match)
        if method != "G-Match":
            pass
        elif method == "G-Match" and config.adjustments_metadata["G-Match"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-Match")
            logger.info("Applying Adjustment Method: G-Match")

        # ************************ #
        # Global Comprehensive (G-Ref-S)
        if method != "G-Ref-S":
            pass
        elif method == "G-Ref-S" and config.adjustments_metadata["G-Ref-S"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-Ref-S")
            logger.info("Applying Adjustment Method: G-Ref-S")

        # ************************ #
        # Global Comprehensive (G-Ref-Sf)
        if method != "G-Ref-Sf":
            pass
        elif method == "G-Ref-Sf" and config.adjustments_metadata["G-Ref-Sf"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-Ref-Sf")
            logger.info("Applying Adjustment Method: G-Ref-Sf")

        # ************************ #
        # Global Comprehensive (G-Ref-SS)
        if method != "G-Ref-SS":
            pass
        elif method == "G-Ref-SS" and config.adjustments_metadata["G-Ref-SS"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-Ref-SS")
            logger.info("Applying Adjustment Method: G-Ref-SS")
        # ************************ #
        # Global Comprehensive (G-Ref-SS-S)
        if method != "G-Ref-SS-S":
            pass
        elif (
            method == "G-Ref-SS-S"
            and config.adjustments_metadata["G-Ref-SS-S"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: G-Ref-SS-S")
            logger.info("Applying Adjustment Method: G-Ref-SS-S")
        # ************************ #
        # Global Comprehensive (G-Ref-WS-Std)
        if method != "G-Ref-WS-Std":
            pass
        elif (
            method == "G-Ref-WS-Std"
            and config.adjustments_metadata["G-Ref-WS-Std"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: G-Ref-WS-Std")
            logger.info("Applying Adjustment Method: G-Ref-WS-Std")

        # ***************************************** #
        # Global LTERRA WC 1Hz Data (G-LTERRA_WC_1Hz)
        if method != "G-LTERRA_WC_1Hz":
            pass
        elif (
            method == "G-LTERRA_WC_1Hz"
            and config.adjustments_metadata["G-LTERRA_WC_1Hz"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: G-LTERRA_WC_1Hz")
            logger.info("Applying Adjustment Method: G-LTERRA_WC_1Hz")

        # ************************************************ #
        # Global LTERRA ZX Machine Learning (G-LTERRA_ZX_ML)
        if method != "G-LTERRA_ZX_ML":
            pass
        elif config.adjustments_metadata["G-LTERRA_ZX_ML"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-LTERRA_ZX_ML")
            logger.info("Applying Adjustment Method: G-LTERRA_ZX_ML")

        # ************************************************ #
        # Global LTERRA WC Machine Learning (G-LTERRA_WC_ML)
        if method != "G-LTERRA_WC_ML":
            pass
        elif config.adjustments_metadata["G-LTERRA_WC_ML"] == False:
            pass
        else:
            print("Applying Adjustment Method: G-LTERRA_WC_ML")
            logger.info("Applying Adjustment Method: G-LTERRA_WC_ML")

        # ************************************************** #
        # Global LTERRA WC w/Stability 1Hz (G-LTERRA_WC_S_1Hz)
        if method != "G-LTERRA_WC_S_1Hz":
            pass
        elif (
            method == "G-LTERRA_WC_S_1Hz"
            and config.adjustments_metadata["G-LTERRA_WC_S_1Hz"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: G-LTERRA_WC_S_1Hz")
            logger.info("Applying Adjustment Method: G-LTERRA_WC_S_1Hz")

        # ************************************************************** #
        # Global LTERRA WC w/Stability Machine Learning (G-LTERRA_WC_S_ML)
        if method != "G-LTERRA_WC_S_ML":
            pass
        elif (
            method == "G-LTERRA_WC_S_ML"
            and config.adjustments_metadata["G-LTERRA_WC_S_ML"] == False
        ):
            pass
        else:
            print("Applying Adjustment Method: G-LTERRA_WC_S_ML")
            logger.info("Applying Adjustment Method: G-LTERRA_WC_S_ML")

    if data.RSD_alphaFlag:
        pass
    else:
        ResultsLists_stability_alpha_RSD = ResultsList_stability

    if cup_alphaFlag:
        pass
    else:
        ResultsLists_stability_alpha_Ane = ResultsList_stability

    if config.RSDtype["Selection"][0:4] != "Wind":
        reg_results_class1 = np.nan
        reg_results_class2 = np.nan
        reg_results_class3 = np.nan
        reg_results_class4 = np.nan
        reg_results_class5 = np.nan
        TI_MBEList_stability = np.nan
        TI_DiffList_stability = np.nan
        TI_DiffRefBinsList_stability = np.nan
        TI_RMSEList_stability = np.nan
        RepTI_MBEList_stability = np.nan
        RepTI_DiffList_stability = np.nan
        RepTI_DiffRefBinsList_stability = np.nan
        RepTI_RMSEList_stability = np.nan
        rep_TI_results_1mps_List_stability = np.nan
        rep_TI_results_05mps_List_stability = np.nan
        TIBinList_stability = np.nan
        TIRefBinList_stability = np.nan
        total_StatsList_stability = np.nan
        belownominal_statsList_stability = np.nan
        abovenominal_statsList_stability = np.nan
        lm_adjList_stability = np.nan
        adjustmentTagList_stability = np.nan
        Distibution_statsList_stability = np.nan
        sampleTestsLists_stability = np.nan

    # Write 10 minute Adjusted data to a csv file
    config.outpath_dir = os.path.dirname(config.results_file)
    config.outpath_file = os.path.basename(config.results_file)
    config.outpath_file = str(
        "TI_10minuteAdjusted_" + config.outpath_file.split(".xlsx")[0] + ".csv"
    )
    out_dir = os.path.join(config.outpath_dir, config.outpath_file)

    TI_10minuteAdjusted.to_csv(out_dir)

    write_all_resultstofile(
        reg_results,
        baseResultsLists,
        count_1mps,
        count_05mps,
        count_1mps_train,
        count_05mps_train,
        count_1mps_test,
        count_05mps_test,
        name_1mps_tke,
        name_1mps_alpha_Ane,
        name_1mps_alpha_RSD,
        name_05mps_tke,
        name_05mps_alpha_Ane,
        name_05mps_alpha_RSD,
        count_05mps_tke,
        count_05mps_alpha_Ane,
        count_05mps_alpha_RSD,
        count_1mps_tke,
        count_1mps_alpha_Ane,
        count_1mps_alpha_RSD,
        config.results_file,
        config.site_metadata,
        config.config_metadata,
        data.timestamps,
        timestamp_train,
        timestamp_test,
        regimeBreakdown_tke,
        regimeBreakdown_ane,
        regimeBreakdown_rsd,
        Ht_1_ane,
        Ht_2_ane,
        config.extrap_metadata,
        reg_results_class1,
        reg_results_class2,
        reg_results_class3,
        reg_results_class4,
        reg_results_class5,
        reg_results_class1_alpha,
        reg_results_class2_alpha,
        reg_results_class3_alpha,
        reg_results_class4_alpha,
        reg_results_class5_alpha,
        data.Ht_1_rsd,
        data.Ht_2_rsd,
        ResultsLists_stability,
        ResultsLists_stability_alpha_RSD,
        ResultsLists_stability_alpha_Ane,
        stabilityFlag,
        cup_alphaFlag,
        data.RSD_alphaFlag,
        TimeTestA_baseline_df,
        TimeTestB_baseline_df,
        TimeTestC_baseline_df,
        time_test_A_adjustment_df,
        time_test_B_adjustment_df,
        time_test_C_adjustment_df,
    )
