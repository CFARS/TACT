try:
    from TACT import logger
except ImportError:
    pass

from TACT.computation.calculations import log_of_ratio
from TACT.computation.adjustments import get_all_regressions

import math
import numpy as np
import pandas as pd
import sys

def get_baseline_stability_results(RSD_h, All_class_data, naming_str): 

    reg_results_class1 = []
    reg_results_class2 = []
    reg_results_class3 = []
    reg_results_class4 = []
    reg_results_class5 = []
 
    for h in RSD_h:
            idx = RSD_h.index(h)
            df = All_class_data[0]
            df = df[idx]
            reg_results_class1.append(
                get_all_regressions(df, title=str(naming_str + h + "class1"))
            )
            df = All_class_data[1]
            df = df[idx]
            reg_results_class2.append(
                get_all_regressions(df, title=str(naming_str + h + "class2"))
            )
            df = All_class_data[2]
            df = df[idx]
            reg_results_class3.append(
                get_all_regressions(df, title=str(naming_str + h + "class3"))
            )
            df = All_class_data[3]
            df = df[idx]
            reg_results_class4.append(
                get_all_regressions(df, title=str(naming_str + h + "class4"))
            )
            df = All_class_data[4]
            df = df[idx]
            reg_results_class5.append(
                get_all_regressions(df, title=str(naming_str + h + "class5"))
            )

    return reg_results_class1, reg_results_class2, reg_results_class3, reg_results_class4, reg_results_class5

def get_stability_df_list(data, stabilityClass_df): 
    """
    returns a list of lists. 5 outer lists (representing stability class). Wihtin each outer list, there is a list of dataframes
    with one dateframe for each height
    should update this to a better structure
    """

    inputdata_class1 = []
    inputdata_class2 = []
    inputdata_class3 = []
    inputdata_class4 = []
    inputdata_class5 = []
    RSD_h = []

    Alldata_inputdata = data
    if isinstance(stabilityClass_df, pd.DataFrame):
        cols = stabilityClass_df.columns.to_list()
    else:
        cols = [stabilityClass_df.name]
        Alldata_inputdata[cols[0]] = stabilityClass_df.values
    for h in cols:
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

    return RSD_h, All_class_data, All_class_data_clean 

def calculate_stability_TKE(data, config):
    """
    From Wharton and Lundquist 2012
    stability class from TKE categories:
    [1]     strongly stable -------- TKE < 0.4 m^(2)/s^(-2))
    [2]              stable -------- 0.4 < TKE < 0.7 m^(2)/s^(-2))
    [3]        near-neutral -------- 0.7 < TKE < 1.0 m^(2)/s^(-2))
    [4]          convective -------- 1.0 < TKE < 1.4 m^(2)/s^(-2))
    [5] strongly convective --------  TKE > 1.4 m^(2)/s^(-2))

    Parameters
    ----------
    inputdata : DataFrame
    config : Config object

    Returns
    -------
    tuple of pandas dataframes that represent the TKE class of each obervation by height
        stabilityClass, stabilityMetric, regimeBreakdown
    """
    regimeBreakdown = pd.DataFrame()

    # check to see if instrument type allows the calculation
    if config.RSDtype["Selection"] == "Triton":
        print("Triton TKE calc")
    elif "ZX" in config.RSDtype["Selection"]:
        # look for pre-calculated TKE column
        TKE_cols = [s for s in data.inputdata.columns.to_list() if "TKE" in s or "tke" in s]
        if len(TKE_cols) < 1:
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!! Warning: Input data does not include calculated TKE. Exiting tool. Either add TKE to input data or contact aea@nrgsystems.com for assistence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            sys.exit()
        else:
            for t in TKE_cols:
                name_stabilityClass = str(t + "_class")
                data.inputdata[name_stabilityClass] = data.inputdata[t]
                data.inputdata.loc[(data.inputdata[t] <= 0.4), name_stabilityClass] = 1
                data.inputdata.loc[
                    (data.inputdata[t] > 0.4) & (data.inputdata[t] <= 0.7), name_stabilityClass
                ] = 2
                data.inputdata.loc[
                    (data.inputdata[t] > 0.7) & (data.inputdata[t] <= 1.0), name_stabilityClass
                ] = 3
                data.inputdata.loc[
                    (data.inputdata[t] > 1.0) & (data.inputdata[t] <= 1.4), name_stabilityClass
                ] = 4
                data.inputdata.loc[(data.inputdata[t] > 1.4), name_stabilityClass] = 5

                # get count and percent of data in each class
                numNans = data.inputdata[t].isnull().sum()
                totalCount = len(data.inputdata) - numNans
                regimeBreakdown[name_stabilityClass] = [
                    "1 (strongly stable)",
                    "2 (stable)",
                    "3 (near-neutral)",
                    "4 (convective)",
                    "5 (strongly convective)",
                ]
                name_count = str(name_stabilityClass.split("_class")[0] + "_count")
                regimeBreakdown[name_count] = [
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 1)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 2)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 3)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 4)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 5)]),
                ]
                name_percent = str(name_stabilityClass.split("_class")[0] + "_percent")
                regimeBreakdown[name_percent] = [
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 1)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 2)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 3)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 4)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 5)]) / totalCount,
                ]

    elif "WindCube" in config.RSDtype["Selection"]:
        # convert to radians
        dir_cols = [s for s in data.inputdata.columns.to_list() if "Direction" in s]
        if len(dir_cols) == 0:
            stabilityClass = None
            stabilityMetric = None
            regimeBreakdown = None
            print(
                "Warning: Could not find direction columns in configuration key.  TKE derived stability, check data."
            )
            sys.exit()
        else:
            for c in dir_cols:
                name_radians = str(c + "_radians")
                data.inputdata[name_radians] = data.inputdata[c] * (math.pi / 180)
                if name_radians.split("_")[2] == "radians":
                    name_u_std = str(name_radians.split("_")[0] + "_u_std")
                    name_v_std = str(name_radians.split("_")[0] + "_v_std")
                else:
                    name_u_std = str(
                        name_radians.split("_")[0]
                        + "_"
                        + name_radians.split("_")[2]
                        + "_u_std"
                    )
                    name_v_std = str(
                        name_radians.split("_")[0]
                        + "_"
                        + name_radians.split("_")[2]
                        + "_v_std"
                    )
                name_dispersion = None
                name_std = c.replace("Direction", "SD")
                data.inputdata[name_u_std] = data.inputdata[name_std] * np.cos(
                    data.inputdata[name_radians]
                )
                data.inputdata[name_v_std] = data.inputdata[name_std] * np.sin(
                    data.inputdata[name_radians]
                )
                name_tke = str(name_u_std.split("_u")[0] + "_LidarTKE")
                data.inputdata[name_tke] = 0.5 * (
                    data.inputdata[name_u_std] ** 2
                    + data.inputdata[name_v_std] ** 2
                    + data.inputdata[name_std] ** 2
                )
                name_stabilityClass = str(name_tke + "_class")
                data.inputdata[name_stabilityClass] = data.inputdata[name_tke]
                data.inputdata.loc[(data.inputdata[name_tke] <= 0.4), name_stabilityClass] = 1
                data.inputdata.loc[
                    (data.inputdata[name_tke] > 0.4) & (data.inputdata[name_tke] <= 0.7),
                    name_stabilityClass,
                ] = 2
                data.inputdata.loc[
                    (data.inputdata[name_tke] > 0.7) & (data.inputdata[name_tke] <= 1.0),
                    name_stabilityClass,
                ] = 3
                data.inputdata.loc[
                    (data.inputdata[name_tke] > 1.0) & (data.inputdata[name_tke] <= 1.4),
                    name_stabilityClass,
                ] = 4
                data.inputdata.loc[(data.inputdata[name_tke] > 1.4), name_stabilityClass] = 5

                # get count and percent of data in each class
                numNans = data.inputdata[name_tke].isnull().sum()
                totalCount = len(data.inputdata) - numNans
                name_class = str(name_u_std.split("_u")[0] + "_class")
                regimeBreakdown[name_class] = [
                    "1 (strongly stable)",
                    "2 (stable)",
                    "3 (near-neutral)",
                    "4 (convective)",
                    "5 (strongly convective)",
                ]
                name_count = str(name_u_std.split("_u")[0] + "_count")
                regimeBreakdown[name_count] = [
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 1)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 2)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 3)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 4)]),
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 5)]),
                ]
                name_percent = str(name_u_std.split("_u")[0] + "_percent")
                regimeBreakdown[name_percent] = [
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 1)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 2)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 3)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 4)]) / totalCount,
                    len(data.inputdata[(data.inputdata[name_stabilityClass] == 5)]) / totalCount,
                ]

    else:
        print("Warning: Due to senor type, TKE is not being calculated.")
        stabilityClass = None
        stabilityMetric = None
        regimeBreakdown = None

    classCols = [s for s in data.inputdata.columns.to_list() if "_class" in s]
    stabilityClass = data.inputdata[classCols]
    tkeCols = [
        s
        for s in data.inputdata.columns.to_list()
        if "_LidarTKE" in s or "TKE" in s or "tke" in s
    ]
    tkeCols = [s for s in tkeCols if "_class" not in s]
    stabilityMetric = data.inputdata[tkeCols]

    return stabilityClass, stabilityMetric, regimeBreakdown


def calculate_stability_alpha(data, config):
    """
    From Wharton and Lundquist 2012
    stability class from shear exponent categories:
    [1]     strongly stable -------- alpha > 0.3
    [2]              stable -------- 0.2 < alpha < 0.3
    [3]        near-neutral -------- 0.1 < TKE < 0.2
    [4]          convective -------- 0.0 < TKE < 0.1
    [5] strongly convective -------- alpha < 0.0

    Parameters
    ----------
    inputdata : DataFrame
    config : object
        TACT.Config object

    Returns
    -------
    tuple
        cup_alphaFlag
        stabilityClass_ane
        stabilityMetric_ane
        regimeBreakdown_ane
        Ht_1_ane
        Ht_2_ane
        stabilityClass_rsd
        stabilityMetric_rsd
        regimeBreakdown_rsd
    """

    regimeBreakdown_ane = pd.DataFrame()

    # check for 2 anemometer heights (use furthest apart) for cup alpha calculation
    configHtData = pd.read_excel(config.config_file, usecols=[3, 4], nrows=17).iloc[
        [3, 12, 13, 14, 15]
    ]
    primaryHeight = configHtData["Selection"].to_list()[0]
    (
        all_heights,
        ane_heights,
        RSD_heights,
        ane_cols,
        RSD_cols,
    ) = config.check_for_additional_heights(primaryHeight)
    if len(list(ane_heights)) > 1:
        all_keys = list(all_heights.values())
        max_key = list(all_heights.keys())[all_keys.index(max(all_heights.values()))]
        min_key = list(all_heights.keys())[all_keys.index(min(all_heights.values()))]
        if max_key == "primary":
            max_cols = [
                s for s in data.inputdata.columns.to_list() if "Ref" in s and "WS" in s
            ]
        else:
            subname = str("Ht" + str(max_key))
            max_cols = [
                s
                for s in data.inputdata.columns.to_list()
                if subname in s and "Ane" in s and "WS" in s
            ]
        if min_key == "primary":
            min_cols = [
                s for s in data.inputdata.columns.to_list() if "Ref" in s and "WS" in s
            ]
        else:
            subname = str("Ht" + str(min_key))
            min_cols = [
                s
                for s in data.inputdata.columns.to_list()
                if subname in s and "Ane" in s and "WS" in s
            ]

        # Calculate shear exponent
        tmp = pd.DataFrame(None)
        baseName = str(max_cols + min_cols)
        tmp[str(baseName + "_y")] = [
            val
            for sublist in log_of_ratio(
                data.inputdata[max_cols].values.astype(float),
                data.inputdata[min_cols].values.astype(float),
            )
            for val in sublist
        ]
        tmp[str(baseName + "_alpha")] = tmp[str(baseName + "_y")] / (
            log_of_ratio(max(all_heights.values()), min(all_heights.values()))
        )

        stabilityMetric_ane = tmp[str(baseName + "_alpha")]
        Ht_2_ane = max(all_heights.values())
        Ht_1_ane = min(all_heights.values())

        tmp[str(baseName + "stabilityClass")] = tmp[str(baseName + "_alpha")]
        tmp.loc[
            (tmp[str(baseName + "_alpha")] <= 0.4), str(baseName + "stabilityClass")
        ] = 1
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 0.4)
            & (tmp[str(baseName + "_alpha")] <= 0.7),
            str(baseName + "stabilityClass"),
        ] = 2
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 0.7)
            & (tmp[str(baseName + "_alpha")] <= 1.0),
            str(baseName + "stabilityClass"),
        ] = 3
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 1.0)
            & (tmp[str(baseName + "_alpha")] <= 1.4),
            str(baseName + "stabilityClass"),
        ] = 4
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 1.4), str(baseName + "stabilityClass")
        ] = 5

        # get count and percent of data in each class
        numNans = tmp[str(baseName) + "_alpha"].isnull().sum()
        totalCount = len(data.inputdata) - numNans
        name_class = str("stability_shear" + "_class")
        name_stabilityClass = str(baseName + "stabilityClass")
        regimeBreakdown_ane[name_class] = [
            "1 (strongly stable)",
            "2 (stable)",
            "3 (near-neutral)",
            "4 (convective)",
            "5 (strongly convective)",
        ]
        name_count = str("stability_shear_obs" + "_count")
        regimeBreakdown_ane[name_count] = [
            len(tmp[(tmp[name_stabilityClass] == 1)]),
            len(tmp[(tmp[name_stabilityClass] == 2)]),
            len(tmp[(tmp[name_stabilityClass] == 3)]),
            len(tmp[(tmp[name_stabilityClass] == 4)]),
            len(tmp[(tmp[name_stabilityClass] == 5)]),
        ]
        name_percent = str("stability_shear_obs" + "_percent")
        regimeBreakdown_ane[name_percent] = [
            len(tmp[(tmp[name_stabilityClass] == 1)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 2)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 3)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 4)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 5)]) / totalCount,
        ]
        stabilityClass_ane = tmp[name_stabilityClass]
        cup_alphaFlag = True
    else:
        stabilityClass_ane = None
        stabilityMetric_ane = None
        regimeBreakdown_ane = None
        Ht_1_ane = None
        Ht_2_ane = None
        cup_alphaFlag = False

    # If possible, perform stability calculation with RSD data
    if data.RSD_alphaFlag:
        regimeBreakdown_rsd = pd.DataFrame()
        tmp = pd.DataFrame(None)
        baseName = str("WS_" + str(data.Ht_1_rsd) + "_" + "WS_" + str(data.Ht_2_rsd))
        max_col = "RSD_alpha_lowHeight"
        min_col = "RSD_alpha_highHeight"
        tmp[str(baseName + "_y")] = log_of_ratio(
            data.inputdata[max_col].values.astype(float),
            data.inputdata[min_col].values.astype(float),
        )
        tmp[str(baseName + "_alpha")] = tmp[str(baseName + "_y")] / (
            log_of_ratio(data.Ht_2_rsd, data.Ht_1_rsd)
        )

        stabilityMetric_rsd = tmp[str(baseName + "_alpha")]

        tmp[str(baseName + "stabilityClass")] = tmp[str(baseName + "_alpha")]
        tmp.loc[
            (tmp[str(baseName + "_alpha")] <= 0.4), str(baseName + "stabilityClass")
        ] = 1
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 0.4)
            & (tmp[str(baseName + "_alpha")] <= 0.7),
            str(baseName + "stabilityClass"),
        ] = 2
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 0.7)
            & (tmp[str(baseName + "_alpha")] <= 1.0),
            str(baseName + "stabilityClass"),
        ] = 3
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 1.0)
            & (tmp[str(baseName + "_alpha")] <= 1.4),
            str(baseName + "stabilityClass"),
        ] = 4
        tmp.loc[
            (tmp[str(baseName + "_alpha")] > 1.4), str(baseName + "stabilityClass")
        ] = 5

        # get count and percent of data in each class
        numNans = tmp[str(baseName) + "_alpha"].isnull().sum()
        totalCount = len(data.inputdata) - numNans
        name_stabilityClass = str(baseName + "stabilityClass")
        regimeBreakdown_rsd[name_class] = [
            "1 (strongly stable)",
            "2 (stable)",
            "3 (near-neutral)",
            "4 (convective)",
            "5 (strongly convective)",
        ]
        name_count = str("stability_shear_obs" + "_count")
        regimeBreakdown_rsd[name_count] = [
            len(tmp[(tmp[name_stabilityClass] == 1)]),
            len(tmp[(tmp[name_stabilityClass] == 2)]),
            len(tmp[(tmp[name_stabilityClass] == 3)]),
            len(tmp[(tmp[name_stabilityClass] == 4)]),
            len(tmp[(tmp[name_stabilityClass] == 5)]),
        ]
        name_percent = str("stability_shear_obs" + "_percent")
        regimeBreakdown_rsd[name_percent] = [
            len(tmp[(tmp[name_stabilityClass] == 1)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 2)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 3)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 4)]) / totalCount,
            len(tmp[(tmp[name_stabilityClass] == 5)]) / totalCount,
        ]
        stabilityClass_rsd = tmp[name_stabilityClass]
    else:
        stabilityClass_rsd = None
        stabilityMetric_rsd = None
        regimeBreakdown_rsd = None
        data.Ht_1_rsd = None
        data.Ht_2_rsd = None

    return (
        cup_alphaFlag,
        stabilityClass_ane,
        stabilityMetric_ane,
        regimeBreakdown_ane,
        Ht_1_ane,
        Ht_2_ane,
        stabilityClass_rsd,
        stabilityMetric_rsd,
        regimeBreakdown_rsd,
    )
