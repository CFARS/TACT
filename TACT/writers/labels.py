try:
    from TACT import logger
except ImportError:
    pass
import pandas as pd


from TACT.computation.TI import (
    get_representative_TI,
    get_TI_bybin,
    get_TI_Diff_r,
    get_TI_byTIrefbin,
    get_description_stats,
    Dist_stats,
)


def initialize_resultsLists(appendString):
    resultsLists = {}
    resultsLists[str("TI_MBEList" + "_" + appendString)] = []
    resultsLists[str("TI_DiffList" + "_" + appendString)] = []
    resultsLists[str("TI_DiffRefBinsList" + "_" + appendString)] = []
    resultsLists[str("TI_RMSEList" + "_" + appendString)] = []
    resultsLists[str("RepTI_MBEList" + "_" + appendString)] = []
    resultsLists[str("RepTI_DiffList" + "_" + appendString)] = []
    resultsLists[str("RepTI_DiffRefBinsList" + "_" + appendString)] = []
    resultsLists[str("RepTI_RMSEList" + "_" + appendString)] = []
    resultsLists[str("rep_TI_results_1mps_List" + "_" + appendString)] = []
    resultsLists[str("rep_TI_results_05mps_List" + "_" + appendString)] = []
    resultsLists[str("TIBinList" + "_" + appendString)] = []
    resultsLists[str("TIRefBinList" + "_" + appendString)] = []
    resultsLists[str("total_StatsList" + "_" + appendString)] = []
    resultsLists[str("belownominal_statsList" + "_" + appendString)] = []
    resultsLists[str("abovenominal_statsList" + "_" + appendString)] = []
    resultsLists[str("lm_adjList" + "_" + appendString)] = []
    resultsLists[str("adjustmentTagList" + "_" + appendString)] = []
    resultsLists[str("Distribution_statsList" + "_" + appendString)] = []
    resultsLists[str("sampleTestsLists" + "_" + appendString)] = []
    return resultsLists


def populate_resultsLists(
    appendString,
    adjustment_name,
    lm_adj,
    inputdata_adj,
    Timestamps,
    method,
    emptyclassFlag=False,
):
    resultDict = {}

    if isinstance(inputdata_adj, pd.DataFrame) == False:
        emptyclassFlag = True
    elif inputdata_adj.empty:
        emptyclassFlag = True
    else:
        try:
            (
                TI_MBE_j_,
                TI_Diff_j_,
                TI_RMSE_j_,
                RepTI_MBE_j_,
                RepTI_Diff_j_,
                RepTI_RMSE_j_,
            ) = get_TI_MBE_Diff_j(inputdata_adj)
            TI_Diff_r_, RepTI_Diff_r_ = get_TI_Diff_r(inputdata_adj)
            rep_TI_results_1mps, rep_TI_results_05mps = get_representative_TI(
                inputdata_adj
            )  # char TI but at bin level
            TIbybin = get_TI_bybin(inputdata_adj)
            TIbyRefbin = get_TI_byTIrefbin(inputdata_adj)
            total_stats, belownominal_stats, abovenominal_stats = get_description_stats(
                inputdata_adj
            )

        except:
            emptyclassFlag = True

    if emptyclassFlag == True:
        resultDict[str("TI_MBEList" + "_" + appendString)]=(None)
        resultDict[str("TI_DiffList" + "_" + appendString)]=(None)
        resultDict[str("TI_DiffRefBinsList" + "_" + appendString)]=(None)
        resultDict[str("TI_RMSEList" + "_" + appendString)]=(None)
        resultDict[str("RepTI_MBEList" + "_" + appendString)]=(None)
        resultDict[str("RepTI_DiffList" + "_" + appendString)]=(None)
        resultDict[str("RepTI_DiffRefBinsList" + "_" + appendString)]=(None)
        resultDict[str("RepTI_RMSEList" + "_" + appendString)]=(None)
        resultDict[str("rep_TI_results_1mps_List" + "_" + appendString)]=(None)
        resultDict[str("rep_TI_results_05mps_List" + "_" + appendString)]=(None)
        resultDict[str("TIBinList" + "_" + appendString)]=(None)
        resultDict[str("TIRefBinList" + "_" + appendString)]=(None)
        resultDict[str("total_StatsList" + "_" + appendString)]=(None)
        resultDict[str("belownominal_statsList" + "_" + appendString)]=(None)
        resultDict[str("abovenominal_statsList" + "_" + appendString)]=(None)
        resultDict[str("lm_adjList" + "_" + appendString)]=(lm_adj)
        resultDict[str("adjustmentTagList" + "_" + appendString)]=(method)
        resultDict[str("Distribution_statsList" + "_" + appendString)]=(None)
        resultDict[str("sampleTestsLists" + "_" + appendString)]=(None)

    else:
        resultDict[str("TI_MBEList" + "_" + appendString)]=(TI_MBE_j_)
        resultDict[str("TI_DiffList" + "_" + appendString)]=(TI_Diff_j_)
        resultDict[str("TI_DiffRefBinsList" + "_" + appendString)]=(TI_Diff_r_)
        resultDict[str("TI_RMSEList" + "_" + appendString)]=(TI_RMSE_j_)
        resultDict[str("RepTI_MBEList" + "_" + appendString)]=(RepTI_MBE_j_)
        resultDict[str("RepTI_DiffList" + "_" + appendString)]=(RepTI_Diff_j_)
        resultDict[str("RepTI_DiffRefBinsList" + "_" + appendString)]=(RepTI_Diff_r_)
        resultDict[str("RepTI_RMSEList" + "_" + appendString)]=(RepTI_RMSE_j_)
        resultDict[str("rep_TI_results_1mps_List" + "_" + appendString)]=(rep_TI_results_1mps)
        resultDict[str("rep_TI_results_05mps_List" + "_" + appendString)]=(rep_TI_results_05mps)
        resultDict[str("TIBinList" + "_" + appendString)]=(TIbybin)
        resultDict[str("TIRefBinList" + "_" + appendString)]=(TIbyRefbin)
        resultDict[str("total_StatsList" + "_" + appendString)]=(total_stats)
        resultDict[str("belownominal_statsList" + "_" + appendString)]=(belownominal_stats)
        resultDict[str("abovenominal_statsList" + "_" + appendString)]=(abovenominal_stats)
        resultDict[str("lm_adjList" + "_" + appendString)]=(lm_adj)
        resultDict[str("adjustmentTagList" + "_" + appendString)]=(method)
    try:
        Distribution_stats, sampleTests = Dist_stats(
            inputdata_adj, Timestamps, adjustment_name
        )
        resultDict[str("Distribution_statsList" + "_" + appendString)]=(
            Distribution_stats
        )
        resultDict[str("sampleTestsLists" + "_" + appendString)]=(sampleTests)

    except:
        resultDict[str("Distribution_statsList" + "_" + appendString)]=(None)
        resultDict[str("sampleTestsLists" + "_" + appendString)]=(None)

    return resultDict


def populate_resultsLists_stability(
    ResultsLists_stability, ResultsLists_class, appendString
):

    ResultsLists_stability[str("TI_MBEList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("TI_MBEList_class_" + appendString)]
    )
    ResultsLists_stability[str("TI_DiffList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("TI_DiffList_class_" + appendString)]
    )
    ResultsLists_stability[
        str("TI_DiffRefBinsList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("TI_DiffRefBinsList_class_" + appendString)])
    ResultsLists_stability[str("TI_RMSEList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("TI_RMSEList_class_" + appendString)]
    )
    ResultsLists_stability[str("RepTI_MBEList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("RepTI_MBEList_class_" + appendString)]
    )
    ResultsLists_stability[str("RepTI_DiffList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("RepTI_DiffList_class_" + appendString)]
    )
    ResultsLists_stability[
        str("RepTI_DiffRefBinsList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("RepTI_DiffRefBinsList_class_" + appendString)])
    ResultsLists_stability[str("RepTI_RMSEList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("RepTI_RMSEList_class_" + appendString)]
    )
    ResultsLists_stability[
        str("rep_TI_results_1mps_List_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("rep_TI_results_1mps_List_class_" + appendString)])
    ResultsLists_stability[
        str("rep_TI_results_05mps_List_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("rep_TI_results_05mps_List_class_" + appendString)])
    ResultsLists_stability[str("TIBinList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("TIBinList_class_" + appendString)]
    )
    ResultsLists_stability[str("TIRefBinList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("TIRefBinList_class_" + appendString)]
    )
    ResultsLists_stability[
        str("total_StatsList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("total_StatsList_class_" + appendString)])
    ResultsLists_stability[
        str("belownominal_statsList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("belownominal_statsList_class_" + appendString)])
    ResultsLists_stability[
        str("abovenominal_statsList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("abovenominal_statsList_class_" + appendString)])
    ResultsLists_stability[str("lm_adjList_stability" + "_" + appendString)].append(
        ResultsLists_class[str("lm_adjList_class_" + appendString)]
    )
    ResultsLists_stability[
        str("adjustmentTagList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("adjustmentTagList_class_" + appendString)])
    ResultsLists_stability[
        str("Distribution_statsList_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("Distribution_statsList_class_" + appendString)])
    ResultsLists_stability[
        str("sampleTestsLists_stability" + "_" + appendString)
    ].append(ResultsLists_class[str("sampleTestsLists_class_" + appendString)])

    return ResultsLists_stability
