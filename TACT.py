"""
This is the main script to analyze projects without an NDA in place.
Authors: Nikhil Kondabala, Alexandra Arntsen, Andrew Black, Barrett Goudeau, Nigel Swytink-Binnema, Nicolas Jolin
Updated: 7/01/2021

Example command line execution:

python TACT.py -in /Users/aearntsen/cfarsMASTER/CFARSPhase3/test/518Tower_Windcube_Filtered_subset.csv -config /Users/aearntsen/cfarsMASTER/CFARSPhase3/test/configuration_518Tower_Windcube_Filtered_subset_ex.xlsx -rtd /Volumes/New\ P/DataScience/CFARS/WISE_Phase3_Implementation/RTD_chunk -res /Users/aearntsen/cfarsMASTER/CFARSPhase3/test/out.xlsx --timetestFlag

python phase3_implementation_noNDA.py -in /Users/aearntsen/cfarsMaster/cfarsMASTER/CFARSPhase3/test/NRG_canyonCFARS_data.csv -config /Users/aearntsen/cfarsMaster/CFARSPhase3/test/Configuration_template_phase3_NRG_ZX.xlsx -rtd /Volumes/New\ P/DataScience/CFARS/WISE_Phase3_Implementation/RTD_chunk -res /Users/aearntsen/cfarsMaster/CFARSPhase3/test/out.xlsx --timetestFlag

"""
try:
    from TACT import logger
except ImportError:
    pass

from TACT.computation.adjustments import Adjustments
from TACT.computation.post_adjustment import post_correction_stats
from TACT.computation.TI_computations import get_count_per_WSbin, get_stats_per_WSbin, get_stats_per_TIbin, get_RMSE_per_WSbin, get_TI_MBE_Diff_j, get_TI_Diff_r, get_representative_TI, get_TI_bybin, get_TI_byTIrefbin, get_description_stats, Dist_stats, get_representative_TI, get_representative_TI_15mps
from TACT.extrapolation.extrapolation import get_extrap_col_and_ht, get_modelRegression_extrap, log_of_ratio, perform_TI_extrapolation, change_extrap_names, extrap_configResult
from TACT.extrapolation.calculations import log_of_ratio, power_law
from TACT.readers.windcube import import_WC_file_VAD, import_WC_file_vr, import_WC_file_VAD, get_10min_spectrum_WC_raw, WC_processing_standard
from TACT.readers.config import Config
from TACT.readers.data import Data
from TACT.writers.files import write_all_resultstofile

import pandas as pd
import numpy as np
import sys
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
import argparse
import re
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings

import os
import math
import datetime
from future.utils import itervalues, iteritems
import json
import requests
from glob2 import glob
import matplotlib.pyplot as plt
from string import printable


def hist_match(inputdata_train, inputdata_test, refCol, testCol):

    test1 = inputdata_test[refCol].copy
    source = inputdata_test[refCol].copy().dropna()
    template = inputdata_train[testCol].copy().dropna()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    n_bins = 200
    #take the cumsum of the counts and normalize by the number of pixels to
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
    imhist_source,bins_source = np.histogram(source,n_bins,density=True)
    cdf_source = imhist_source.cumsum() #cumulative distribution function
    cdf_source = n_bins * cdf_source / cdf_source[-1] #normalize

    imhist_template,bins_template = np.histogram(template,n_bins,density=True)
    cdf_template = imhist_template.cumsum() #cumulative distribution function
    cdf_template = n_bins * cdf_template / cdf_template[-1] #normalize

    im2 = np.interp(source,bins_template[:-1],cdf_source)
    output = np.interp(im2,cdf_template, bins_template[:-1])

#    plt.plot(cdf_source,label='source')
#    plt.plot(cdf_template,label='template')

#    plt.legend()
#    plt.show()

    output_df = source
    output_df = output_df.to_frame()
    output_df['output'] = output
    not_outs = output_df.columns.to_list()
    not_outs = [i for i in not_outs if i != 'output']
    output_df = output_df.drop(columns= not_outs)

    res = inputdata_test.join(output_df, how='left')
    output_res = res['output'].values

    return output_res


def get_modelRegression(inputdata, column1, column2, fit_intercept=True):
    '''
    :param inputdata: input data (dataframe)
    :param column1: string, column name for x-variable
    :param column2: string, column name for y-variable
    :param columnNameOut: string, column name for predicted value
    :return: dict with output of regression
    '''
    x = inputdata[column1].values.astype(float)
    y = inputdata[column2].values.astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)
    regr = linear_model.LinearRegression(fit_intercept=fit_intercept)
    regr.fit(x, y)
    slope = regr.coef_[0][0]
    intercept = regr.intercept_[0]
    predict = regr.predict(x)
    y = y.astype(np.float)
    r = np.corrcoef(x, y)[0, 1]
    r2 = r2_score(y, predict)  # coefficient of determination, explained variance
    mse = mean_squared_error(y, predict, multioutput='raw_values')[0]
    rmse = np.sqrt(mse)
    difference = abs((x - y).mean())
    resultsDict = {'c': intercept, 'm': slope, 'r': r, 'r2': r2, 'mse': mse, 'rmse': rmse, 'predicted': predict,
               'difference': difference}
    results = [slope, intercept , r2 , difference, mse, rmse]

    return results


def get_all_regressions(inputdata, title=None):
    # get the ws regression results for all the col required pairs. Title is the name of subset of data being evaluated
    # Note the order in input to regression function. x is reference.


    pairList = [['Ref_WS','RSD_WS'],['Ref_WS','Ane2_WS'],['Ref_TI','RSD_TI'],['Ref_TI','Ane2_TI'],['Ref_SD','RSD_SD'],['Ref_SD','Ane2_SD']]

    lenFlag = False
    if len(inputdata) < 2:
        lenFlag = True

    columns = [title, 'm', 'c', 'rsquared', 'mean difference', 'mse', 'rmse']
    results = pd.DataFrame(columns=columns)

    logger.debug(f"getting regr for {title}")

    for p in pairList:

        res_name = str(p[0].split('_')[1] + '_regression_' + p[0].split('_')[0] + '_' + p[1].split('_')[0])

        if p[1] in inputdata.columns and lenFlag == False:
            _adjuster = Adjustments(inputdata)
            results_regr = [res_name] + _adjuster.get_regression(inputdata[p[0]], inputdata[p[1]])

        else:
            results_regr = [res_name, 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']

        _results = pd.DataFrame(columns=columns, data=[results_regr])
        results = pd.concat([results, _results], ignore_index=True, axis=0, join='outer')

    # labels not required
    labelsExtra = ['RSD_SD_Ht1','RSD_TI_Ht1', 'RSD_WS_Ht1','RSD_SD_Ht2', 'RSD_TI_Ht2',
                   'RSD_WS_Ht2', 'RSD_SD_Ht3', 'RSD_TI_Ht3', 'RSD_WS_Ht3',
                   'RSD_WS_Ht4', 'RSD_SD_Ht4', 'RSD_TI_Ht4']
    labelsRef = ['Ref_WS', 'Ref_TI', 'Ref_SD']
    labelsAne = ['Ane_SD_Ht1', 'Ane_TI_Ht1', 'Ane_WS_Ht1', 'Ane_SD_Ht2', 'Ane_TI_Ht2', 'Ane_WS_Ht2',
                 'Ane_SD_Ht3', 'Ane_TI_Ht3', 'Ane_WS_Ht3', 'Ane_WS_Ht4', 'Ane_SD_Ht4','Ane_TI_Ht4']

    for l in labelsExtra:

        parts = l.split('_')
        reg_type = list(set(parts).intersection(['WS', 'TI', 'SD']))

        if 'RSD' in l:
            ht_type = parts[2]
            ref_type = [s for s in labelsAne if reg_type[0] in s]
            ref_type = [s for s in ref_type if ht_type in s]

        res_name = str(reg_type[0] + '_regression_' + parts[0])

        if 'Ht' in parts[2]:
            res_name = res_name + parts[2] + '_' + ref_type[0].split('_')[0] + ref_type[0].split('_')[2]

        else:
            res_name = res_name + '_Ref'

        logger.debug(res_name)

        if l in inputdata.columns and lenFlag == False:
            _adjuster = Adjustments(inputdata)
            res = [res_name] + _adjuster.get_regression(inputdata[ref_type[0]],inputdata[l])

        else:
            res = [res_name, 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']

        logger.debug(res)

        _results = pd.DataFrame(columns=columns, data=[res])
        results = pd.concat([results, _results], ignore_index=True, axis=0, join='outer')

    return results


def perform_G_Sa_correction(inputdata,override):
    '''
    simple filtered regression results from phase2 averages with simple regressionf rom this data
    '''
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()
    _adjuster_G_Sa = Adjustments()

    if override:
        m_ph2 = override[0]
        c_ph2 = override[1]
    else:
        # set up which coefficients to use from phase 2 for testing
        if 'Wind' in RSDtype['Selection']:
            m_ph2 = 0.70695
            c_ph2 = 0.02289
        elif 'ZX' in RSDtype['Selection']:
            m_ph2 = 0.68647
            c_ph2 = 0.03901
        elif 'Triton' in RSDtype['Selection']:
            m_ph2 = 0.36532
            c_ph2 = 0.08662
        else:
            print ('Warning: Did not apply regression results from phase 2')
            inputdata = pd.DataFrame()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        full = pd.DataFrame()
        full['Ref_TI'] = inputdata_test['Ref_TI']
        full['RSD_TI'] = inputdata_test['RSD_TI']
        full = full.dropna()
        if len(full) < 2:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            model = _adjuster_G_Sa.get_regression(inputdata_train['RSD_TI'], inputdata_train['Ref_TI'])
            m = (model[0] + m_ph2)/2
            c = (model[1] + c_ph2)/2
            RSD_TI = inputdata_test['RSD_TI'].copy()
            RSD_TI = (m*RSD_TI) + c
            inputdata_test['corrTI_RSD_TI'] = RSD_TI
            inputdata_test['corrRepTI_RSD_RepTI'] = RSD_TI + 1.28 * inputdata_test['RSD_SD']
            results = post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht1']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht1']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_G_Sa.get_regression(inputdata_train['RSD_TI'], inputdata_train['Ref_TI'])
                m = (model[0] + m_ph2)/2
                c = (model[1] + c_ph2)/2
                RSD_TI = inputdata_test['RSD_TI'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata_test['corrTI_RSD_TI_Ht1'] = RSD_TI
                inputdata_test['corrRepTI_RSD_RepTI_Ht1'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht1']
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht2']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht2']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_G_Sa.get_regression(inputdata_train['RSD_TI_Ht2'],inputdata_train['Ane_TI_Ht2'])
                m = (model[0] + m_ph2)/2
                c = (model[1] + c_ph2)/2
                RSD_TI = inputdata_test['RSD_TI'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata_test['corrTI_RSD_TI_Ht2'] = RSD_TI
                inputdata_test['corrRepTI_RSD_RepTI_Ht2'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht2']
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht3']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht3']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_G_Sa.get_regression(inputdata_train['RSD_TI_Ht3'], inputdata_train['Ane_TI_Ht3'])
                m = (model[0] + m_ph2)/2
                c = (model[1] + c_ph2)/2
                RSD_TI = inputdata_test['RSD_TI'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata_test['corrTI_RSD_TI_Ht3'] = RSD_TI
                inputdata_test['corrRepTI_RSD_RepTI_Ht3'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht3']
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht4']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht4']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_G_Sa.get_regression(inputdata_train['RSD_TI_Ht4'], inputdata_train['Ane_TI_Ht4'])
                m = (model[0] + m_ph2)/2
                c = (model[1] + c_ph2)/2
                RSD_TI = inputdata_test['RSD_TI'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata_test['corrTI_RSD_TI_Ht4'] = RSD_TI
                inputdata_test['corrRepTI_RSD_RepTI_Ht4'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht4']
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    results['correction'] = ['G-Sa'] * len(results)
    results = results.drop(columns=['sensor','height'])

    return inputdata_test, results, m, c


def empirical_stdAdjustment(inputdata,results,Ref_TI_col, RSD_TI_col, Ref_SD_col, RSD_SD_col, Ref_WS_col, RSD_WS_col):
    '''
    set adjustment values
    '''
    inputdata_test = inputdata.copy()

    # get col names
    name_ref = Ref_TI_col.split('_TI')
    name_rsd = RSD_TI_col.split('_TI')
    name = RSD_TI_col.split('_TI')
    corrTI_name = str('corrTI_'+RSD_TI_col)

    if len(inputdata) < 2:
        results = post_correction_stats([None],results, Ref_TI_col, corrTI_name)
        m = np.NaN
        c = np.NaN
    else:
        # add the new columns, initialized by uncorrected Data
        tmp = str('corr'+ RSD_SD_col)
        inputdata_test[tmp] = inputdata_test[RSD_SD_col].copy()
        inputdata_test[str('corrTI_'+  RSD_TI_col)] = inputdata_test[RSD_TI_col].copy()

        inputdata_test.loc[((inputdata[Ref_WS_col] >= 4) & (inputdata_test[Ref_WS_col] < 8)), tmp] = ((1.116763*inputdata_test[tmp]) + 0.024685) - (((1.116763*inputdata_test[tmp]) + 0.024685)*0.00029)
        inputdata_test.loc[((inputdata[Ref_WS_col] >= 4) & (inputdata_test[Ref_WS_col] < 8)), corrTI_name] = inputdata_test[tmp]/inputdata_test[RSD_WS_col]

        inputdata_test.loc[((inputdata[Ref_WS_col] >= 8) & (inputdata_test[Ref_WS_col] < 12)), tmp] = ((1.064564*inputdata_test[tmp]) + 0.040596) - (((1.064564*inputdata_test[tmp]) + 0.040596)*-0.00161)
        inputdata_test.loc[((inputdata[Ref_WS_col] >= 8) & (inputdata_test[Ref_WS_col] < 12)), corrTI_name] = inputdata_test[tmp]/inputdata_test[RSD_WS_col]

        inputdata_test.loc[((inputdata[Ref_WS_col] >= 12) & (inputdata_test[Ref_WS_col] < 16)), tmp] = ((0.97865*inputdata_test[tmp]) + 0.124371) - (((0.97865*inputdata_test[tmp]) + 0.124371)*-0.00093)
        inputdata_test.loc[((inputdata[Ref_WS_col] >= 12) & (inputdata_test[Ref_WS_col] < 16)), corrTI_name] = inputdata_test[tmp]/inputdata_test[RSD_WS_col]

        results = post_correction_stats(inputdata_test,results, Ref_TI_col, corrTI_name)

    return inputdata_test,results


def perform_G_C_correction(inputdata):
    '''
    Note: comprehensive empirical correction from a dozen locations. Focuses on std. deviation
    '''
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata_test = inputdata.copy()
        inputdata = False
    else:
        inputdata_test, results = empirical_stdAdjustment(inputdata,results,'Ref_TI', 'RSD_TI', 'Ref_SD', 'RSD_SD', 'Ref_WS', 'RSD_WS')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(inputdata_test,results,'Ane_TI_Ht1','RSD_TI_Ht1', 'Ane_SD_Ht1', 'RSD_SD_Ht1','Ane_WS_Ht1', 'RSD_WS_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(inputdata_test,results,'Ane_TI_Ht2','RSD_TI_Ht2', 'Ane_SD_Ht2', 'RSD_SD_Ht2','Ane_WS_Ht2', 'RSD_WS_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(inputdata_test,results,'Ane_TI_Ht3','RSD_TI_Ht3', 'Ane_SD_Ht3', 'RSD_SD_Ht3','Ane_WS_Ht3', 'RSD_WS_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            inputdata_test, results = empirical_stdAdjustment(inputdata_test,results,'Ane_TI_Ht4','RSD_TI_Ht4', 'Ane_SD_Ht4', 'RSD_SD_Ht4','Ane_WS_Ht4', 'RSD_WS_Ht4')
    results['correction'] = ['G-C'] * len(results)
    results = results.drop(columns=['sensor','height'])
    m = np.NaN
    c = np.NaN

    return inputdata_test, results, m, c


def perform_G_SFc_correction(inputdata):
    '''
    simple filtered regression results from phase 2 averages used
    '''

    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    m = 0.7086
    c = 0.0225

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        inputdata = False
    else:
        full = pd.DataFrame()
        full['Ref_TI'] = inputdata['Ref_TI']
        full['RSD_TI'] = inputdata['RSD_TI']
        full = full.dropna()
        if len(full) < 2:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        else:
            RSD_TI = inputdata['RSD_TI'].copy()
            RSD_TI = (float(m)*RSD_TI) + float(c)
            inputdata['corrTI_RSD_TI'] = RSD_TI
            results = post_correction_stats(inputdata,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            full = pd.DataFrame()
            full['Ane_TI_Ht1'] = inputdata['Ane_TI_Ht1']
            full['RSD_TI_Ht1'] = inputdata['RSD_TI_Ht1']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
            else:
                RSD_TI = inputdata['RSD_TI_Ht1'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata['corrTI_RSD_TI_Ht1'] = RSD_TI
                results = post_correction_stats(inputdata,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            full = pd.DataFrame()
            full['Ane_TI_Ht2'] = inputdata['Ane_TI_Ht2']
            full['RSD_TI_Ht2'] = inputdata['RSD_TI_Ht2']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
            else:
                RSD_TI = inputdata['RSD_TI_Ht2'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata['corrTI_RSD_TI_Ht2'] = RSD_TI
                results = post_correction_stats(inputdata,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            full = pd.DataFrame()
            full['Ane_TI_Ht3'] = inputdata['Ane_TI_Ht3']
            full['RSD_TI_Ht3'] = inputdata['RSD_TI_Ht3']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
            else:
                RSD_TI = inputdata['RSD_TI_Ht3'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata['corrTI_RSD_TI_Ht3'] = RSD_TI
                results = post_correction_stats(inputdata,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            full = pd.DataFrame()
            full['Ane_TI_Ht4'] = inputdata['Ane_TI_Ht4']
            full['RSD_TI_Ht4'] = inputdata['RSD_TI_Ht4']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
            else:
                RSD_TI = inputdata['RSD_TI_Ht4'].copy()
                RSD_TI = (m*RSD_TI) + c
                inputdata['corrTI_RSD_TI_Ht4'] = RSD_TI
                results = post_correction_stats(inputdata,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    results['correction'] = ['G-SFa'] * len(results)
    results = results.drop(columns=['sensor','height'])
    return inputdata, results, m, c

def perform_SS_LTERRA_ML_correction(inputdata):

    inputdata_test_result = pd.DataFrame()

    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        all_train = pd.DataFrame()
        all_train['y_train'] = inputdata_train['Ref_TI'].copy()
        all_train['x_train'] = inputdata_train['RSD_TI'].copy()
        all_test = pd.DataFrame()
        all_test['y_test'] = inputdata_test['Ref_TI'].copy()
        all_test['x_test'] = inputdata_test['RSD_TI'].copy()
        all_test['TI_test'] = inputdata_test['RSD_TI'].copy()
        all_test['RSD_SD'] = inputdata_test['RSD_SD'].copy()
        all_train = all_train.dropna()
        all_test = all_test.dropna()

        if len(all_train) < 5 and len(all_test) < 5:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            m = np.NaN
            c = np.NaN
            TI_pred_RF = machine_learning_TI(all_train['x_train'], all_train['y_train'], all_test['x_test'], all_test['y_test'],'RF', all_test['TI_test'])
            all_test['corrTI_RSD_TI'] = TI_pred_RF
            all_test['Ref_TI'] = all_test['y_test']
            inputdata_test_result = pd.merge(inputdata_test,all_test,how='left')
            results = post_correction_stats(inputdata_test_result,results, 'Ref_TI','corrTI_RSD_TI')

        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns and 'RSD_SD_Ht1' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht1'].copy()
            all_train['x_train'] = inputdata_train['RSD_TI_Ht1'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht1'].copy()
            all_test['x_test'] = inputdata_test['RSD_TI_Ht1'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht1'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht1'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()

            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train['x_train'], all_train['y_train'], all_test['x_test'], all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht1'] = TI_pred_RF
                 all_test['Ane_TI_Ht1'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')

        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns and 'RSD_SD_Ht2' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht2'].copy()
            all_train['x_train'] = inputdata_train['RSD_TI_Ht2'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht2'].copy()
            all_test['x_test'] = inputdata_test['RSD_TI_Ht2'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht2'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht2'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()

            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train['x_train'], all_train['y_train'], all_test['x_test'], all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht2'] = TI_pred_RF
                 all_test['Ane_TI_Ht2'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')

        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns and 'RSD_SD_Ht3' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht3'].copy()
            all_train['x_train'] = inputdata_train['RSD_TI_Ht3'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht3'].copy()
            all_test['x_test'] = inputdata_test['RSD_TI_Ht3'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht3'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht3'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()

            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                m = np.NaN
                c = np.NaN
                TI_pred_RF = machine_learning_TI(all_train['x_train'], all_train['y_train'], all_test['x_test'], all_test['y_test'],'RF', all_test['TI_test'])
                all_test['corrTI_RSD_TI_Ht3'] = TI_pred_RF
                all_test['Ane_TI_Ht3'] = all_test['y_test']
                inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                results = post_correction_stats(inputdata_test_result, results, 'Ane_TI_Ht3', 'corrTI_RSD_TI_Ht3')

        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns and 'RSD_Sd_Ht4' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht4'].copy()
            all_train['x_train'] = inputdata_train['RSD_TI_Ht4'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht4'].copy()
            all_test['x_test'] = inputdata_test['RSD_TI_Ht4'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht4'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht4'].copy()
            all_train = all_train.dropna()

            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train['x_train'], all_train['y_train'], all_test['x_test'], all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht4'] = TI_pred_RF
                 all_test['Ane_TI_Ht4'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    if inputdata_test_result.empty:
        inputdata_test_result = inputdata_test
    return inputdata_test_result, results, m, c

def perform_SS_LTERRA_S_ML_correction(inputdata,all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols):

    inputdata_test_result = pd.DataFrame()

    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        all_train = pd.DataFrame()
        all_train['y_train'] = inputdata_train['Ref_TI'].copy()
        all_train['x_train_TI'] = inputdata_train['RSD_TI'].copy()
        all_train['x_train_TKE'] = inputdata_train['RSD_LidarTKE'].copy()
        all_train['x_train_WS'] = inputdata_train['RSD_WS'].copy()
        all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
        all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
        all_test = pd.DataFrame()
        all_test['y_test'] = inputdata_test['Ref_TI'].copy()
        all_test['x_test_TI'] = inputdata_test['RSD_TI'].copy()
        all_test['x_test_TKE'] = inputdata_test['RSD_LidarTKE'].copy()
        all_test['x_test_WS'] = inputdata_test['RSD_WS'].copy()
        all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
        all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
        all_test['TI_test'] = inputdata_test['RSD_TI'].copy()
        all_test['RSD_SD'] = inputdata_test['RSD_SD'].copy()
        all_train = all_train.dropna()
        all_test = all_test.dropna()

        if len(all_train) < 5 and len(all_test) < 5:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            m = np.NaN
            c = np.NaN
            TI_pred_RF = machine_learning_TI(all_train[all_trainX_cols],all_train[all_trainY_cols], all_test[all_testX_cols],
                                             all_test[all_testY_cols],'RF', all_test['TI_test'])
            all_test['corrTI_RSD_TI'] = TI_pred_RF
            all_test['corrRepTI_RSD_RepTI'] = TI_pred_RF + 1.28 * all_test['RSD_SD']
            all_test['Ref_TI'] = all_test['y_test']
            inputdata_test_result = pd.merge(inputdata_test,all_test,how='left')
            results = post_correction_stats(inputdata_test_result,results, 'Ref_TI','corrTI_RSD_TI')

        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns and 'RSD_SD_Ht1' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht1'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht1'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht1_LidarTKE'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht1'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht1'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht1'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht1_LidarTKE'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht1'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht1'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht1'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()

            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[all_trainX_cols],all_train[all_trainY_cols], all_test[all_testX_cols],
                                                  all_test[all_testY_cols],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht1'] = TI_pred_RF
                 all_test['Ane_TI_Ht1'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')

        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns and 'RSD_SD_Ht2' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht2'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht2'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht2_LidarTKE'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht2'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht2'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht2'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht2_LidarTKE'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht2'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht2'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht2'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[all_trainX_cols],all_train[all_trainY_cols], all_test[all_testX_cols],
                                                  all_test[all_testY_cols],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht2'] = TI_pred_RF
                 all_test['Ane_TI_Ht2'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns and 'RSD_SD_Ht3' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht3'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht3'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht3_LidarTKE'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht3'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht3'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht3'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht3_LidarTKE'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht3'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht3'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht3'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[all_trainX_cols],all_train[all_trainY_cols], all_test[all_testX_cols],
                                                  all_test[all_testY_cols],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht3'] = TI_pred_RF
                 all_test['Ane_TI_Ht3'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns and 'RSD_Sd_Ht4' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht4'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht4'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht4_LidarTKE'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht4'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht4'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht4'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht4_LidarTKE'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht4'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht4'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht4'].copy()
            all_train = all_train.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[all_trainX_cols],all_train[all_trainY_cols], all_test[all_testX_cols],
                                                  all_test[all_testY_cols],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht4'] = TI_pred_RF
                 all_test['Ane_TI_Ht4'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    if inputdata_test_result.empty:
        inputdata_test_result = inputdata_test

    return inputdata_test_result, results, m, c


def machine_learning_TI(x_train,y_train,x_test,y_test,mode,TI_test):

    if len(x_train.shape) == 1:
        y_train = np.array(y_train).reshape(-1,1).ravel()
        y_test = np.array(y_test).reshape(-1,1).ravel()
        x_train = np.array(x_train).reshape(-1,1)
        x_test = np.array(x_test).reshape(-1,1)
        TI_test = np.array(TI_test).reshape(-1,1)
    if len(x_train.shape) != 1 and x_train.shape[1] == 1:
        y_train = np.array(y_train).reshape(-1,1).ravel()
        y_test = np.array(y_test).reshape(-1,1).ravel()
        x_train = np.array(x_train).reshape(-1,1)
        x_test = np.array(x_test).reshape(-1,1)
        TI_test = np.array(TI_test).reshape(-1,1)
    else:
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        TI_test = np.array(TI_test)

    if "RF" in mode:
        from sklearn.ensemble import RandomForestRegressor
        rfc_new = RandomForestRegressor(random_state=42, n_estimators = 100)
        #rfc_new = RandomForestRegressor(random_state=42,max_features=2,n_estimators=100)
        rfc_new = rfc_new.fit(x_train,y_train.ravel())
        TI_pred = rfc_new.predict(x_test)

    if "SVR" in mode:
       from sklearn.svm import SVR
       clf = SVR(C=1.0, epsilon=0.2,kernel='poly',degree=2)
       clf.fit(x_train, y_train)
       TI_pred = clf.predict(x_test)

    if "MARS" in mode:
       from pyearth import Earth
       MARS_model = Earth()
       #MARS_model = Earth(max_terms=8,max_degree=2)
       MARS_model.fit(x_test,y_test)
       TI_pred[mask2] = MARS_model.predict(x_test)
       print(MARS_model.summary())

    if "NN" in mode:
        # import stuff
        NN_model = None

    return TI_pred

def min_diff(array_orig,array_to_find,tol):
    #Finds indices in array_orig that correspond to values closest to numbers in array_to_find with tolerance tol

    #Inputs
    #array_orig: Original array where you want to find matching values
    #array_to_find: Array of numbers to find in array_orig
    #tol: Tolerance to find matching value

    #Outputs
    #found_indices: Indices corresponding to matching values. If no values matched with desired tolerance, index will be filled by NaN.

    import numpy as np
    found_indices = []
    if not np.shape(array_to_find):
        array_to_find = [array_to_find]
    for i in array_to_find:
        min_difference = tol
        found_index_temp = np.nan
        for j in range(0,len(array_orig)):
            diff_temp = abs(i-array_orig[j])
            if diff_temp < min_difference:
                min_difference = diff_temp
                found_index_temp = j
        found_indices.append(found_index_temp)
    return np.array(found_indices)

def perform_SS_NN_correction(inputdata):

    inputdata_test_result = pd.DataFrame()
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        all_train = pd.DataFrame()
        all_train['y_train'] = inputdata_train['Ref_TI'].copy()
        all_train['x_train_TI'] = inputdata_train['RSD_TI'].copy()
        all_train['x_train_TKE'] = inputdata_train['RSD_LidarTKE_class'].copy()
        all_train['x_train_WS'] = inputdata_train['RSD_WS'].copy()
        all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
        all_train['x_train_Hour'] = inputdata_train['Hour'].copy()

        all_train['x_train_TEMP'] = inputdata_train['Temp'].copy()
        all_train['x_train_HUM'] = inputdata_train['Humidity'].copy()
        all_train['x_train_SD'] = inputdata_train['SD'].copy()
        all_train['x_train_Tshift1'] = inputdata_train['x_train_Tshift1'].copy()
        all_train['x_train_Tshift2'] = inputdata_train['x_train_Tshift3'].copy()
        all_train['x_train_Tshift3'] = inputdata_train['x_train_Tshift3'].copy()

        all_test = pd.DataFrame()
        all_test['y_test'] = inputdata_test['Ref_TI'].copy()
        all_test['x_test_TI'] = inputdata_test['RSD_TI'].copy()
        all_test['x_test_TKE'] = inputdata_test['RSD_LidarTKE_class'].copy()
        all_test['x_test_WS'] = inputdata_test['RSD_WS'].copy()
        all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
        all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
        all_test['TI_test'] = inputdata_test['RSD_TI'].copy()
        all_test['RSD_SD'] = inputdata_test['RSD_SD'].copy()
        all_train = all_train.dropna()
        all_test = all_test.dropna()
        if len(all_train) < 5 and len(all_test) < 5:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            m = np.NaN
            c = np.NaN
            TI_pred_RF = machine_learning_TI(all_train[['x_train_TI', 'x_train_TKE','x_train_WS','x_train_DIR','x_train_Hour']],
                                             all_train['y_train'], all_test[['x_test_TI','x_test_TKE','x_test_WS','x_test_DIR','x_test_Hour']],
                                             all_test['y_test'],'RF', all_test['TI_test'])
            all_test['corrTI_RSD_TI'] = TI_pred_RF
            all_test['Ref_TI'] = all_test['y_test']
            inputdata_test_result = pd.merge(inputdata_test,all_test,how='left')
            results = post_correction_stats(inputdata_test_result,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns and 'RSD_SD_Ht1' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht1'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht1'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht1_LidarTKE_class'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht1'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht1'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht1'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht1_LidarTKE_class'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht1'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht1'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht1'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[['x_train_TI', 'x_train_TKE','x_train_WS','x_train_DIR','x_train_Hour']],
                                                  all_train['y_train'], all_test[['x_test_TI','x_test_TKE','x_test_WS','x_test_DIR','x_test_Hour']],
                                                  all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht1'] = TI_pred_RF
                 all_test['Ane_TI_Ht1'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns and 'RSD_SD_Ht2' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht2'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht2'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht2_LidarTKE_class'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht2'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht2'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht2'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht2_LidarTKE_class'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht2'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht2'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht2'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[['x_train_TI', 'x_train_TKE','x_train_WS','x_train_DIR','x_train_Hour']],
                                                  all_train['y_train'], all_test[['x_test_TI','x_test_TKE','x_test_WS','x_test_DIR','x_test_Hour']],
                                                  all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht2'] = TI_pred_RF
                 all_test['Ane_TI_Ht2'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns and 'RSD_SD_Ht3' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht3'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht3'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht3_LidarTKE_class'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht3'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht3'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht3'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht3_LidarTKE_class'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht3'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht3'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht3'].copy()
            all_train = all_train.dropna()
            all_test = all_test.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[['x_train_TI', 'x_train_TKE','x_train_WS','x_train_DIR','x_train_Hour']],
                                                  all_train['y_train'], all_test[['x_test_TI','x_test_TKE','x_test_WS','x_test_DIR','x_test_Hour']],
                                                  all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht3'] = TI_pred_RF
                 all_test['Ane_TI_Ht3'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns and 'RSD_Sd_Ht4' in inputdata.columns:
            all_train = pd.DataFrame()
            all_train['y_train'] = inputdata_train['Ane_TI_Ht4'].copy()
            all_train['x_train_TI'] = inputdata_train['RSD_TI_Ht4'].copy()
            all_train['x_train_TKE'] = inputdata_train['RSD_Ht4_LidarTKE_class'].copy()
            all_train['x_train_WS'] = inputdata_train['RSD_WS_Ht4'].copy()
            all_train['x_train_DIR'] = inputdata_train['RSD_Direction'].copy()
            all_train['x_train_Hour'] = inputdata_train['Hour'].copy()
            all_test = pd.DataFrame()
            all_test['y_test'] = inputdata_test['Ane_TI_Ht4'].copy()
            all_test['x_test_TI'] = inputdata_test['RSD_TI_Ht4'].copy()
            all_test['x_test_TKE'] = inputdata_test['RSD_Ht4_LidarTKE_class'].copy()
            all_test['x_test_WS'] = inputdata_test['RSD_WS_Ht4'].copy()
            all_test['x_test_DIR'] = inputdata_test['RSD_Direction'].copy()
            all_test['x_test_Hour'] = inputdata_test['Hour'].copy()
            all_test['TI_test'] = inputdata_test['RSD_TI_Ht4'].copy()
            all_test['RSD_SD'] = inputdata_test['RSD_SD_Ht4'].copy()
            all_train = all_train.dropna()
            if len(all_train) < 5 and len(all_test) < 5:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                 m = np.NaN
                 c = np.NaN
                 TI_pred_RF = machine_learning_TI(all_train[['x_train_TI', 'x_train_TKE','x_train_WS','x_train_DIR','x_train_Hour']],
                                                  all_train['y_train'], all_test[['x_test_TI','x_test_TKE','x_test_WS','x_test_DIR','x_test_Hour']],
                                                  all_test['y_test'],'RF', all_test['TI_test'])
                 all_test['corrTI_RSD_TI_Ht4'] = TI_pred_RF
                 all_test['Ane_TI_Ht4'] = all_test['y_test']
                 inputdata_test_result = pd.merge(inputdata_test_result,all_test,how='left')
                 results = post_correction_stats(inputdata_test_result,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    if inputdata_test_result.empty:
        inputdata_test_result = inputdata_test

    return inputdata_test_result, results, m, c


def perform_G_LTERRA_WC_1HZ_correction(inputdata):
    '''
    In development
    '''

    from functools import reduce
    import glob
    import csv
    from sklearn.metrics import mean_absolute_error as MAE
    # first learn what is in the reference npz files
    # 'wd' = wind direction, 'U' = mean ws, 'u_var' = variance, 'time'

    # make the ref files (uncomment below to make new reference npz files)
#    reference_data_dir = os.path.join('/Users/aearntsen/Desktop/Sept2020_LTERRA/reference_directory/ReferenceData','NRG_canyonCFARS_data_tower.csv')
#    ref_data = pd.read_csv(reference_data_dir)
#    ref_data = ref_data[['Timestamp','tower_353012_Ch1_Anem_55.50m_WSW_Avg_m/s','tower_353012_Ch1_Anem_55.50m_WSW_SD_m/s','tower_353012_Ch13_Vane_57.00m_SSW_Avg_deg']]
#    ref_data = ref_data.rename(columns={'Timestamp':'time', 'tower_353012_Ch1_Anem_55.50m_WSW_Avg_m/s':'U','tower_353012_Ch1_Anem_55.50m_WSW_SD_m/s':'u_var','tower_353012_Ch13_Vane_57.00m_SSW_Avg_deg':'wd'})
#    for row in ref_data.index:
#        tt = datetime.datetime.strptime(ref_data['time'][row], '%m/%d/%y %H:%M').strftime('%Y%m%d_%H%M')
#        filename = str('Ref_'+ str(tt) + '_UTC.npz')
#        print (filename)
#        #filepath = os.path.join('/Users/aearntsen/Desktop/Sept2020_LTERRA/reference_directory/ProcessedReferenceData',filename)
#        filepath = os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/reference_directory/ProcessedReferenceData',filename)
#        np.savez(filepath, wd = ref_data['wd'][row], U = ref_data['U'][row], u_var = ref_data['u_var'][row], time = ref_data['time'][row])

    # make wc npz files
  #  dir_rtd = os.path.join('/Volumes/New P/DataScience/CFARS/August31_test/test','WindcubeRTDs')
    dir_rtd = os.path.join('/Users/aearntsen/L-TERRA','WindcubeRTDs')
    dir_npz = os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/lidar_directory', 'ProcessedRTDData')

    #Height where variance and wind speed should be extracted
#    height_needed = 55
    #Specify the time period for data extraction
#    years = [2019,2020]
#    months = ['09','10','11','12','01','02','03','04','05','06','07','08']
#    days = np.arange(1,31)

#    for i in years:
#        print (i)
#        for j in months:
#            print (j)
#            for k in days:
#                print (str('day' + str(k)))
#                dir_str_rtd = str(dir_rtd + '/')
#                filenames_WC = glob.glob(dir_str_rtd + '*' + str(i) + '_' + j + '_' + str(k).zfill(2) + '__' + '*')
#                print (filenames_WC)
#                if len(filenames_WC) != 0:
#                    for jj in filenames_WC:
#                        print (jj)
#                        #Extract raw rotated wind speed, 10-min. mean wind speed, 10-min. mean wind direction, 10-min. shear parameter,
#                        #and 10-min. timestamp using DBS technique. (DBS is the raw output from the .rtd files.)
#                        try:
#                            [u_rot_DBS,U_DBS,wd_DBS,p,time_datenum_10min,extra1,extra2,extra3,extra4] = WC_processing_standard(jj,"raw",height_needed)
#                            #Calculate 10-min. streamwise variance
#                            u_var_DBS = get_10min_var(u_rot_DBS,1.)
#                        except:
#                            pass
#                        #Extract rotated streamwise wind speed from the VAD technique, 10-min. mean wind speed, and rotated vertical
#                        #wind speed
#                        try:
#                            [u_rot_VAD,U_VAD,w_rot, time_datenum_10min_VAD,time_interp_VAD] = WC_processing_standard(jj,"VAD",height_needed)
#                            #Calculate 10-min. streamwise and vertical variance
#                            u_var_VAD = get_10min_var(u_rot_VAD,1./4)
#                            w_var = get_10min_var(w_rot,1./4)
#                        except:
#                            pass

#                        try:
                            #Extract raw off-vertical radial wind speed components
#                            [vr_n,vr_e,vr_s,vr_w] = WC_processing_standard(jj,"vr",height_needed)
#                        except:
#                            pass
                       # [vr_n,vr_n_dispersion_interp,vr_e,vr_e_dispersion_interp,vr_s,vr_s_dispersion_interp,vr_w,vr_w_dispersion_interp,
                       #  vert_beam_rot,vert_beam_dispersion_interp,heights,U,time_datenum_10min,hub_height_index,SNR_n_interp,SNR_e_interp,
                       #  SNR_s_interp,SNR_w_interp,SNR_vert_beam_interp] = WC_processing_standard(jj,"vr",height_needed)

                        #Write output to 10-min. files
#                        for kk in range(len(time_datenum_10min)):
#                            print (kk)
#                            filename = dir_npz + "WC_DBS_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
#                            str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
#                            try:
#                                np.savez(filename,u_rot=u_rot_DBS[kk*600:(kk+1)*600 + 1],time=time_datenum_10min[kk],\
#                                U = U_DBS[kk],wd=wd_DBS[kk],u_var=u_var_DBS[kk],p=p[kk])
#                            except:
#                                pass
#
#                            filename = dir_npz + "WC_VAD_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
#                            str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
#                            try:
#                                np.savez(filename,u_rot=u_rot_VAD[kk*150:(kk+1)*150 + 1],time=time_datenum_10min[kk],\
#                                U = U_VAD[kk],u_var=u_var_VAD[kk],vert_beam_var=w_var[kk],w_rot = w_rot[kk*150:(kk+1)*150 + 1])
#                            except:
#                                pass

#                            filename = dir_npz + "WC_vr_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
#                            str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
#                            try:
#                                np.savez(filename,vr_n=vr_n[kk*150:(kk+1)*150 + 1],vr_e=vr_e[kk*150:(kk+1)*150 + 1],\
#                                vr_s=vr_s[kk*150:(kk+1)*150 + 1],vr_w=vr_w[kk*150:(kk+1)*150 + 1],time=time_datenum_10min[kk])
#                            except:
#                                pass

    # test corrections
#    import csv
#    from sklearn.metrics import mean_absolute_error as MAE
#    import functools

#    MAE_min = 10
#    MAE_min_s = 10
#    MAE_min_n = 10
#    MAE_min_u = 10
#    opts_best = "None"
#    opts_best_s = "None"
#    opts_best_n = "None"
#    opts_best_u = "None"
#    ##########################################################
#    #USER INPUTS GO HERE
#    #Directory where CSV output file will be stored
#    main_directory = os.path.join('/Volumes/New P/DataScience/CFARS','NRGSeptLTERRA')
#    #Directory where lidar data are saved
#    lidar_directory = os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/lidar_directory')
#    #Directory where data from reference device is saved
#    reference_directory = os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/reference_directory', 'ProcessedReferenceData')
#    #Height were TI values will be compared
#    height_needed = 55
#    #Time period for data extraction
#    years = [2019,2020]
#    years = [2019]
#    months = [9,10,11,12,1,2,3,4,5,6,7,8]
#    months = [9]
#    days = np.arange(1,31)
#    days = np.arange(21,28)
#    #Wind speed minimum and maximum thresholds. If threshold is not needed, put 'none'.
#    U_min = 'none'
#    U_max = 'none'
#    #Measurement to use for wind speed threshold ('reference' or 'lidar')
#    U_opt = 'reference'
#    #Wind direction sectors to exclude. If exclusion sector is not needed, put 'none' for wd_min and wd_max values.
#    wd_sector_min1 = 'none'
#    wd_sector_max1 = 'none'
#    wd_sector_min2 = 'none'
#    wd_sector_max2 = 'none'
#    #Measurement to use for wind direction sector ('reference' or 'lidar')
#    wd_opt = 'reference'
#    ##########################################################
#    if 'none' in str(U_min):
#        U_min = 0
#    if 'none' in str(U_max):
#        U_max = np.inf
#    if 'none' in str(wd_sector_min1):
#        wd_sector_min1 = np.nan
#    if 'none' in str(wd_sector_max1):
#        wd_sector_max1 = np.nan
#    if 'none' in str(wd_sector_min2):
#        wd_sector_min2 = np.nan
#    if 'none' in str(wd_sector_max2):
#        wd_sector_max2 = np.nan

#    files_ref = []
#    files_WC_DBS = []
#    files_WC_VAD = []
#    files_WC_vr = []

#    for i in years:
#        print (i)
#        for j in months:
#            print (j)
#            for k in days:
#                print (str('day ' + str(k)))
#                #Find all reference and lidar files that correspond to this date
#                files_found_ref = glob.glob(reference_directory +'/*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
#                print (files_found_ref)
#                if len(files_found_ref) > 0:
#                    files_found_WC = glob.glob(lidar_directory + '/*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
#                else:
#                    pass
#                #Find 10-min. periods where data files for both the lidar and reference device exist
#                for ii in files_found_ref:
#                    str_match = str(ii[-21:])
#                    matching_WC = [s for s in files_found_WC if str_match in s]
#                    if(matching_WC):
#                        files_ref.append(ii)
#                        for jj in matching_WC:
#                            if "DBS" in jj:
#                                files_WC_DBS.append(jj)
#                            if "VAD" in jj:
#                                files_WC_VAD.append(jj)
#                           if "vr" in jj:
#                                files_WC_vr.append(jj)

#    TI_ref_all = np.zeros(len(files_ref))
#    TI_ref_all[:] = np.nan

#    TI_WC_orig_all = np.zeros(len(files_ref))
#    TI_WC_orig_all[:] = np.nan

    #Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
    #For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
    #not be used in a power performance test.

#    for i in range(len(files_ref)):
#        print (i)
#        if 'reference' in U_opt:
#            file_ws = files_ref[i]
#       else:
#            file_ws = files_WC_DBS[i]
#        if 'reference' in wd_opt:
#            file_wd = files_ref[i]
#        else:
#            file_wd = files_WC_DBS[i]
#        U = np.load(file_ws)['U']
#        wd = np.load(file_wd)['wd']
#        if ~(wd >= wd_sector_min1 and wd < wd_sector_max1) and ~(wd >= wd_sector_min2 and wd < wd_sector_max2) and U >=U_min and U < U_max:
#            TI_ref_all[i] = (np.sqrt(np.load(files_ref[i])['u_var'])/np.load(files_ref[i])['U'])*100
#            TI_WC_orig_all[i] = (np.sqrt(np.load(files_WC_DBS[i])['u_var'])/np.load(files_WC_DBS[i])['U'])*100
#
#    with open(main_directory + 'L_TERRA_combination_summary_WC.csv', 'w') as fp:
#        a = csv.writer(fp, delimiter=',')
#        data = [['Module Options','MAE Overall','MAE Stable','MAE Neutral','MAE Unstable']]
#        a.writerows(data)

#    #Options to test in L-TERRA
#    ws_opts = ["raw_WC","VAD"]
#    noise_opts = ["none","spike","lenschow_linear","lenschow_subrange","lenschow_spectrum"]
#    vol_opts = ["none","spectral_correction_fit","acf"]
#    contamination_opts = ["none","taylor_ws","taylor_var"]

#    for ii in ws_opts:
#        print (str('ws opts' + ii))
#        for jj in noise_opts:
#            print (str('noise opts' + jj))
#            for kk in vol_opts:
#                print (str('vol opts' + kk))
#                for mm in contamination_opts:
#                    print (str('contam opts' + mm))
#                    #Initialize arrays for the new lidar TI after each correction has been applied
#                    TI_WC_noise_all = np.zeros(len(files_WC_DBS))
#                    TI_WC_noise_all[:] = np.nan
#                    TI_WC_vol_avg_all = np.zeros(len(files_WC_DBS))
#                    TI_WC_vol_avg_all[:] = np.nan
#                    TI_WC_var_contamination_all = np.zeros(len(files_WC_DBS))
#                    TI_WC_var_contamination_all[:] = np.nan

#                    p_all = np.zeros(len(files_WC_DBS))
#                    p_all[:] = np.nan

#                    mode_ws,mode_noise,mode_vol,mode_contamination = [ii,jj,kk,mm]

#                    for i in range(len(files_WC_DBS)):
#                        print (i)
#                        if ~np.isnan(TI_ref_all[i]):
#                            if "raw" in mode_ws:
#                                frequency = 1.
#                                file_temp = files_WC_DBS[i]
#                            else:
#                                frequency = 1./4
#                                file_temp = files_WC_VAD[i]

#                            u_rot = np.load(file_temp)['u_rot']
#                            u_var = np.load(file_temp)['u_var']
#                            U = np.load(file_temp)['U']

#                            p_all[i] = np.load(files_WC_DBS[i])['p']

#                            vr_n = np.load(files_WC_vr[i])['vr_n']
#                            vr_e = np.load(files_WC_vr[i])['vr_e']
#                            vr_s = np.load(files_WC_vr[i])['vr_s']
#                            vr_w = np.load(files_WC_vr[i])['vr_w']
#                            vert_beam = np.load(files_WC_VAD[i])['w_rot']

#                            wd = np.load(files_WC_DBS[i])['wd']

#                            #Apply noise correction and calculate variance
#                            if "none" in mode_noise:
#                                u_var_noise = u_var
#                            else:
#                                u_var_noise = lidar_processing_noise(u_rot,frequency,mode_ws,mode_noise)#
#
#                            TI_WC_noise_all[i] = (np.sqrt(u_var_noise)/U)*100
#
#                            #Estimate loss of variance due to volume averaging
#                            if "none" in mode_vol:
#                                u_var_diff = 0.
#                            else:
#                                try:
#                                    u_var_diff = lidar_processing_vol_averaging(u_rot,frequency,mode_ws,mode_vol)
#                                except:
#                                    u_var_diff = 0.
#                            u_var_vol = u_var_noise + u_var_diff
#                            TI_WC_vol_avg_all[i] = (np.sqrt(u_var_vol)/U)*100
#                            #Estimate increase in variance due to variance contamination
#                            if "none" in mode_contamination:
#                                u_var_diff = 0.
#                            else:
#                                u_var_diff = lidar_processing_var_contam(vr_n,vr_e,vr_s,vr_w,vert_beam,wd,U,height_needed,1./4,62.,mode_contamination)
#                                try:
#                                    if np.isnan(u_var_diff):
#                                        u_var_diff = 0.
#                                except:
#                                    u_var_diff = 0.
#                            u_var_contam = u_var_vol - u_var_diff
#                            TI_WC_var_contamination_all[i] = (np.sqrt(u_var_contam)/U)*100

                    #Corrected TI is the value of the TI after it has been through all correction modules
#                    TI_WC_corrected_all = TI_WC_var_contamination_all
#
#                    opts = 'WS_' + mode_ws +'_N_' + mode_noise + '_V_' + mode_vol + '_C_' + mode_contamination

#                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all)]
#                    mask = functools.reduce(np.logical_and, mask)
#                    MAE_all = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])

#                    #Classify stability by shear parameter, p. A different stability metric could be used if available.
#                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all),p_all >= 0.2]
#                    mask = functools.reduce(np.logical_and, mask)
#                    MAE_s = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])

#                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all),p_all >= 0.1,p_all < 0.2]
#                    mask = functools.reduce(np.logical_and, mask)
#                    MAE_n = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])

#                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all),p_all < 0.1]
#                    mask = functools.reduce(np.logical_and, mask)
#                    MAE_u = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])

#                    if MAE_all < MAE_min:
#                        MAE_min = MAE_all
#                        opts_best = opts
#                    if MAE_s < MAE_min_s:
#                        MAE_min_s = MAE_s
#                        opts_best_s = opts
#                    if MAE_n < MAE_min_n:
#                        MAE_min_n = MAE_n
#                        opts_best_n = opts
#                    if MAE_u < MAE_min_u:
#                        MAE_min_u = MAE_u
#                        opts_best_u = opts

                    #Write out final MAE values after all corrections have been applied for this model combination
#                    opts_temp = 'WS_' + mode_ws +'_N_' + mode_noise + '_V_' + mode_vol + '_C_' + mode_contamination
#                    with open(main_directory + '/'+'L_TERRA_combination_summary_WC.csv', 'a') as fp:
#                        a = csv.writer(fp, delimiter=',')
#                        data = [[opts_temp,'{:0.2f}'.format(MAE_all),'{:0.2f}'.format(MAE_s),\
#                        '{:0.2f}'.format(MAE_n),'{:0.2f}'.format(MAE_u)]]
#                        a.writerows(data)

    #Write out minimum MAE values for each stability class and model options associated with these minima
#    with open(main_directory + '/'+ 'L_TERRA_combination_summary_WC.csv', 'a') as fp:
#         a = csv.writer(fp, delimiter=',')
#         data = [['Overall Min. MAE','{:0.2f}'.format(MAE_min)]]
#         a.writerows(data)
#         data = [['Best Options',opts_best]]
#         a.writerows(data)
#         data = [['Overall Min. MAE Stable','{:0.2f}'.format(MAE_min_s)]]
#         a.writerows(data)
#         data = [['Best Options Stable',opts_best_s]]
#         a.writerows(data)
#         data = [['Overall Min. MAE Neutral','{:0.2f}'.format(MAE_min_n)]]
#         a.writerows(data)
#         data = [['Best Options Neutral',opts_best_n]]
#         a.writerows(data)
#         data = [['Overall Min. MAE Unstable','{:0.2f}'.format(MAE_min_u)]]
#         a.writerows(data)
#         data = [['Best Options Unstable',opts_best_u]]
#         a.writerows(data)

    # apply corrections
    from itertools import compress
    ##########################################################
    #USER INPUTS GO HERE
    #Directory where CSV output file will be stored
    main_directory = os.path.join('/Volumes/New P/DataScience/CFARS','NRGSeptLTERRA')
    #Directory where lidar data are saved
    lidar_directory = os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/lidar_directory', 'ProcessedRTDData')

    #Height were TI values will be compared
    height_needed = 55
    #Time period for data extraction
    years = [2019,2020]
    years = [2019]
    months = [9,10,11,12,1,2,3,4,5,6,7,8,9]
    months = [9]
    days = np.arange(1,31)
    days = np.arange(25,27)
    #Wind speed minimum and maximum thresholds. If threshold is not needed, put 'none'.
    U_min = 'none'
    U_max = 'none'
    #Wind direction sectors to exclude. If exclusion sector is not needed, put 'none' for wd_min and wd_max values.
    wd_sector_min1 = 'none'
    wd_sector_max1 = 'none'
    wd_sector_min2 = 'none'
    wd_sector_max2 = 'none'
    #Model options to use for different stability conditions. If correction is not needed, put 'none'.
    #Options for stable conditions (p >= 0.2)
    mode_ws_s = 'raw_WC'
    mode_noise_s = 'spike'
    mode_vol_s = 'acf'
    mode_contamination_s = 'taylor_ws'
    #Options for neutral conditions (0.1 <= p < 0.2)
    mode_ws_n = 'raw_WC'
    mode_noise_n = 'spike'
    mode_vol_n = 'acf'
    mode_contamination_n = 'taylor_ws'
    #Options for unstable conditions (p < 0.1)
    mode_ws_u = 'raw_WC'
    mode_noise_u = 'none'
    mode_vol_u = 'none'
    mode_contamination_u = 'none'
    ##########################################################
    if 'none' in str(U_min):
        U_min = 0
    if 'none' in str(U_max):
        U_max = np.inf
    if 'none' in str(wd_sector_min1):
        wd_sector_min1 = np.nan
    if 'none' in str(wd_sector_max1):
        wd_sector_max1 = np.nan
    if 'none' in str(wd_sector_min2):
        wd_sector_min2 = np.nan
    if 'none' in str(wd_sector_max2):
        wd_sector_max2 = np.nan

    files_WC_DBS = []
    files_WC_VAD = []
    files_WC_vr = []

    for i in years:
        print (i)
        for j in months:
            print (j)
            for k in days:
                print (k)
                #Find all lidar files that correspond to this date
                files_found_WC = glob.glob(lidar_directory + '*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
                for jj in files_found_WC:
                    if "DBS" in jj:
                        files_WC_DBS.append(jj)
                    if "VAD" in jj:
                        files_WC_VAD.append(jj)
                    if "vr" in jj:
                        files_WC_vr.append(jj)

    TI_WC_orig_all = np.zeros(len(files_WC_DBS))
    TI_WC_orig_all[:] = np.nan
    time_all = []

    #Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
    #For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
    #not be used in a power performance test. In this case, all measurements come from the lidar data.
    for i in range(len(files_WC_DBS)):
        print (i)
        wd = np.load(files_WC_DBS[i],allow_pickle=True)['wd']
        U = np.load(files_WC_DBS[i],allow_pickle=True)['U']
        time_all.append(np.load(files_WC_DBS[i],allow_pickle=True)['time'].item())
        if ~(wd.any() >= wd_sector_min1 and wd.any() < wd_sector_max1) and ~(wd.any() >= wd_sector_min2 and wd.any() < wd_sector_max2) and U.any() >=U_min and U.any() < U_max:
            TI_WC_orig_all[i] = (np.sqrt(np.load(files_WC_DBS[i], allow_pickle=True)['u_var'])/np.load(files_WC_DBS[i],allow_pickle=True)['U'])*100

    TI_WC_orig_all = np.array(TI_WC_orig_all)

    #Convert time from datetime format to a normal timestamp.
    #Timestamp is in UTC and corresponds to start of 10-min. averaging period.
    timestamp_10min_all = []

    for i in time_all:
        timestamp_10min_all.append(datetime.datetime.strftime(i,"%Y/%m/%d %H:%M"))

    with open(main_directory + 'L_TERRA_corrected_TI_WC.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        data = [['Timestamp (UTC)','Original TI (%)','Corrected TI (%)']]
        a.writerows(data)

    #Initialize arrays for the new lidar TI after each correction has been applied
    TI_WC_noise_all = np.zeros(len(files_WC_DBS))
    TI_WC_noise_all[:] = np.nan
    TI_WC_vol_avg_all = np.zeros(len(files_WC_DBS))
    TI_WC_vol_avg_all[:] = np.nan
    TI_WC_var_contamination_all = np.zeros(len(files_WC_DBS))
    TI_WC_var_contamination_all[:] = np.nan

    p_all = np.zeros(len(files_WC_DBS))
    p_all[:] = np.nan

    for i in range(len(files_WC_DBS)):
        print (i)
        if ~np.isnan(TI_WC_orig_all[i]):
            p_all[i] = np.load(files_WC_DBS[i],allow_pickle=True)['p']
            if p_all[i] >= 0.2:
                mode_ws = mode_ws_s
                mode_noise = mode_noise_s
                mode_vol = mode_vol_s
                mode_contamination = mode_contamination_s
            elif p_all[i] >= 0.1:
                mode_ws = mode_ws_n
                mode_noise = mode_noise_n
                mode_vol = mode_vol_n
                mode_contamination = mode_contamination_n
            else:
                mode_ws = mode_ws_u
                mode_noise = mode_noise_u
                mode_vol = mode_vol_u
                mode_contamination = mode_contamination_u
            if "raw" in mode_ws:
                frequency = 1.
                file_temp = files_WC_DBS[i]
            else:
                frequency = 1./4
                file_temp = files_WC_VAD[i]

            u_rot = np.load(file_temp,allow_pickle=True)['u_rot']
            u_var = np.load(file_temp,allow_pickle=True)['u_var']
            U = np.load(file_temp)['U']

            vr_n = np.load(files_WC_vr[i],allow_pickle=True)['vr_n']
            vr_e = np.load(files_WC_vr[i],allow_pickle=True)['vr_e']
            vr_s = np.load(files_WC_vr[i],allow_pickle=True)['vr_s']
            vr_w = np.load(files_WC_vr[i],allow_pickle=True)['vr_w']
            vert_beam = np.load(files_WC_VAD[i],allow_pickle=True)['w_rot']

            wd = np.load(files_WC_DBS[i],allow_pickle=True)['wd']

            #Apply noise correction and calculate variance
            if "none" in mode_noise:
                u_var_noise = u_var
            else:
                u_var_noise = lidar_processing_noise(u_rot,frequency,mode_ws,mode_noise)

            TI_WC_noise_all[i] = (np.sqrt(u_var_noise)/U)*100

            #Estimate loss of variance due to volume averaging
            if "none" in mode_vol:
                u_var_diff = 0.
            else:
                try:
                    u_var_diff = lidar_processing_vol_averaging(u_rot,frequency,mode_ws,mode_vol)
                except:
                    u_var_diff = 0.

            u_var_vol = u_var_noise + u_var_diff
            TI_WC_vol_avg_all[i] = (np.sqrt(u_var_vol)/U)*100

            #Estimate increase in variance due to variance contamination
            if "none" in mode_contamination:
                u_var_diff = 0.
            else:
                u_var_diff = lidar_processing_var_contam(vr_n,vr_e,vr_s,vr_w,vert_beam,wd,U,height_needed,1./4,62.,mode_contamination)

                try:
                    if np.isnan(u_var_diff).any():
                        u_var_diff = 0.
                except:
                    u_var_diff = 0.

            u_var_contam = u_var_vol - u_var_diff
            TI_WC_var_contamination_all[i] = (np.sqrt(u_var_contam)/U)*100

    #Extract TI values and timestamps for all times when corrected TI value is valid
    mask = ~np.isnan(TI_WC_var_contamination_all)

    timestamp_10min_all = list(compress(timestamp_10min_all,mask))
    TI_WC_orig_all = TI_WC_orig_all[mask]
    TI_WC_var_contamination_all = TI_WC_var_contamination_all[mask]

    #Reduce number of decimal places in output TI data
    TI_orig_temp = ["%0.2f" % i for i in TI_WC_orig_all]
    TI_corrected_temp = ["%0.2f" % i for i in TI_WC_var_contamination_all]

    #Write out timestamp, original lidar TI, and corrected lidar TI
    with open(main_directory + 'L_TERRA_corrected_TI_WC.csv', 'a') as fp:
         a = csv.writer(fp, delimiter=',')
         data = np.vstack([timestamp_10min_all,TI_orig_temp,TI_corrected_temp]).transpose()
         a.writerows(data)

    print ('finished')
    sys.exit()
    # Merge ^ LTERRA TI result with timestamps, inputdata on timestamp
    lterraCorrected = pd.read_csv(os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA','L_Terra_corrected_TI_WC.csv'))
    fullData = inputdata
    fullData['Timestamp'] = Timestamp
    fullData = pd.merge(fullData,lterraCorrected, on='Timestamp')
    fullData = fullData.drop(columns=['Timestamp'])

    # save orig TI to inject into input data
    fullData['orig_RSD_TI'] = inputdata['RSD_TI']
    fullData.to_csv(os.path.join('/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA','DataWithOrigTI.csv'))
    # make new TI RSD_TI, make old RSD_TI the Orig_RSD_TI
    #inputdata['RSD_TI'] =

    # apply ML model
    print ('Applying Correction Method: SS-LTERRA_1HZ')
    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(inputdata.copy())
    lm_corr['sensor'] = sensor
    lm_corr['height'] = height
    lm_corr['correction'] = 'SS_LTERRA_1HZ'
    correctionName = 'SS_LTERRA_1HZ'
    baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr, Timestamps, method)

    if RSDtype['Selection'][0:4] == 'Wind':
        print ('Applying Correction Method: SS-LTERRA_1HZ by stability class (TKE)')
        # stability subset output for primary height (all classes)
        ResultsLists_class = initialize_resultsLists('class_')
        className = 1
        for item in All_class_data:
            inputdata_corr, lm_corr, m, c= perform_SS_LTERRA_S_ML_correction(item[primary_idx].copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = str('SS_LTERRA_1HZ' + '_' + 'class_' + str(className))
            correctionName = str('SS_LTERRA_1HZ' + '_TKE_' + str(className))
            ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                       inputdata_corr, Timestamps, method)
            className += 1
        ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
    if RSD_alphaFlag:
        print ('Applying Correction Method: SS-LTERRA_1HZ by stability class Alpha w/ RSD')
        ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
        className = 1
        for item in All_class_data_alpha_RSD:
            inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = str('SS-LTERRA_1HZ' + '_' + 'class_' + str(className))
            correctionName = str('SS_LTERRA_1HZ' + '_alphaRSD_' + str(className))
            ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                 inputdata_corr, Timestamps, method)
            className += 1
        ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
    if cup_alphaFlag:
        print ('Applying correction Method: SS-LTERRA_1HZ by stability class Alpha w/cup')
        ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
        className = 1
        for item in All_class_data_alpha_Ane:
            inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = str('SS_LTERRA_1HZ' + '_' + 'class_' + str(className))
            emptyclassFlag = False
            correctionName = str('SS_LTERRA_1HZ' + '_alphaCup_' + str(className))
            ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                 inputdata_corr, Timestamps, method)
            className += 1
        ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

def var_correction(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode):
    #Uses Taylor's frozen turbulence hypothesis with data from the vertically
    #pointing beam to estimate new values of the u and v variance.

    #Inputs
    #vr_n, vr_e, vr_s, vr_w, vr_z: Time series of radial velocity from north-, east-, south, west-, and
    #vertically pointing beams, respectively, at height of interest.
    #wd: 10-min. Mean wind direction
    #U: 10-min. Mean horizontal wind speed
    #height_needed: Measurement height corresponding to velocity data
    #frequency_vert_beam: Sampling frequency of data from vertically pointing beam
    #el_angle: Elevation angle of off-vertical beam positions (in degrees, measured from the ground)
    #mode: Type of variance contamination correction to be applied. Options are taylor_ws and taylor_var.

    #Outputs
    #var_diff: Estimate of increase in streamwise variance due to variance contamination

    import numpy as np

    w_N = np.zeros(len(vr_z))
    w_N[:] = np.nan
    w_E = np.zeros(len(vr_z))
    w_E[:] = np.nan
    w_S = np.zeros(len(vr_z))
    w_S[:] = np.nan
    w_W = np.zeros(len(vr_z))
    w_W[:] = np.nan

    u_bar = np.sin(np.radians(wd - 180))*U
    v_bar = np.cos(np.radians(wd - 180))*U

    delta_t_vert_beam = 1./frequency_vert_beam

    #Calculate the number of time steps needed for eddies to travel from one
    #side of the scanning circle to the other
    dist = height_needed/np.tan(np.radians(el_angle))
    delta_t_u = dist/u_bar
    interval_u = round(delta_t_u/delta_t_vert_beam)
    delta_t_v = dist/v_bar
    interval_v = round(delta_t_v/delta_t_vert_beam)

    #Estimate values of w at different sides of the scanning circle by using
    #Taylor's frozen turbulence hypothesis
    for i in range(len(vr_z)):
        try:
            w_N[i] = vr_z[i-interval_v]
            w_E[i] = vr_z[i-interval_u]
        except:
            w_N[i] = np.nan
            w_E[i] = np.nan
        try:
            w_S[i] = vr_z[i+interval_v]
            w_W[i] = vr_z[i+interval_u]
        except:
            w_S[i] = np.nan
            w_W[i] = np.nan

    if "taylor_ws" in mode:
        #Use the new values of w to estimate the u and v components using the DBS technique
        #and calculate the variance
        u_DBS_new = ((vr_e-vr_w) - (w_E-w_W)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))
        v_DBS_new = ((vr_n-vr_s) - (w_N-w_S)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))

        u_var_lidar_new = get_10min_var(u_DBS_new,frequency_vert_beam)
        v_var_lidar_new = get_10min_var(v_DBS_new,frequency_vert_beam)
    else:
        #Calculate change in w across the scanning circle in north-south and east-west directions
        dw_est1 = w_S - w_N
        dw_est2 = w_W - w_E

        vr1_var = get_10min_var(vr_n,1./4)
        vr2_var = get_10min_var(vr_e,1./4)
        vr3_var = get_10min_var(vr_s,1./4)
        vr4_var = get_10min_var(vr_w,1./4)
        dw_var1 = get_10min_var(dw_est1,1./4)
        dw_var2 = get_10min_var(dw_est2,1./4)

        vr1_vr3_var = get_10min_covar(vr_n,vr_s,1./4)
        vr2_vr4_var = get_10min_covar(vr_e,vr_w,1./4)

        vr1_dw_var = get_10min_covar(vr_n,dw_est1,1./4)
        vr3_dw_var = get_10min_covar(vr_s,dw_est1,1./4)
        vr2_dw_var = get_10min_covar(vr_e,dw_est2,1./4)
        vr4_dw_var = get_10min_covar(vr_w,dw_est2,1./4)

        #These equations are adapted from Newman et al. (2016), neglecting terms involving
        #du or dv, as these terms are expected to be small compared to dw
        #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
        #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
        #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
        u_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr2_var + vr4_var- 2*vr2_vr4_var + 2*vr2_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr4_dw_var*np.sin(np.radians(el_angle)) + dw_var2*np.sin(np.radians(el_angle))**2)

        v_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr1_var + vr3_var- 2*vr1_vr3_var + 2*vr1_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr3_dw_var*np.sin(np.radians(el_angle)) + dw_var1*np.sin(np.radians(el_angle))**2)

    #Rotate the variance into the mean wind direction
    #Note: The rotation should include a term with the uv covariance, but the
    #covariance terms are also affected by variance contamination. In Newman
    #et al. (2016), it was found that the uv covariance is usually close to 0 and
    #can safely be neglected.
    #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
    #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
    #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
    u_rot_var_new = u_var_lidar_new*(np.sin(np.radians(wd)))**2 + v_var_lidar_new*(np.cos(np.radians(wd)))**2

    #Calculate the wind speed and variance if w is assumed to be the same on all
    #sides of the scanning circle
    u_DBS = (vr_e-vr_w)/(2*np.cos(np.radians(el_angle)))
    v_DBS = (vr_n-vr_s)/(2*np.cos(np.radians(el_angle)))

    u_var_DBS = get_10min_var(u_DBS,frequency_vert_beam)
    v_var_DBS = get_10min_var(v_DBS,frequency_vert_beam)

    u_rot_var = u_var_DBS*(np.sin(np.radians(wd)))**2 + v_var_DBS*(np.cos(np.radians(wd)))**2

    return u_rot_var-u_rot_var_new

def acvf(ts):
    #Calculate autocovariance function for a time series

    #Inputs
    #ts: Time series of data

    #Outputs
    #ts_corr: Values of autovariance function starting from lag 0

    import numpy as np
    lags = range(0,len(ts))
    ts_corr = []
    for i in lags:
        ts_subset_temp = ts[i:len(ts)]
        ts_subset_temp2 = ts[0:len(ts)-i]
        ts_corr.append(np.nanmean((ts_subset_temp-np.nanmean(ts_subset_temp))*(ts_subset_temp2-np.nanmean(ts_subset_temp2))))
    return ts_corr

def inertial_subrange_func(t, b, C):
    #Inertial subrange fit for autocovariance function
    #t is lag time, b is the variance at lag 0, and C is a parameter corresponding to eddy dissipation

    return -C*t**(2./3) + b

def lenschow_technique(ts,frequency,mode_ws,option):
    #Apply different forms of the Lenschow et al. (2000) technique
    #Reference: Lenschow, D. H., V. Wulfmeyer, and C. Senff, 2000: Measuring second-through fourth-order moments in noisy data. J. Atmos. Oceanic Technol., 17, 1330–1347.

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #mode_noise: Type of Lenschow noise correction to be applied. Options are linear, subrange, and spectrum.

    #Outputs
    #new_ts_var: 10-min. variance after noise correction has been applied

    import numpy as np
    from scipy.optimize import curve_fit

    ts = fill_nan(ts)

    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)
    var_diff = []
    var_orig = []
    lags = np.arange(0,ten_min_count)/float(frequency)
    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        #10-min. window of data
        ts_window = ts[i:i+ten_min_count]
        ten_min_index = (i-1)/ten_min_count + 1
        var_orig.append(get_10min_var(ts_window,frequency))
        if 'linear' in option:
            #Use values of ACVF from first four non-zero lags to linearly extrpolate
            #ACVF to lag 0
            ts_corr = acvf(ts_window)
            x_vals = lags[1:4];
            y_vals = ts_corr[1:4]
            p = np.polyfit(x_vals,y_vals,1)
            var_diff.append(var_orig[ten_min_index]-p[1])
        if 'subrange' in option:
            #Use values of ACVF from first four non-zero lags to produce fit
            #to inertial subrange function. Value of function at lag 0 is assumed
            #to be the true variance.
            ts_corr = acvf(ts_window)
            x_vals = lags[1:4];
            y_vals = ts_corr[1:4]
            try:
                popt, pcov = curve_fit(inertial_subrange_func, x_vals, y_vals,\
                p0 = [np.mean((ts_window-np.mean(ts_window))**2),0.002])
                var_diff.append(var_orig[ten_min_index]-popt[0])
            except:
                 var_diff.append(np.nan)
        if 'spectrum' in option:
           #Assume spectral power at high frequencies is due only to noise. Average
           #the spectral power at the highest 20% of frequencies in the time series
           #and integrate the average power across all frequencies to estimate the
           #noise floor
           import numpy.ma as ma
           if "raw_WC" in mode_ws:
               [S,fr] = get_10min_spectrum_WC_raw(ts_window,frequency)
           else:
               [S,fr] = get_10min_spectrum(ts_window,frequency)
           x = ma.masked_inside(fr,0.8*fr[-1],fr[-1])
           func_temp = []
           for j in range(len(fr)):
               func_temp.append(np.mean(S[x.mask]))
           noise_floor = np.trapz(func_temp,fr)
           var_diff.append(noise_floor)

    var_diff = np.array(var_diff)
    #Only use var_diff values where the noise variance is positive
    var_diff[var_diff < 0] = 0
    new_ts_var = np.array(var_orig)-var_diff
    return new_ts_var

def get_10min_spectrum(ts,frequency):
    #Calculate power spectrum for 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #S_A_fast: Spectral power
    #frequency_fft: Frequencies correspond to spectral power values

    import numpy as np
    N = len(ts)
    delta_f = float(frequency)/N
    frequency_fft = np.linspace(0,float(frequency)/2,float(N/2))
    F_A_fast = np.fft.fft(ts)/N
    E_A_fast = 2*abs(F_A_fast[0:N/2]**2)
    S_A_fast = (E_A_fast)/delta_f
    return S_A_fast,frequency_fft

def get_10min_spectrum_WC_raw(ts,frequency):
    #Calculate power spectrum for 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #S_A_fast: Spectral power
    #frequency_fft: Frequencies correspond to spectral power values

    import numpy as np
    N = len(ts)
    delta_f = float(frequency)/N
    frequency_fft = np.linspace(0,float(frequency)/2,float(N/2))
    F_A_fast = np.fft.fft(ts)/N
    E_A_fast = 2*abs(F_A_fast[0:N/2]**2)
    S_A_fast = (E_A_fast)/delta_f
    #Data are only used for frequencies lower than 0.125 Hz. Above 0.125 Hz, the
    #WINDCUBE spectrum calculated using raw data begins to show an artifact. This
    #artifact is due to the recording of the u, v, and w components for every beam
    #position, which results in repeating components.
    S_A_fast = S_A_fast[frequency_fft <= 0.125]
    frequency_fft = frequency_fft[frequency_fft <= 0.125]
    return S_A_fast,frequency_fft

def get_10min_covar(ts1,ts2,frequency):
    #Calculate the covariance of two variables

    #Inputs
    #ts1: Time series of variable 1
    #ts2: Time series of variable 2
    #frequency: Sampling frequency

    #Outputs
    #ts_covar: 10-min. covariance of variables 1 and 2

    import numpy as np
    import functools
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    ts_covar = []

    for i in np.arange(0,len(ts1)-ten_min_count+1,ten_min_count):
        ts_temp1 = ts1[i:i+ten_min_count]
        ts_temp2 = ts2[i:i+ten_min_count]
        mask = [~np.isnan(ts_temp1),~np.isnan(ts_temp2)]
        total_mask = functools.reduce(np.logical_and, mask)
        ts_temp1 = ts_temp1[total_mask]
        ts_temp2 = ts_temp2[total_mask]
        ts_covar.append(np.nanmean((ts_temp1-np.nanmean(ts_temp1))*(ts_temp2-np.nanmean(ts_temp2))))

    return np.array(ts_covar)

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    #Adapted from code posted on Stack Overflow: http://stackoverflow.com/a/9815522
    #1-D linear interpolation to fill missing values

    #Inputs
    #A: Time series where NaNs need to be filled

    #Outputs
    #B: Time series with NaNs filled

    from scipy import interpolate
    import numpy as np
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    #Only perform interpolation if more than 75% of the data are valid
    if float(len(np.array(good).ravel()))/len(A) > 0.75:
        f = interpolate.interp1d(inds[good], A[good],bounds_error=False,fill_value='extrapolate')
        B = np.where(np.isfinite(A),A,f(inds))
    else:
        B = A
    return B

def spike_filter(ts,frequency):
    #Spike filter based on procedure used in Wang et al. (2015)
    #Reference: Wang, H., R. J. Barthelmie,  A. Clifton, and S. C. Pryor, 2015:
    #Wind measurements from arc scans with Doppler wind lidar, J. Atmos.
    #Ocean. Tech., 32, 2024–2040.


    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #ts_filtered_interp: Filtered time series with NaNs filled in

    import numpy as np


    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    ts_filtered = np.copy(ts)
    ts_filtered_interp = np.copy(ts)

    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        ts_window = ts_filtered[i:i+ten_min_count]

        #Calculate delta_v, difference between adjacent velocity values
        delta_v = np.zeros(len(ts_window)-1)
        for j in range(len(ts_window)-1):
            delta_v[j] = ts_window[j+1] - ts_window[j]
        q75, q25 = np.percentile(delta_v, [75 ,25])
        IQR= q75 - q25
        #If abs(delta_v) at times i and i-1 are larger than twice the interquartile
        #range (IQR) and the delta_v values at times i and i-1 are of opposite sign,
        #the velocity at time i is considered a spike and set to NaN.
        for j in range(1,len(ts_window)-1):
            if abs(delta_v[j]) > 2*IQR and abs(delta_v[j-1]) > 2*IQR:
                if np.sign(delta_v[j]) != np.sign(delta_v[j-1]):
                    ts_window[j] = np.nan
                    ts_filtered[i+j] = np.nan
        #Set entire 10-min. period to NaN if more than 40% of the velocity points
        #are already NaN.
        if (float(len(ts_window[np.isnan(ts_window)]))/len(ts_window)) > 0.4:
            ts_filtered[i:i+ten_min_count] = np.nan
        #Use a 1-D linear interpolation to fill in missing values
        ts_filtered_interp[i:i+ten_min_count] = fill_nan(ts_filtered[i:i+ten_min_count])
    return ts_filtered_interp


def lidar_processing_noise(ts,frequency,mode_ws,mode_noise):
    #Function to apply noise correction to time series. Outputs new variance after
    #noise correction has been applied.

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #mode_noise: Type of noise correction to be applied. Options are spike, lenschow_linear, lenschow_subrange, and lenschow_spectrum.

    #Outputs
    #new_ts_var: New 10-min. variance values after noise correction has been applied

    if "spike" in mode_noise:
        ts_filtered = spike_filter(ts,frequency)
        new_ts_var = get_10min_var(ts_filtered,frequency)

    if "lenschow_linear" in mode_noise:

        new_ts_var = lenschow_technique(ts,frequency,mode_ws,'linear')

    if "lenschow_subrange" in mode_noise:

        new_ts_var = lenschow_technique(ts,frequency,mode_ws,'subrange')

    if "lenschow_spectrum" in mode_noise:

        new_ts_var = lenschow_technique(ts,frequency,mode_ws,'spectrum')

    return new_ts_var

def Kaimal_spectrum_func(X, L):
    #Given values of frequency (fr), mean horizontal wind speed (U), streamwise
    #variance (u_var), and length scale (L), calculate idealized Kaimal spectrum
    #This uses the form given by Eq. 2.24 in Burton et al. (2001)
    #Reference: Burton, T., D. Sharpe, N. Jenkins, N., and E. Bossanyi, 2001:
    #Wind Energy Handbook, John Wiley & Sons, Ltd., 742 pp.
    fr,U,u_var = X
    return u_var*fr*((4*(L/U)/((1+6*(fr*L/U))**(5./3))))


def Kaimal_spectrum_func2(pars, x, data=None):
    #Kaimal spectrum function for fitting. Trying to minimize the difference
    #between the actual spectrum (data) and the modeled spectrum (model)
    vals = pars.valuesdict()
    L =  vals['L']
    U =  vals['U']
    u_var = vals['u_var']
    model = u_var*x*((4*(L/U)/((1+6*(x*L/U))**(5./3))))
    if data is None:
        return model
    return model-data

def spectral_correction(u_rot,frequency,mode_ws,option):
    #Estimates loss of variance due to volume averaging by extrapolating spectrum
    #out to higher frequencies and integrating spectrum over higher frequencies


    #Inputs
    #u_rot: Time series of streamwise wind speed
    #frequency: Sampling frequency of time series
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #option: Type of volume averaging correction to be applied. Options are spectral_correction_fit and acf.

    #Outputs
    #var_diff: Estimate of loss of streamwise variance due to volume averaging

    import numpy as np
    import scipy.signal
    from lmfit import minimize,Parameters
    ten_min_count = frequency*60*10

    var_diff = []

    for i in np.arange(0,len(u_rot)-ten_min_count+1,ten_min_count):
        u_temp = u_rot[i:i+ten_min_count]
        U = np.mean(u_temp)
        u_var = get_10min_var(u_temp,frequency)
        #Detrend time series before estimating parameters for modeled spectrum
        u_temp = scipy.signal.detrend(u_temp)
        if "raw_WC" in mode_ws:
            [S,fr] = get_10min_spectrum_WC_raw(u_temp,frequency)
        else:
            [S,fr] = get_10min_spectrum(u_temp,frequency)
        if "spectral_correction_fit" in option:
            #Find value of length scale that produces best fit to idealized
            #Kaimal spectrum
            fit_params = Parameters()
            fit_params.add('L', value=500,min=0,max=1500)
            fit_params.add('U', value=U,vary=False)
            fit_params.add('u_var', value=u_var,vary=False)
            out = minimize(Kaimal_spectrum_func2, fit_params, args=(fr,), kws={'data':fr*S})
            L = out.params['L'].value

        else:
            #Otherwise, use ACF to estimate integral length scale and use this
            #value for the length scale in the Kaimal modeled spectrum
            lags = np.arange(0,ten_min_count)/float(frequency)
            u_corr = acvf(u_temp)
            u_acf = u_corr/u_corr[0]
            indices = np.arange(0,len(u_acf))
            x = indices[np.array(u_acf)<=0]
            #ACF is integrated to the first zero crossing to esimate the integral
            #time scale and multipled by the mean horizontal wind speed to estimate
            #the integral length scale
            L = np.trapz(u_acf[:x[0]],lags[:x[0]])*U

        fr2 = np.linspace(0,float(10)/2,float(6000/2))
        #Calculate Kaimal spectrum from 0 to 5 Hz
        S_model = Kaimal_spectrum_func((fr2,U,u_var),L)
        #Integrate spectrum for frequency values higher than those in the original
        #spectrum from the lidar
        var_diff.append(np.trapz((S_model[fr2 > fr[-1]]/fr2[fr2 > fr[-1]]),fr2[fr2 > fr[-1]]))


    return np.array(var_diff)

def var_correction(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode):
    #Uses Taylor's frozen turbulence hypothesis with data from the vertically
    #pointing beam to estimate new values of the u and v variance.

    #Inputs
    #vr_n, vr_e, vr_s, vr_w, vr_z: Time series of radial velocity from north-, east-, south, west-, and
    #vertically pointing beams, respectively, at height of interest.
    #wd: 10-min. Mean wind direction
    #U: 10-min. Mean horizontal wind speed
    #height_needed: Measurement height corresponding to velocity data
    #frequency_vert_beam: Sampling frequency of data from vertically pointing beam
    #el_angle: Elevation angle of off-vertical beam positions (in degrees, measured from the ground)
    #mode: Type of variance contamination correction to be applied. Options are taylor_ws and taylor_var.

    #Outputs
    #var_diff: Estimate of increase in streamwise variance due to variance contamination

    import numpy as np

    w_N = np.zeros(len(vr_z))
    w_N[:] = np.nan
    w_E = np.zeros(len(vr_z))
    w_E[:] = np.nan
    w_S = np.zeros(len(vr_z))
    w_S[:] = np.nan
    w_W = np.zeros(len(vr_z))
    w_W[:] = np.nan

    u_bar = np.sin(np.radians(wd - 180))*U
    v_bar = np.cos(np.radians(wd - 180))*U

    delta_t_vert_beam = 1./frequency_vert_beam

    #Calculate the number of time steps needed for eddies to travel from one
    #side of the scanning circle to the other
    dist = height_needed/np.tan(np.radians(el_angle))
    delta_t_u = dist/u_bar
    interval_u = np.round(delta_t_u/delta_t_vert_beam)
    delta_t_v = dist/v_bar
    interval_v = np.round(delta_t_v/delta_t_vert_beam)

    #Estimate values of w at different sides of the scanning circle by using
    #Taylor's frozen turbulence hypothesis
    for i in range(len(vr_z)):
        try:
            w_N[i] = vr_z[i-interval_v]
            w_E[i] = vr_z[i-interval_u]
        except:
            w_N[i] = np.nan
            w_E[i] = np.nan
        try:
            w_S[i] = vr_z[i+interval_v]
            w_W[i] = vr_z[i+interval_u]
        except:
            w_S[i] = np.nan
            w_W[i] = np.nan

    if "taylor_ws" in mode:
        #Use the new values of w to estimate the u and v components using the DBS technique
        #and calculate the variance
        u_DBS_new = ((vr_e-vr_w) - (w_E-w_W)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))
        v_DBS_new = ((vr_n-vr_s) - (w_N-w_S)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))

        u_var_lidar_new = get_10min_var(u_DBS_new,frequency_vert_beam)
        v_var_lidar_new = get_10min_var(v_DBS_new,frequency_vert_beam)
    else:
        #Calculate change in w across the scanning circle in north-south and east-west directions
        dw_est1 = w_S - w_N
        dw_est2 = w_W - w_E

        vr1_var = get_10min_var(vr_n,1./4)
        vr2_var = get_10min_var(vr_e,1./4)
        vr3_var = get_10min_var(vr_s,1./4)
        vr4_var = get_10min_var(vr_w,1./4)
        dw_var1 = get_10min_var(dw_est1,1./4)
        dw_var2 = get_10min_var(dw_est2,1./4)

        vr1_vr3_var = get_10min_covar(vr_n,vr_s,1./4)
        vr2_vr4_var = get_10min_covar(vr_e,vr_w,1./4)

        vr1_dw_var = get_10min_covar(vr_n,dw_est1,1./4)
        vr3_dw_var = get_10min_covar(vr_s,dw_est1,1./4)
        vr2_dw_var = get_10min_covar(vr_e,dw_est2,1./4)
        vr4_dw_var = get_10min_covar(vr_w,dw_est2,1./4)

        #These equations are adapted from Newman et al. (2016), neglecting terms involving
        #du or dv, as these terms are expected to be small compared to dw
        #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
        #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
        #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
        u_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr2_var + vr4_var- 2*vr2_vr4_var + 2*vr2_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr4_dw_var*np.sin(np.radians(el_angle)) + dw_var2*np.sin(np.radians(el_angle))**2)

        v_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr1_var + vr3_var- 2*vr1_vr3_var + 2*vr1_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr3_dw_var*np.sin(np.radians(el_angle)) + dw_var1*np.sin(np.radians(el_angle))**2)

    #Rotate the variance into the mean wind direction
    #Note: The rotation should include a term with the uv covariance, but the
    #covariance terms are also affected by variance contamination. In Newman
    #et al. (2016), it was found that the uv covariance is usually close to 0 and
    #can safely be neglected.
    #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
    #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
    #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
    u_rot_var_new = u_var_lidar_new*(np.sin(np.radians(wd)))**2 + v_var_lidar_new*(np.cos(np.radians(wd)))**2

    #Calculate the wind speed and variance if w is assumed to be the same on all
    #sides of the scanning circle
    u_DBS = (vr_e-vr_w)/(2*np.cos(np.radians(el_angle)))
    v_DBS = (vr_n-vr_s)/(2*np.cos(np.radians(el_angle)))

    u_var_DBS = get_10min_var(u_DBS,frequency_vert_beam)
    v_var_DBS = get_10min_var(v_DBS,frequency_vert_beam)

    u_rot_var = u_var_DBS*(np.sin(np.radians(wd)))**2 + v_var_DBS*(np.cos(np.radians(wd)))**2

    return u_rot_var-u_rot_var_new


def lidar_processing_vol_averaging(u,frequency,mode_ws,mode_vol):
    #Function to estimate variance lost due to volume/temporal averaging

    #Inputs
    #u: Time series of streamwise wind speed
    #frequency: Sampling frequency of time series
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #mode_vol: Type of volume averaging correction to be applied. Options are spectral_correction_fit and acf.

    #Outputs
    #var_diff: Estimate of loss of streamwise variance due to volume averaging

    var_diff = spectral_correction(u,frequency,mode_ws,mode_vol)

    return var_diff


def lidar_processing_var_contam(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode):
    #Function to estimate additional variance that results from variance contamination

    #Inputs
    #vr_n, vr_e, vr_s, vr_w, vr_z: Time series of radial velocity from north-, east-, south, west-, and
    #vertically pointing beams, respectively, at height of interest.
    #wd: 10-min. Mean wind direction
    #U: 10-min. Mean horizontal wind speed
    #height_needed: Measurement height corresponding to velocity data
    #frequency_vert_beam: Sampling frequency of data from vertically pointing beam
    #el_angle: Elevation angle of off-vertical beam positions (in degrees, measured from the ground)
    #mode: Type of variance contamination correction to be applied. Options are taylor_ws and taylor_var.

    #Outputs
    #var_diff: Estimate of increase in streamwise variance due to variance contamination

    var_diff = var_correction(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode)

    #Set negative values of var_diff to 0 as they would increase the corrected variance
    #Note: This is not the best procedure and should probably be fixed at some point.
    #It's possible that at times, the change in w across the scanning circle could
    #decrease, rather than increase, the u and v variance.

    try:
        if var_diff < 0:
            var_diff = 0.
        return var_diff
    except:
        var_diff = 0.

def VAD_func(az, x1, x2, x3):
    import numpy as np
    return np.array(x3+x1*np.cos(np.radians(az)-x2))


# def import_WC_file_VAD(filename,height_needed):
#     #Reads in WINDCUBE .rtd file and performs VAD technique at desired height. Outputs u, v, and w values from VAD technique,
#     #w values from vertical beam, measurement heights, and timestamp.
# 
#     #Inputs
#     #filename: WINDCUBE v2 .rtd file to read
#     #height_needed: Height where VAD analysis should be performed
# 
#     #Outputs
#     #u_VAD, v_VAD, w_VAD: u, v, and w values from VAD fit at height_needed
#     #vert_beam: Radial velocity from vertical beam at height_needed
#     #time_datenum: Timestamps corresponding to the start of each scan in datetime format
#     #time_datenum_vert_beam: Timestamps corresponding to vertical beam position in datetime format
# 
#     from scipy.optimize import curve_fit
#     from datetime import datetime
# 
#     inp = open(filename).readlines()
#     height_array = str.split(inp[38])
#     heights_temp = height_array[2:]
# 
#     if len(heights_temp) == 0:
#         height_array = str.split(inp[39])
#         heights_temp = height_array[2:]
# 
#     heights = [int(i) for i in heights_temp]
# 
#     height_needed_index = min_diff(heights,height_needed,6.1)
# 
#     num_rows = 41
#     timestamp = np.loadtxt(filename, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
# 
#     try:
#         datetime.strptime(timestamp[0],"%Y/%m/%d %H:%M:%S.%f")
#     except:
#         num_rows = 42
#         timestamp = np.loadtxt(filename, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
# 
#     time_datenum_temp = []
#     bad_rows = []
#     #Create list of rows where timestamp cannot be converted to datetime
#     for i in range(0,len(timestamp)):
#         try:
#             time_datenum_temp.append(datetime.strptime(timestamp[i],"%Y/%m/%d %H:%M:%S.%f"))
#         except:
#             bad_rows.append(i)
# 
#     #Delete all timestamp and datetime values from first bad row to end of dataset
#     if(bad_rows):
#         footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
#         timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
#         time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
#     else:
#         footer_lines = 0
# 
#     #Skip lines that correspond to bad data
#     az_angle = np.genfromtxt(filename, delimiter='\t', usecols=(1,),dtype=str, unpack=True,skip_header=num_rows,skip_footer=footer_lines)
#     vr_nan = np.empty(len(time_datenum_temp))
#     vr_nan[:] = np.nan
# 
#     vr = []
#     for i in range(1,len(heights)+1):
#         try:
#             vr.append(-np.genfromtxt(filename, delimiter='\t', usecols=(i*9 -4),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
#         except:
#             vr.append(vr_nan)
# 
#     vr  = np.array(vr)
#     vr = vr.transpose()
# 
#     bad_rows = []
#     #Find rows where time decreases instead of increasing
#     for i in range(1,len(time_datenum_temp)):
#         if time_datenum_temp[i] < time_datenum_temp[0]:
#             bad_rows.append(i)
# 
#     #Delete rows where time decreases instead of increasing
#     if(bad_rows):
#         vr = np.delete(vr,bad_rows,axis=0)
#         time_datenum_temp = np.delete(time_datenum_temp,bad_rows,axis=0)
#         timestamp = np.delete(timestamp,bad_rows,axis=0)
#         az_angle = np.delete(az_angle,bad_rows,axis=0)
# 
#     #Sort timestamp, vr, and az angle in order of ascending datetime value
#     timestamp_sorted = [timestamp[i] for i in np.argsort(time_datenum_temp)]
#     vr_sorted = np.array([vr[i,:] for i in np.argsort(time_datenum_temp)])
#     az_angle_sorted = np.array([az_angle[i] for i in np.argsort(time_datenum_temp)])
# 
#     vert_beam = []
#     vr_temp = []
#     az_temp = []
#     timestamp_az = []
#     timestamp_vert_beam = []
#     #Separate vertical beam values (where az angle = "V") from off-vertical beam values
#     for i in range(0,len(az_angle_sorted)):
#         if "V" in az_angle_sorted[i]:
#             vert_beam.append(vr_sorted[i,height_needed_index])
#             timestamp_vert_beam.append(timestamp_sorted[i])
#         else:
#             vr_temp.append(vr_sorted[i,height_needed_index])
#             az_temp.append(float(az_angle_sorted[i]))
#             timestamp_az.append(timestamp_sorted[i])
# 
#     vr_temp = np.array(vr_temp)
#     elevation = 62
#     u_VAD = []
#     v_VAD = []
#     w_VAD = []
#     timestamp_VAD = []
# 
# 
#     #Perform a VAD fit on each full scan
#     for i in range(0,round(len(az_temp)/4)):
#             x_vals = np.array(az_temp[i*4 + 1:i*4 + 5])
#             y_vals= np.array(vr_temp[i*4 + 1:i*4 + 5])
#             if len(y_vals[np.isnan(y_vals)]) == 0:
#                 #Initial guesses for the VAD fit parameters
#                 p0 = np.array([(np.max(y_vals)-np.min(y_vals))/2,2*np.pi,np.nanmean(y_vals)])
#                 popt, pcov = curve_fit(VAD_func, x_vals.ravel(), y_vals.ravel(),p0.ravel())
#                 ws_temp = popt[0]/np.cos(np.radians(elevation))
#                 wd_temp = np.degrees(popt[1]-np.pi)
#                 if wd_temp > 360:
#                     wd_temp-= 360
#                 u_VAD.append(np.sin(np.radians(wd_temp) - np.pi)*ws_temp)
#                 v_VAD.append(np.cos(np.radians(wd_temp) - np.pi)*ws_temp)
#                 w_VAD.append(popt[2]/np.sin(np.radians(elevation)))
#             else:
#                 u_VAD.append(np.nan)
#                 v_VAD.append(np.nan)
#                 w_VAD.append(np.nan)
#             timestamp_VAD.append(timestamp_az[i*4 +1])
# 
#     #Convert VAD and vertical beam timestamps to datetime format
# 
#     time_datenum = []
# 
#     for i in range(0,len(timestamp_VAD)):
#         time_datenum.append(datetime.strptime(timestamp_VAD[i],"%Y/%m/%d %H:%M:%S.%f"))
# 
#     time_datenum_vert_beam = []
# 
#     for i in range(0,len(timestamp_vert_beam)):
#         time_datenum_vert_beam.append(datetime.strptime(timestamp_vert_beam[i],"%Y/%m/%d %H:%M:%S.%f"))
# 
# 
#     return np.array(u_VAD),np.array(v_VAD),np.array(w_VAD),np.array(vert_beam)[:,0],np.array(time_datenum),np.array(time_datenum_vert_beam)
# 
def get_10min_var(ts,frequency):
    #Calculates variance for each 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #ts_var: 10-min. variance values from time series

    import numpy as np
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    ts_var = []

    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        ts_temp = ts[i:i+ten_min_count]
        ts_var.append(np.nanmean((ts_temp-np.nanmean(ts_temp))**2))

    return np.array(ts_var)

def get_10min_spectrum(ts,frequency):
    #Calculate power spectrum for 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #S_A_fast: Spectral power
    #frequency_fft: Frequencies correspond to spectral power values

    import numpy as np
    N = len(ts)
    delta_f = float(frequency)/N
    frequency_fft = np.linspace(0,float(frequency)/2,float(N/2))
    F_A_fast = np.fft.fft(ts)/N
    E_A_fast = 2*abs(F_A_fast[0:N/2]**2)
    S_A_fast = (E_A_fast)/delta_f
    return S_A_fast,frequency_fft

# def get_10min_spectrum_WC_raw(ts,frequency):
#     #Calculate power spectrum for 10-min. period
# 
#     #Inputs
#     #ts: Time series of data
#     #frequency: Sampling frequency of data
# 
#     #Outputs
#     #S_A_fast: Spectral power
#     #frequency_fft: Frequencies correspond to spectral power values
# 
#     import numpy as np
#     N = len(ts)
#     delta_f = float(frequency)/N
#     frequency_fft = np.linspace(0,float(frequency)/2,float(N/2))
#     F_A_fast = np.fft.fft(ts)/N
#     E_A_fast = 2*abs(F_A_fast[0:N/2]**2)
#     S_A_fast = (E_A_fast)/delta_f
#     #Data are only used for frequencies lower than 0.125 Hz. Above 0.125 Hz, the
#     #WINDCUBE spectrum calculated using raw data begins to show an artifact. This
#     #artifact is due to the recording of the u, v, and w components for every beam
#     #position, which results in repeating components.
#     S_A_fast = S_A_fast[frequency_fft <= 0.125]
#     frequency_fft = frequency_fft[frequency_fft <= 0.125]
#     return S_A_fast,frequency_fft
# 
# 
# def import_WC_file(filename):
#     #Reads in WINDCUBE .rtd file and outputs raw u, v, and w components, measurement heights, and timestamp.
# 
#     #Inputs
#     #filename: WINDCUBE v2 .rtd file to read
# 
#     #Outputs
#     #u_sorted, v_sorted, w_sorted: Raw u, v, and w values from all measurement heights
#     #heights: Measurement heights from file
#     #time_datenum_sorted: All timestamps in datetime format
# 
#     import numpy as np
#     from datetime import datetime
#     import codecs
# 
#     #Read in row containing heights (either row 38 or 39) and convert heights to a set of integers.
#     #inp = open(filename,encoding='ISO-8859-1').readlines()
#     inp = open(filename).readlines()
#     height_array = str.split(inp[38])
#     heights_temp = height_array[2:]
# 
#     if len(heights_temp) == 0:
#         height_array = str.split(inp[39])
#         heights_temp = height_array[2:]
# 
# 
#     heights = [int(i) for i in heights_temp]
# 
#     #Read in timestamps. There will be either 41 or 42 headerlines.
#     num_rows = 41
#     filecp = codecs.open(filename, encoding='ISO-8859-1')
#     timestamp = np.loadtxt(filename, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
# 
#     try:
#         datetime.strptime(timestamp[0],"%Y/%m/%d %H:%M:%S.%f")
#     except:
#         num_rows = 42
#         timestamp = np.loadtxt(filecp, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
# 
#     #Convert timestamps to Python datetime format. Some timestamps may be blank and will raise an error during the
#     #datetime conversion. The rows corresponding to these bad timestamps are recorded.
#     time_datenum_temp = []
#     bad_rows = []
#     for i in range(0,len(timestamp)):
#         try:
#             time_datenum_temp.append(datetime.strptime(timestamp[i],"%Y/%m/%d %H:%M:%S.%f"))
#         except:
#             bad_rows.append(i)
# 
#     #If bad timestamps are detected, an error message is output to the screen and all timestamps including and following
#     #the bad timestamp are deleted. The rows corresponding to these timestamps are categorized as footer lines and are not
#     #used when reading in the velocity data.
#     if(bad_rows):
#         print (filename,': Issue reading timestamp')
#         footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
#         timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
#         time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
#     else:
#         footer_lines = 0
# 
# 
# 
#     #Create column of NaNs for measurement heights that raise an error.
#     v_nan = np.empty(len(time_datenum_temp))
#     v_nan[:] = np.nan
# 
#     u = []
#     v = []
#     w = []
# 
#     #Read in values of u, v, and w one measurement height at a time. Definitions of the wind components are as follows:
#     #u is east-west wind (u > 0 means wind is coming from the west)
#     #v is north-south wind (v > 0 means wind is coming from the south)
#     #w is vertical wind (w > 0 means wind is upward)
# 
#     for i in range(1,len(heights)+1):
#         try:
#             u.append(-np.genfromtxt(filename, delimiter='\t', usecols=(i*9+1),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
#             v.append(-np.genfromtxt(filename, delimiter='\t', usecols=(i*9),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
#             w.append(-np.genfromtxt(filename, delimiter='\t', usecols=(i*9 + 2),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
#         except:
#             u.append(v_nan)
#             v.append(v_nan)
#             w.append(v_nan)
# 
#     u  = np.array(u).transpose()
#     v  = np.array(v).transpose()
#     w  = np.array(w).transpose()
# 
# 
#     #Check to make sure all timestamps follow the initial timestamp. If a particular timestamp is earlier than the first
#     #timestamp in the data file, this row is marked as a bad row and removed from the data.
#     bad_rows = []
#     for i in range(1,len(time_datenum_temp)):
#         if time_datenum_temp[i] < time_datenum_temp[0]:
#             bad_rows.append(i)
# 
#     if(bad_rows):
#         print (filename,': Issue with timestamp order')
#         u = np.delete(u,bad_rows,axis=0)
#         v = np.delete(v,bad_rows,axis=0)
#         w = np.delete(w,bad_rows,axis=0)
#         time_datenum_temp = np.delete(time_datenum_temp,axis=0)
#         timestamp = np.delete(timestamp,axis=0)
# 
#     #Sort data by timestamp to ensure that variables are in correct temporal order.
#     time_datenum_sorted = np.array([time_datenum_temp[i] for i in np.argsort(time_datenum_temp)])
#     u_sorted = np.array([u[i,:] for i in np.argsort(time_datenum_temp)])
#     v_sorted = np.array([v[i,:] for i in np.argsort(time_datenum_temp)])
#     w_sorted = np.array([w[i,:] for i in np.argsort(time_datenum_temp)])
# 
#     return u_sorted,v_sorted,w_sorted,heights,timestamp,time_datenum_sorted
# 
# def import_WC_file_vr(filename,height_needed):
#     #Reads in WINDCUBE .rtd file and extracts off-vertical radial wind speed components at desired height.
# 
#     #Inputs
#     #filename: WINDCUBE v2 .rtd file to read
#     #height_needed: Height where off-vertical measurements should be extracted
# 
#     #Outputs
#     #vr_n,vr_e,vr_s,vr_w: Time series from north-, east-, south-, and west-pointing beams,
#     #respectively, at height_needed
#     #time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w: Timestamps corresponding to
#     #north-, east-, south-, and west-pointing beams, respectively, in datetime format
# 
#     inp = open(filename).readlines()
#     height_array = str.split(inp[38])
#     heights_temp = height_array[2:]
# 
#     if len(heights_temp) == 0:
#         height_array = str.split(inp[39])
#         heights_temp = height_array[2:]
# 
#     heights = [int(i) for i in heights_temp]
# 
#     height_needed_index = min_diff(heights,height_needed,6.1)
# 
#     num_rows = 41
#     timestamp = np.loadtxt(filename, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
# 
#     try:
#         datetime.datetime.strptime(timestamp[0],"%Y/%m/%d %H:%M:%S.%f")
#     except:
#         num_rows = 42
#         timestamp = np.loadtxt(filename, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
# 
# 
#     time_datenum_temp = []
#     bad_rows = []
#     #Create list of rows where timestamp cannot be converted to datetime
#     for i in range(0,len(timestamp)):
#         try:
#             time_datenum_temp.append(datetime.datetime.strptime(timestamp[i],"%Y/%m/%d %H:%M:%S.%f"))
#         except:
#             bad_rows.append(i)
# 
#     #Delete all timestamp and datetime values from first bad row to end of dataset
#     if(bad_rows):
#         footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
#         timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
#         time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
#     else:
#         footer_lines = 0
# 
#     #Skip lines that correspond to bad data
#     az_angle = np.genfromtxt(filename, delimiter='\t', usecols=(1,),dtype=str, unpack=True,skip_header=num_rows,skip_footer=footer_lines)
# 
#     vr_nan = np.empty(len(time_datenum_temp))
#     vr_nan[:] = np.nan
# 
#     vr = []
#     for i in range(1,len(heights)+1):
#         try:
#             vr.append(-np.genfromtxt(filename, delimiter='\t', usecols=(i*9 -4),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
#         except:
#             vr.append(vr_nan)
# 
#     vr  = np.array(vr)
#     vr = vr.transpose()
# 
#     bad_rows = []
#     #Find rows where time decreases instead of increasing
#     for i in range(1,len(time_datenum_temp)):
#         if time_datenum_temp[i] < time_datenum_temp[0]:
#             bad_rows.append(i)
# 
#     #Delete rows where time decreases instead of increasing
#     if(bad_rows):
#         vr = np.delete(vr,bad_rows,axis=0)
#         time_datenum_temp = np.delete(time_datenum_temp,bad_rows,axis=0)
#         timestamp = np.delete(timestamp,bad_rows,axis=0)
#         az_angle = np.delete(az_angle,bad_rows,axis=0)
# 
#     #Sort timestamp, vr, and az angle in order of ascending datetime value
#     timestamp_sorted = [timestamp[i] for i in np.argsort(time_datenum_temp)]
#     vr_sorted = np.array([vr[i,:] for i in np.argsort(time_datenum_temp)])
#     az_angle_sorted = np.array([az_angle[i] for i in np.argsort(time_datenum_temp)])
# 
#     vr_sorted  = np.array(vr_sorted)
# 
#     vr_temp = []
#     az_temp = []
#     timestamp_az = []
#     vert_beam = []
#     timestamp_vert_beam = []
#     #Separate vertical beam values (where az angle = "V") from off-vertical beam values
#     for i in range(0,len(az_angle_sorted)):
#         if "V" in az_angle_sorted[i]:
#             vert_beam.append(vr_sorted[i,:])
#             timestamp_vert_beam.append(timestamp_sorted[i])
#         else:
#             vr_temp.append(vr_sorted[i,:])
#             az_temp.append(float(az_angle_sorted[i]))
#             timestamp_az.append(timestamp_sorted[i])
# 
#     vr_temp = np.array(vr_temp)
#     az_temp = np.array(az_temp)
# 
#     #Extract data for north-, east-, south-, and west-pointing beams at height of interest
#     vr_n = vr_temp[az_temp==0,height_needed_index]
#     vr_e = vr_temp[az_temp==90,height_needed_index]
#     vr_s = vr_temp[az_temp==180,height_needed_index]
#     vr_w = vr_temp[az_temp==270,height_needed_index]
# 
# 
#     #Convert timestamps to datetime format
#     time_datenum = []
#     time_datenum_vert_beam = []
# 
#     for i in range(0,len(timestamp_az)):
#         time_datenum.append(datetime.datetime.strptime(timestamp_az[i],"%Y/%m/%d %H:%M:%S.%f"))
# 
#     for i in range(0,len(timestamp_vert_beam)):
#         time_datenum_vert_beam.append(datetime.datetime.strptime(timestamp_vert_beam[i],"%Y/%m/%d %H:%M:%S.%f"))
# 
#     time_datenum = np.array(time_datenum)
#     time_datenum_vert_beam = np.array(time_datenum_vert_beam)
#     time_datenum_n = time_datenum[az_temp==0]
#     time_datenum_e = time_datenum[az_temp==90]
#     time_datenum_s = time_datenum[az_temp==180]
#     time_datenum_w = time_datenum[az_temp==270]
# 
#     return vr_n,vr_e,vr_s,vr_w,heights,time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w, vert_beam,time_datenum_vert_beam

def rotate_ws(u,v,w,frequency):
    #Performs coordinate rotation according to Eqs. 22-29 in Wilczak et al. (2001)
    #Reference: Wilczak, J. M., S. P. Oncley, and S. A. Stage, 2001: Sonic anemometer tilt correction algorithms.
    #Bound.-Layer Meteor., 99, 127–150.

    #Inputs
    #u, v, w: Time series of east-west, north-south, and vertical wind speed components, respectively
    #frequency: Sampling frequency of velocity

    #Outputs
    #u_rot, v_rot, w_rot: Rotated u, v, and w wind speed, with u rotated into the 10-min. mean wind direction and
    #the 10-min. mean of v and w forced to 0

    import numpy as np
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    u_rot = []
    v_rot = []
    w_rot = []

    #Perform coordinate rotation. First rotation rotates u into the mean wind direction and forces the mean v to 0.
    #Second rotation forces the mean w to 0.
    for i in np.arange(0,len(u)-ten_min_count+1,ten_min_count):
        u_temp = u[i:i+ten_min_count]
        v_temp = v[i:i+ten_min_count]
        w_temp = w[i:i+ten_min_count]
        phi_temp = np.arctan2(np.nanmean(v_temp),np.nanmean(u_temp))
        u1_temp = u_temp*np.cos(phi_temp) + v_temp*np.sin(phi_temp)
        v1_temp = -u_temp*np.sin(phi_temp) + v_temp*np.cos(phi_temp)
        w1_temp = w_temp;
        phi_temp2 = np.arctan2(np.nanmean(w1_temp),np.nanmean(u1_temp))
        u_rot.append(u1_temp*np.cos(phi_temp2) + w1_temp*np.sin(phi_temp2))
        v_rot.append(v1_temp)
        w_rot.append(-u1_temp*np.sin(phi_temp2) + w1_temp*np.cos(phi_temp2))

    return np.array(u_rot).ravel(),np.array(v_rot).ravel(),np.array(w_rot).ravel()

def get_10min_mean_ws_wd(u,v,time,frequency):
    #Calculates the 10-min. scalar average wind speed and wind direction at all measurement heights

    #Inputs
    #u: East-west velocity time series
    #v: North-south velocity time series
    #time: Timestamps in datetime format
    #frequency: Sampling frequency of velocity data

    #Outputs
    #U: 10-min. mean horizontal wind speeds
    #wd: 10-min. mean wind direction
    #time_datenum_10min: Timestamp corresponding to the start of each 10-min. averaging period

    import numpy as np
    ten_min_count = int(frequency*60*10)
    U = []
    wd = []
    time_datenum_10min = []

    for i in np.arange(0,len(u)-ten_min_count+1,ten_min_count):
        U_height = []
        wd_height = []
        #10-min. window of data
        if len(np.shape(u)) > 1:
            u_temp = u[i:i+ten_min_count,:]
            v_temp = v[i:i+ten_min_count,:]
        else:
            u_temp = u[i:i+ten_min_count]
            v_temp = v[i:i+ten_min_count]
        for j in range(np.shape(u_temp)[1]):
            U_height.append(np.nanmean((u_temp[:,j]**2 + v_temp[:,j]**2)**0.5,axis=0));
            u_bar = np.nanmean(u_temp[:,j])
            v_bar = np.nanmean(v_temp[:,j])
            wd_height.append((180./np.pi)*(np.arctan2(u_bar,v_bar) + np.pi))
        U.append(U_height)
        wd.append(wd_height)
        time_datenum_10min.append(time[i])
    return np.array(U),np.array(wd),time_datenum_10min

def get_10min_shear_parameter(U,heights,height_needed):
    import functools

    #Calculates the shear parameter for every 10-min. period of data by fitting power law equation to
    #10-min. mean wind speeds

    #Inputs
    #U: 10-min. mean horizontal wind speed at all measurement heights
    #heights: Measurement heights
    #height_needed: Height where TI is being extracted - values used to calculate shear parameter
    #should be centered around this height

    #Outputs
    #p: 10-min. values of shear parameter

    import warnings
    p = []

    #Set heights for calculation of shear parameter and find corresponding indices
    zprofile = np.arange(0.5*height_needed,1.5*height_needed + 10,10)
    height_indices = np.unique(min_diff(heights,zprofile,5))
    height_indices = height_indices[~np.isnan(height_indices)]

    #Arrays of height and mean wind speed to use for calculation
    heights_temp = np.array([heights[int(i)] for i in height_indices])
    U_temp = np.array([U[:,int(i)] for i in height_indices])

    mask = [~np.isnan(U_temp)]
    mask = functools.reduce(np.logical_and, mask)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')

    #For each set of 10-min. U values, use linear fit to determine value of shear parameter
    for i in range(0,len(U)):
        try:
            try:
                p_temp = np.polyfit(np.log(heights_temp[mask[:,i]]),np.log(U_temp[mask[:,i],i]),1)
                p.append(p_temp[0])
            except np.RankWarning:
                p.append(np.nan)
        except:
            p.append(np.nan)
    return np.array(p)


def interp_ts(ts,time_datenum,interval):
    #Interpolates time series ts with timestamps time_datenum to a grid with constant temporal spacing of "interval"

    #Inputs
    #ts: Time series for interpolation
    #time_datenum: Original timestamps for time series in datetime format
    #interval: Temporal interval to use for interpolation

    #Outputs
    #ts_interp: Interpolated time series
    #time_interp: Timestamps of interpolated time series in datetime format

    import numpy as np
    from datetime import datetime
    import calendar as cal

    #Convert timestamps to unix time (seconds after 1970 01-01) as it's easier to perform the interpolation
    unix_time = []
    for i in range(0,len(time_datenum)):
        unix_time.append(cal.timegm(datetime.timetuple(time_datenum[i])) + (time_datenum[i].microsecond/1e6))

    unix_time = np.array(unix_time)

    #Select the start and end time for the interpolation

    #The starting minute value of the interpolation should be the next multiple of 10
    if time_datenum[0].minute%10 == 0:
        start_minute = str((time_datenum[0].minute//10)*10)
    else:
        start_minute = str((time_datenum[0].minute//10 + 1)*10)

    start_hour = str(time_datenum[0].hour)

    if int(start_minute) == 60:
        start_minute = '00'
        start_hour = str(time_datenum[0].hour + 1)


    end_hour = str(time_datenum[-1].hour)

    #The ending minute value of the interpolation should end with a 9
    if (time_datenum[-1].minute-9)%10 == 0:
        end_minute = str((time_datenum[-1].minute//10*10) + 9)
    else:
        end_minute = str((time_datenum[-1].minute//10)*10 - 1)


    if int(end_minute) < 0:
        end_minute = '59'
        end_hour = str(time_datenum[-1].hour - 1)

    #Convert start and end times into unix time and get interpolation times in unix time
    timestamp_start = str(time_datenum[0].year) + "/" + str(time_datenum[0].month) + "/" + str(time_datenum[0].day) + \
    " " + start_hour + ":" + start_minute + ":00"

    time_datenum_start = datetime.strptime(timestamp_start,"%Y/%m/%d %H:%M:%S")
    unix_time_start = cal.timegm(datetime.timetuple(time_datenum_start))

    timestamp_end = str(time_datenum[-1].year) + "/" + str(time_datenum[-1].month) + "/" + str(time_datenum[-1].day) + \
    " " + end_hour + ":" + end_minute + ":59"

    time_datenum_end = datetime.strptime(timestamp_end,"%Y/%m/%d %H:%M:%S")
    unix_time_end = cal.timegm(datetime.timetuple(time_datenum_end))

    time_interp_unix = np.arange(unix_time_start,unix_time_end+1,interval)
    #Interpolate time series

    ts_interp = []

    #If more than 75% of the data are valid, perform interpolation using only non-NaN data. (Every fifth point of the
    #u and v data will be NaNs because of the vertically pointing beam.)
    if float(len(ts[~np.isnan(ts)])/float(len(ts))) > 0.75:
        ts_temp = ts[~np.isnan(ts)]
        time_temp = unix_time[~np.isnan(ts)]
    else:
        ts_temp = ts
        time_temp = unix_time
    ts_interp = np.interp(time_interp_unix,time_temp,ts_temp)

    #If several points in a row have the same value, set these points to NaN. This can occur when the interpolation is
    #performed on a dataset with one valid value surrounded by several NaNs.
    for i in range(2,len(ts_interp)-2):
        if ts_interp[i-2] == ts_interp[i] and ts_interp[i+2] == ts_interp[i]:
            ts_interp[i-2:i+2] = np.nan

    time_interp = [datetime.utcfromtimestamp(int(i) + round(i-int(i),10)) for i in time_interp_unix]

    return np.transpose(ts_interp),time_interp


# def WC_processing_standard(filename,option,hub_height):
# 
#     if not "vr" in option:
#         if "raw" in option:
#             frequency = 1.
#             [u,v,w,heights,timestamp,time_datenum] = import_WC_file(filename)
#             u_interp = []
#             v_interp = []
#             w_interp = []
# 
#             for i in range(len(heights)):
#                 [u_interp_temp,time_interp] = interp_ts(u[:,i],time_datenum,1./frequency)
#                 [v_interp_temp,time_interp] = interp_ts(v[:,i],time_datenum,1./frequency)
#                 [w_interp_temp,time_interp] = interp_ts(w[:,i],time_datenum,1./frequency)
#                 u_interp.append(u_interp_temp)
#                 v_interp.append(v_interp_temp)
#                 w_interp.append(w_interp_temp)
# 
#             u_interp = np.transpose(np.array(u_interp))
#             v_interp = np.transpose(np.array(v_interp))
#             w_interp = np.transpose(np.array(w_interp))
# 
#             hub_height_index = min_diff(heights,[hub_height],6.1)
# 
#             [U,wd,time_datenum_10min] = get_10min_mean_ws_wd(u_interp,v_interp,time_interp,frequency)
# 
#             if len(time_datenum_10min) != 0:
# 
#                 p = get_10min_shear_parameter(U,heights,hub_height)
#                 U = U[:,hub_height_index]
#                 [u_rot,v_rot,w_rot] = rotate_ws(u_interp[:,hub_height_index],v_interp[:,hub_height_index],\
#                 w_interp[:,hub_height_index],frequency)
#                 return u_rot,U[:,0],wd,p,time_datenum_10min,time_interp,hub_height_index,u_interp[:,hub_height_index],v_interp[:,hub_height_index]
#             else:
#                 p = []
#                 U = []
#                 [u_rot,v_rot,w_rot] = rotate_ws(u_interp[:,hub_height_index],v_interp[:,hub_height_index],\
#                 w_interp[:,hub_height_index],frequency)
#                 return u_rot,U,p,time_datenum_10min,time_interp,hub_height_index,u_interp[:,hub_height_index],v_interp[:,hub_height_index]
#                 #return u_rot,U[:,0],p,time_datenum_10min,time_interp,hub_height_index
#         elif "VAD" in option:
#             frequency = 1./4
#             [u,v,w,vert_beam,time_datenum,time_datenum_vert_beam] = import_WC_file_VAD(filename,hub_height)
# 
#             [u_interp,time_interp] = interp_ts(u,time_datenum,1./frequency)
#             [v_interp,time_interp] = interp_ts(v,time_datenum,1./frequency)
#             [w_interp,time_interp] = interp_ts(w,time_datenum,1./frequency)
#             [vert_beam_interp,time_interp] = interp_ts(vert_beam,time_datenum_vert_beam,1./frequency)
# 
# 
#             u_interp = np.transpose(np.array(u_interp))
#             v_interp = np.transpose(np.array(v_interp))
#             w_interp = np.transpose(np.array(w_interp))
#             vert_beam_interp = np.transpose(np.array(vert_beam_interp))
# 
#             u_interp = u_interp.reshape(len(u_interp),1)
#             v_interp = v_interp.reshape(len(v_interp),1)
#             w_interp = w_interp.reshape(len(w_interp),1)
#             vert_beam_interp = vert_beam_interp.reshape(len(vert_beam_interp),1)
# 
#             [U,wd,time_datenum_10min] = get_10min_mean_ws_wd(u_interp,v_interp,time_interp,frequency)
# 
# 
#             [u_rot,v_rot,w_rot] = rotate_ws(u_interp,v_interp,\
#             vert_beam_interp,frequency)
# 
#             return u_rot,U,w_rot,time_datenum_10min,time_interp
# 
#     else:
#         frequency = 1./4
#       #  [vr_n,vr_n_dispersion,vr_e,vr_e_dispersion,vr_s,vr_s_dispersion,vr_w,vr_w_dispersion,vert_beam,vert_beam_dispersion,
#       #   u_VAD,v_VAD,w_VAD,heights,time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w,time_datenum_vert_beam,time_datenum_VAD,
#       #   SNR_n,SNR_e,SNR_s,SNR_w,SNR_vert_beam] = import_WC_file_vr(filename,hub_height)
#         [vr_n,vr_e,vr_s,vr_w,heights,time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w,vert_beam,time_datenum_vert_beam] = import_WC_file_vr(filename,hub_height)
# 
#         vr_n_interp = []
#         vr_e_interp = []
#         vr_s_interp = []
#         vr_w_interp = []
# 
#         #Perform temporal interpolation
#         [vr_n_interp,time_interp] = interp_ts(vr_n,time_datenum_n,1./frequency)
#         [vr_e_interp,time_interp] = interp_ts(vr_e,time_datenum_e,1./frequency)
#         [vr_s_interp,time_interp] = interp_ts(vr_s,time_datenum_s,1./frequency)
#         [vr_w_interp,time_interp] = interp_ts(vr_w,time_datenum_w,1./frequency)
# 
# 
#         return vr_n_interp,vr_e_interp,vr_s_interp,vr_w_interp

def perform_SS_SS_correction(inputdata,All_class_data,primary_idx):
    '''
    simple site specific correction, but adjust each TKE class differently
    '''
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    _adjuster_SS_SS = Adjustments()

    className = 1
    items_corrected = []
    for item in All_class_data:
        temp = item[primary_idx]
        if temp.empty:
            pass
        else:
            inputdata_test = temp[temp['split'] == True].copy()
            inputdata_train = temp[temp['split'] == False].copy()
            if inputdata_test.empty or len(inputdata_test) < 2 or inputdata_train.empty or len(inputdata_train) < 2:
                pass
                items_corrected.append(inputdata_test)
            else:
                # get te correction for this TKE class
                full = pd.DataFrame()
                full['Ref_TI'] = inputdata_test['Ref_TI']
                full['RSD_TI'] = inputdata_test['RSD_TI']
                full = full.dropna()
                if len(full) < 2:
                    pass
                else:
                    model = _adjuster_SS_SS.get_regression(inputdata_train['RSD_TI'], inputdata_train['Ref_TI'])
                    m = model[0]
                    c = model[1]
                    RSD_TI = inputdata_test['RSD_TI'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI'] = RSD_TI
                items_corrected.append(inputdata_test)
        del temp
        className += 1

    correctedData = items_corrected[0]
    for item in items_corrected[1:]:
        correctedData = pd.concat([correctedData, item])
    results = post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')

    results['correction'] = ['SS-SS'] * len(results)
    results = results.drop(columns=['sensor','height'])

    return inputdata_test, results, m, c


def perform_SS_WS_correction(inputdata):
    '''
    correct ws before computing TI
    '''
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])

    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()
    _adjuster_SS_WS = Adjustments()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_WS_Ht1' in inputdata.columns and 'RSD_WS_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_WS_Ht2' in inputdata.columns and 'RSD_WS_Ht2' in inputdata.columns:
             results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_WS_Ht3' in inputdata.columns and 'RSD_WS_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_WS_Ht4' in inputdata.columns and 'RSD_WS_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        full = pd.DataFrame()
        full['Ref_WS'] = inputdata_test['Ref_WS']
        full['RSD_WS'] = inputdata_test['RSD_WS']
        full = full.dropna()
        if len(full) < 2:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            model = _adjuster_SS_WS.get_regression(inputdata_train['RSD_WS'], inputdata_train['Ref_WS'])
            m = model[0]
            c = model[1]
            RSD_WS = inputdata_test['RSD_WS']
            RSD_SD = inputdata_test['RSD_SD']
            RSD_corrWS = (model[0]*RSD_WS) + model[1]
            inputdata_test['RSD_corrWS'] = RSD_corrWS
            RSD_TI = RSD_SD/inputdata_test['RSD_corrWS']
            inputdata_test['corrTI_RSD_TI'] = RSD_TI
            results = post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_WS_Ht1' in inputdata.columns and 'RSD_WS_Ht1' in inputdata.columns and 'RSD_SD_Ht1' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht1']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht1']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(inputdata_train['RSD_WS_Ht1'], inputdata_train['Ane_WS_Ht1'])
                RSD_WS = inputdata_test['RSD_WS_Ht1']

                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                inputdata_test['RSD_corrWS_Ht1'] = RSD_corrWS
                RSD_TI = inputdata_test['RSD_SD_Ht1']/inputdata_test['RSD_corrWS_Ht1']
                inputdata_test['corrTI_RSD_TI_Ht1'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_WS_Ht2' in inputdata.columns and 'RSD_WS_Ht2' in inputdata.columns and 'RSD_SD_Ht2' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht2']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht2']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(inputdata_train['RSD_WS_Ht2'],inputdata_train['Ane_WS_Ht2'])
                RSD_WS = inputdata_test['RSD_WS_Ht2']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                inputdata_test['RSD_corrWS_Ht2'] = RSD_corrWS
                RSD_TI = inputdata_test['RSD_SD_Ht2']/inputdata_test['RSD_corrWS_Ht2']
                inputdata_test['corrTI_RSD_TI_Ht2'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_WS_Ht3' in inputdata.columns and 'RSD_WS_Ht3' in inputdata.columns and 'RSD_SD_Ht3' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht3']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht3']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(inputdata_train['RSD_WS_Ht3'], inputdata_train['Ane_WS_Ht3'])
                RSD_WS = inputdata_test['RSD_WS_Ht3']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                inputdata_test['RSD_corrWS_Ht3'] = RSD_corrWS
                RSD_TI = inputdata_test['RSD_SD_Ht3']/inputdata_test['RSD_corrWS_Ht3']
                inputdata_test['corrTI_RSD_TI_Ht3'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_WS_Ht4' in inputdata.columns and 'RSD_WS_Ht4' in inputdata.columns and 'RSD_SD_Ht4' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht4']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht4']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS.get_regression(inputdata_train['RSD_WS_Ht4'], inputdata_train['Ane_WS_Ht4'])
                RSD_WS = inputdata_test['RSD_WS_Ht4']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                inputdata_test['RSD_corrWS_Ht4'] = RSD_corrWS
                RSD_TI = inputdata_test['RSD_SD_Ht4']/inputdata_test['RSD_corrWS_Ht4']
                inputdata_test['corrTI_RSD_TI_Ht4'] = RSD_TI
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    results['correction'] = ['SS-WS'] * len(results)
    results = results.drop(columns=['sensor','height'])

    return inputdata_test, results, m, c


def perform_SS_WS_Std_correction(inputdata):
    '''
    correct ws and std before computing TI
    '''
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])

    _adjuster_SS_WS_Std = Adjustments()
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()
    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_WS_Ht1' in inputdata.columns and 'RSD_WS_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_WS_Ht2' in inputdata.columns and 'RSD_WS_Ht2' in inputdata.columns:
             results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_WS_Ht3' in inputdata.columns and 'RSD_WS_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_WS_Ht4' in inputdata.columns and 'RSD_WS_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
        inputdata = False
    else:
        full = pd.DataFrame()
        full['Ref_WS'] = inputdata_test['Ref_WS']
        full['RSD_WS'] = inputdata_test['RSD_WS']
        full['Ref_SD'] = inputdata_test['Ref_SD']
        full['RSD_Sd'] = inputdata_test['RSD_SD']
        full = full.dropna()
        if len(full) < 2:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            model = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_WS'], inputdata_train['Ref_WS'])
            model_std = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_SD'], inputdata_train['Ref_SD'])
            m = model[0]
            c = model[1]
            m_std = model_std[0]
            c_std = model_std[1]
            RSD_WS = inputdata_test['RSD_WS']
            RSD_SD = inputdata_test['RSD_SD']
            RSD_corrWS = (model[0]*RSD_WS) + model[1]
            RSD_corrSD = (model_std[0]*RSD_SD) + model_std[1]
            inputdata_test['RSD_corrWS'] = RSD_corrWS
            inputdata_test['RSD_corrSD'] = RSD_corrSD
            RSD_TI = inputdata_test['RSD_corrSD']/inputdata_test['RSD_corrWS']
            inputdata_test['corrTI_RSD_TI'] = RSD_TI
            results = post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_WS_Ht1' in inputdata.columns and 'RSD_WS_Ht1' in inputdata.columns and 'RSD_SD_Ht1' in inputdata.columns and 'Ane_SD_Ht1' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht1']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht1']
            full['Ref_SD'] = inputdata_test['Ane_SD_Ht1']
            full['RSD_SD'] = inputdata_test['RSD_SD_Ht1']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_WS_Ht1'], inputdata_train['Ane_WS_Ht1'])
                model_std = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_SD_Ht1'], inputdata_train['Ane_SD_Ht1'])
                RSD_WS = inputdata_test['RSD_WS_Ht1']
                RSD_SD = inputdata_test['RSD_SD_Ht1']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                RSD_corrSD = (model_std[0]*RSD_SD) + model_std[1]
                inputdata_test['RSD_corrWS_Ht1'] = RSD_corrWS
                inputdata_test['RSD_corrSD_Ht1'] = RSD_corrSD
                RSD_TI = inputdata_test['RSD_corrSD_Ht1']/inputdata_test['RSD_corrWS_Ht1']
                inputdata_test['corrTI_RSD_TI_Ht1'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_WS_Ht2' in inputdata.columns and 'RSD_WS_Ht2' in inputdata.columns and 'RSD_SD_Ht2' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht2']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht2']
            full['Ref_SD'] = inputdata_test['Ane_SD_Ht2']
            full['RSD_SD'] = inputdata_test['RSD_SD_Ht2']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_WS_Ht2'], inputdata_train['Ane_WS_Ht2'])
                model_std = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_SD_Ht2'], inputdata_train['Ane_SD_Ht2'])
                RSD_WS = inputdata_test['RSD_WS_Ht2']
                RSD_SD = inputdata_test['RSD_SD_Ht2']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                RSD_corrSD = (model_std[0]*RSD_SD) + model_std[1]
                inputdata_test['RSD_corrWS_Ht2'] = RSD_corrWS
                inputdata_test['RSD_corrSD_Ht2'] = RSD_corrSD
                RSD_TI = inputdata_test['RSD_corrSD_Ht2']/inputdata_test['RSD_corrWS_Ht2']
                inputdata_test['corrTI_RSD_TI_Ht2'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_WS_Ht3' in inputdata.columns and 'RSD_WS_Ht3' in inputdata.columns and 'RSD_SD_Ht3' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht3']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht3']
            full['Ref_SD'] = inputdata_test['Ane_SD_Ht3']
            full['RSD_SD'] = inputdata_test['RSD_SD_Ht3']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_WS_Ht3'], inputdata_train['Ane_WS_Ht3'])
                model_std = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_SD_Ht3'], inputdata_train['Ane_SD_Ht3'])
                RSD_WS = inputdata_test['RSD_WS_Ht3']
                RSD_SD = inputdata_test['RSD_SD_Ht3']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                RSD_corrSD = (model_std[0]*RSD_SD) + model_std[1]
                inputdata_test['RSD_corrWS_Ht3'] = RSD_corrWS
                inputdata_test['RSD_corrSD_Ht3'] = RSD_corrSD
                RSD_TI = inputdata_test['RSD_corrSD_Ht3']/inputdata_test['RSD_corrWS_Ht3']
                inputdata_test['corrTI_RSD_TI_Ht3'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_WS_Ht4' in inputdata.columns and 'RSD_WS_Ht4' in inputdata.columns and 'RSD_SD_Ht4' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht4']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht4']
            full['Ref_SD'] = inputdata_test['Ane_SD_Ht4']
            full['RSD_SD'] = inputdata_test['RSD_SD_Ht4']
            full = full.dropna()
            if len(full) < 2:
                results = post_correction_stats([None],results,'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                model = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_WS_Ht4'], inputdata_train['Ane_WS_Ht4'])
                model_std = _adjuster_SS_WS_Std.get_regression(inputdata_train['RSD_SD_Ht4'], inputdata_train['Ane_SD_Ht4'])
                RSD_WS = inputdata_test['RSD_WS_Ht4']
                RSD_SD = inputdata_test['RSD_SD_Ht4']
                RSD_corrWS = (model[0]*RSD_WS) + model[1]
                RSD_corrSD = (model_std[0]*RSD_SD) + model_std[1]
                inputdata_test['RSD_corrWS_Ht4'] = RSD_corrWS
                inputdata_test['RSD_corrSD_Ht4'] = RSD_corrSD
                RSD_TI = inputdata_test['RSD_corrSD_Ht4']/inputdata_test['RSD_corrWS_Ht4']
                inputdata_test['corrTI_RSD_TI_Ht4'] = RSD_TI
                results = post_correction_stats(inputdata_test,results,'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    results['correction'] = ['SS-WS-Std'] * len(results)
    results = results.drop(columns=['sensor','height'])

    return inputdata_test, results, m, c


def perform_Match_input(inputdata):
    '''
    correct the TI inputs separately
    '''
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
    else:
        full = pd.DataFrame()
        full['Ref_WS'] = inputdata_test['Ref_WS']
        full['RSD_WS'] = inputdata_test['RSD_WS']
        full = full.dropna()
        if len(full) < 10:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            WS_output = hist_match(inputdata_train,inputdata_test,'Ref_WS', 'RSD_WS')
            SD_output = hist_match(inputdata_train,inputdata_test,'Ref_SD', 'RSD_SD')
            inputdata_test['corrTI_RSD_TI'] = SD_output/ WS_output
            results = post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_WS_Ht1' in inputdata.columns and 'Ane_SD_Ht1' in inputdata.columns and 'RSD_WS_Ht1' in inputdata.columns and 'RSD_SD_Ht1' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht1']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht1']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(inputdata_train, inputdata_test, 'Ane_WS_Ht1', 'RSD_WS_Ht1')
                SD_output = hist_match(inputdata_train, inputdata_test, 'Ane_SD_Ht1', 'RSD_SD_Ht1')
                inputdata_test['corrTI_RSD_TI_Ht1'] = SD_output/ WS_output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_WS_Ht2' in inputdata.columns and 'Ane_SD_Ht2' in inputdata.columns and 'RSD_WS_Ht2' in inputdata.columns and 'RSD_SD_Ht2' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht2']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht2']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(inputdata_train,inputdata_test,'Ane_WS_Ht2', 'RSD_WS_Ht2')
                SD_output = hist_match(inputdata_train,inputdata_test,'Ane_SD_Ht2', 'RSD_SD_Ht2')
                inputdata_test['corrTI_RSD_TI_Ht2'] = SD_output/ WS_output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_WS_Ht3' in inputdata.columns and 'Ane_SD_Ht3' in inputdata.columns and 'RSD_WS_Ht3' in inputdata.columns and 'RSD_SD_Ht3' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht3']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht3']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(inputdata_train, inputdata_test, 'Ane_WS_Ht3', 'RSD_WS_Ht3')
                SD_output = hist_match(inputdata_train, inputdata_test, 'Ane_SD_Ht3', 'RSD_SD_Ht3')
                inputdata_test['corrTI_RSD_TI_Ht3'] = SD_output/ WS_output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_WS_Ht4' in inputdata.columns and 'Ane_SD_Ht4' in inputdata.columns and 'RSD_WS_Ht4' in inputdata.columns and 'RSD_SD_Ht4' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_WS'] = inputdata_test['Ane_WS_Ht4']
            full['RSD_WS'] = inputdata_test['RSD_WS_Ht4']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                WS_output = hist_match(inputdata_train, inputdata_test, 'Ane_WS_Ht4', 'RSD_WS_Ht4')
                SD_output = hist_match(inputdata_train, inputdata_test, 'Ane_SD_Ht4', 'RSD_SD_Ht4')
                inputdata_test['corrTI_RSD_TI_Ht4'] = SD_output/ WS_output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    results['correction'] = ['SS-Match_input'] * len(results)
    results = results.drop(columns=['sensor','height'])
    return inputdata_test, results


def perform_Match(inputdata):
    # manipulate histogram to match template histogram (ref) - virtually a look-up table.
    results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()

    if inputdata.empty or len(inputdata) < 2:
        results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
        m = np.NaN
        c = np.NaN
    else:
        full = pd.DataFrame()
        full['Ref_TI'] = inputdata_test['Ref_TI']
        full['RSD_TI'] = inputdata_test['RSD_TI']
        full = full.dropna()
        if len(full) < 10:
            results = post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            m = np.NaN
            c = np.NaN
        else:
            output = hist_match(inputdata_train,inputdata_test,'Ref_TI', 'RSD_TI')
            inputdata_test['corrTI_RSD_TI'] = output
            results = post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
        if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht1']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht1']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(inputdata_train, inputdata_test, 'Ane_TI_Ht1', 'RSD_TI_Ht1')
                inputdata_test['corrTI_RSD_TI_Ht1'] = output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
        if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht2']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht2']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(inputdata_train,inputdata_test,'Ane_TI_Ht2', 'RSD_TI_Ht2')
                inputdata_test['corrTI_RSD_TI_Ht2'] = output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
        if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht3']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht3']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(inputdata_train, inputdata_test, 'Ane_TI_Ht3', 'RSD_TI_Ht3')
                inputdata_test['corrTI_RSD_TI_Ht3'] = output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
        if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ane_TI_Ht4']
            full['RSD_TI'] = inputdata_test['RSD_TI_Ht4']
            full = full.dropna()
            if len(full) < 10:
                results = post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                m = np.NaN
                c = np.NaN
            else:
                output = hist_match(inputdata_train, inputdata_test, 'Ane_TI_Ht4', 'RSD_TI_Ht4')
                inputdata_test['corrTI_RSD_TI_Ht4'] = output
                results = post_correction_stats(inputdata_test,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

    import matplotlib.pyplot as plt
#    plt.plot(inputdata_test['Ref_TI'])
#    plt.plot(inputdata_test['RSD_TI'])
#    plt.plot(inputdata_test['corrTI_RSD_TI'])

#    plt.scatter(inputdata_test['Ref_TI'], inputdata_test['RSD_TI'], label='RefvsRSD')
#    plt.scatter(inputdata_test['Ref_TI'], inputdata_test['corrTI_RSD_TI'], label='RefvsCorrectedRSD')
#    plt.scatter(inputdata_test['Ref_TI'], inputdata_test['Ane2_TI'], label='RefvsRedundant')
#    plt.legend()
#    plt.show()
    results['correction'] = ['SS-Match'] * len(results)
    results = results.drop(columns=['sensor','height'])
    return inputdata_test, results


def calculate_stability_alpha(inputdata, config_file, RSD_alphaFlag, Ht_1_rsd, Ht_2_rsd):
    '''
    from Wharton and Lundquist 2012
    stability class from shear exponent categories:
    [1]     strongly stable -------- alpha > 0.3
    [2]              stable -------- 0.2 < alpha < 0.3
    [3]        near-neutral -------- 0.1 < TKE < 0.2
    [4]          convective -------- 0.0 < TKE < 0.1
    [5] strongly convective -------- alpha < 0.0
    '''

    regimeBreakdown_ane = pd.DataFrame()

    #check for 2 anemometer heights (use furthest apart) for cup alpha calculation
    configHtData = pd.read_excel(config_file, usecols=[3, 4], nrows=17).iloc[[3,12,13,14,15]]
    primaryHeight = configHtData['Selection'].to_list()[0]
    all_heights, ane_heights, RSD_heights, ane_cols, RSD_cols = config.check_for_additional_heights(primaryHeight)
    if len(list(ane_heights))> 1:
        all_keys = list(all_heights.values())
        max_key = list(all_heights.keys())[all_keys.index(max(all_heights.values()))]
        min_key = list(all_heights.keys())[all_keys.index(min(all_heights.values()))]
        if max_key == 'primary':
            max_cols = [s for s in inputdata.columns.to_list() if 'Ref' in s and 'WS' in s]
        else:
            subname = str('Ht' + str(max_key))
            max_cols = [s for s in inputdata.columns.to_list() if subname in s and 'Ane' in s and 'WS' in s]
        if min_key == 'primary':
            min_cols = [s for s in inputdata.columns.to_list() if 'Ref' in s and 'WS' in s]
        else:
            subname = str('Ht' + str(min_key))
            min_cols = [s for s in inputdata.columns.to_list() if subname in s and 'Ane' in s and 'WS' in s]

        # Calculate shear exponent
        tmp = pd.DataFrame(None)
        baseName = str(max_cols + min_cols)
        tmp[str(baseName + '_y')] = [val for sublist in log_of_ratio(inputdata[max_cols].values.astype(float),
                                                                     inputdata[min_cols].values.astype(float)) for val in sublist]
        tmp[str(baseName + '_alpha')] = tmp[str(baseName + '_y')] / (log_of_ratio(max(all_heights.values()), min(all_heights.values())))

        stabilityMetric_ane = tmp[str(baseName + '_alpha')]
        Ht_2_ane = max(all_heights.values())
        Ht_1_ane = min(all_heights.values())

        tmp[str(baseName + 'stabilityClass')] = tmp[str(baseName + '_alpha')]
        tmp.loc[(tmp[str(baseName + '_alpha')] <= 0.4), str(baseName + 'stabilityClass')] = 1
        tmp.loc[(tmp[str(baseName + '_alpha')] > 0.4) & (tmp[str(baseName + '_alpha')] <= 0.7), str(baseName + 'stabilityClass')] = 2
        tmp.loc[(tmp[str(baseName + '_alpha')] > 0.7) & (tmp[str(baseName + '_alpha')] <= 1.0), str(baseName + 'stabilityClass')] = 3
        tmp.loc[(tmp[str(baseName + '_alpha')] > 1.0) & (tmp[str(baseName + '_alpha')] <= 1.4), str(baseName + 'stabilityClass')] = 4
        tmp.loc[(tmp[str(baseName + '_alpha')] > 1.4), str(baseName + 'stabilityClass')] = 5

        # get count and percent of data in each class
        numNans = tmp[str(baseName) + '_alpha'].isnull().sum()
        totalCount = len(inputdata) - numNans
        name_class = str('stability_shear' + '_class')
        name_stabilityClass = str(baseName + 'stabilityClass')
        regimeBreakdown_ane[name_class] = ['1 (strongly stable)', '2 (stable)', '3 (near-neutral)', '4 (convective)', '5 (strongly convective)']
        name_count = str('stability_shear_obs' + '_count')
        regimeBreakdown_ane[name_count] = [len(tmp[(tmp[name_stabilityClass] == 1)]), len(tmp[(tmp[name_stabilityClass] == 2)]),
                                       len(tmp[(tmp[name_stabilityClass] == 3)]), len(tmp[(tmp[name_stabilityClass] == 4)]),
                                       len(tmp[(tmp[name_stabilityClass] == 5)])]
        name_percent = str('stability_shear_obs' + '_percent')
        regimeBreakdown_ane[name_percent] = [len(tmp[(tmp[name_stabilityClass] == 1)])/totalCount, len(tmp[(tmp[name_stabilityClass] == 2)])/totalCount,
                                         len(tmp[(tmp[name_stabilityClass] == 3)])/totalCount, len(tmp[(tmp[name_stabilityClass] == 4)])/totalCount,
                                         len(tmp[(tmp[name_stabilityClass] == 5)])/totalCount]
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
    if RSD_alphaFlag:
        regimeBreakdown_rsd = pd.DataFrame()
        tmp = pd.DataFrame(None)
        baseName = str('WS_' + str(Ht_1_rsd) + '_' + 'WS_' + str(Ht_2_rsd))
        max_col = 'RSD_alpha_lowHeight'
        min_col = 'RSD_alpha_highHeight'
        tmp[str(baseName + '_y')] = log_of_ratio(inputdata[max_col].values.astype(float),inputdata[min_col].values.astype(float))
        tmp[str(baseName + '_alpha')] = tmp[str(baseName + '_y')] / (log_of_ratio(Ht_2_rsd, Ht_1_rsd))

        stabilityMetric_rsd = tmp[str(baseName + '_alpha')]

        tmp[str(baseName + 'stabilityClass')] = tmp[str(baseName + '_alpha')]
        tmp.loc[(tmp[str(baseName + '_alpha')] <= 0.4), str(baseName + 'stabilityClass')] = 1
        tmp.loc[(tmp[str(baseName + '_alpha')] > 0.4) & (tmp[str(baseName + '_alpha')] <= 0.7), str(baseName + 'stabilityClass')] = 2
        tmp.loc[(tmp[str(baseName + '_alpha')] > 0.7) & (tmp[str(baseName + '_alpha')] <= 1.0), str(baseName + 'stabilityClass')] = 3
        tmp.loc[(tmp[str(baseName + '_alpha')] > 1.0) & (tmp[str(baseName + '_alpha')] <= 1.4), str(baseName + 'stabilityClass')] = 4
        tmp.loc[(tmp[str(baseName + '_alpha')] > 1.4), str(baseName + 'stabilityClass')] = 5

        # get count and percent of data in each class
        numNans = tmp[str(baseName) + '_alpha'].isnull().sum()
        totalCount = len(inputdata) - numNans
        name_stabilityClass = str(baseName + 'stabilityClass')
        regimeBreakdown_rsd[name_class] = ['1 (strongly stable)', '2 (stable)', '3 (near-neutral)', '4 (convective)', '5 (strongly convective)']
        name_count = str('stability_shear_obs' + '_count')
        regimeBreakdown_rsd[name_count] = [len(tmp[(tmp[name_stabilityClass] == 1)]), len(tmp[(tmp[name_stabilityClass] == 2)]),
                                       len(tmp[(tmp[name_stabilityClass] == 3)]), len(tmp[(tmp[name_stabilityClass] == 4)]),
                                       len(tmp[(tmp[name_stabilityClass] == 5)])]
        name_percent = str('stability_shear_obs' + '_percent')
        regimeBreakdown_rsd[name_percent] = [len(tmp[(tmp[name_stabilityClass] == 1)])/totalCount, len(tmp[(tmp[name_stabilityClass] == 2)])/totalCount,
                                         len(tmp[(tmp[name_stabilityClass] == 3)])/totalCount, len(tmp[(tmp[name_stabilityClass] == 4)])/totalCount,
                                         len(tmp[(tmp[name_stabilityClass] == 5)])/totalCount]
        stabilityClass_rsd = tmp[name_stabilityClass]
    else:
        stabilityClass_rsd = None
        stabilityMetric_rsd = None
        regimeBreakdown_rsd = None
        Ht_1_rsd = None
        Ht_2_rsd = None

    return cup_alphaFlag,stabilityClass_ane, stabilityMetric_ane, regimeBreakdown_ane, Ht_1_ane, Ht_2_ane, stabilityClass_rsd, stabilityMetric_rsd, regimeBreakdown_rsd


def calculate_stability_TKE(inputdata):
    '''
    from Wharton and Lundquist 2012
    stability class from TKE categories:
    [1]     strongly stable -------- TKE < 0.4 m^(2)/s^(-2))
    [2]              stable -------- 0.4 < TKE < 0.7 m^(2)/s^(-2))
    [3]        near-neutral -------- 0.7 < TKE < 1.0 m^(2)/s^(-2))
    [4]          convective -------- 1.0 < TKE < 1.4 m^(2)/s^(-2))
    [5] strongly convective --------  TKE > 1.4 m^(2)/s^(-2))
    '''
    regimeBreakdown = pd.DataFrame()

    # check to see if instrument type allows the calculation
    if RSDtype['Selection']=='Triton':
        print ('Triton TKE calc')
    elif 'ZX' in RSDtype['Selection']:
        # look for pre-calculated TKE column
        TKE_cols = [s for s in inputdata.columns.to_list() if 'TKE' in s or 'tke' in s]
        if len(TKE_cols) < 1:
            print ('!!!!!!!!!!!!!!!!!!!!!!!! Warning: Input data does not include calculated TKE. Exiting tool. Either add TKE to input data or contact aea@nrgsystems.com for assistence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()
        else:
            for t in TKE_cols:
                name_stabilityClass = str(t + '_class')
                inputdata[name_stabilityClass] = inputdata[t]
                inputdata.loc[(inputdata[t] <= 0.4), name_stabilityClass] = 1
                inputdata.loc[(inputdata[t] > 0.4) & (inputdata[t] <= 0.7), name_stabilityClass] = 2
                inputdata.loc[(inputdata[t] > 0.7) & (inputdata[t] <= 1.0), name_stabilityClass] = 3
                inputdata.loc[(inputdata[t] > 1.0) & (inputdata[t] <= 1.4), name_stabilityClass] = 4
                inputdata.loc[(inputdata[t] > 1.4), name_stabilityClass] = 5

                # get count and percent of data in each class
                numNans = inputdata[t].isnull().sum()
                totalCount = len(inputdata) - numNans
                regimeBreakdown[name_stabilityClass] = ['1 (strongly stable)', '2 (stable)', '3 (near-neutral)', '4 (convective)', '5 (strongly convective)']
                name_count = str(name_stabilityClass.split('_class')[0] + '_count')
                regimeBreakdown[name_count] = [len(inputdata[(inputdata[name_stabilityClass] == 1)]), len(inputdata[(inputdata[name_stabilityClass] == 2)]),
                                               len(inputdata[(inputdata[name_stabilityClass] == 3)]), len(inputdata[(inputdata[name_stabilityClass] == 4)]),
                                               len(inputdata[(inputdata[name_stabilityClass] == 5)])]
                name_percent = str(name_stabilityClass.split('_class')[0] + '_percent')
                regimeBreakdown[name_percent] = [len(inputdata[(inputdata[name_stabilityClass] == 1)])/totalCount, len(inputdata[(inputdata[name_stabilityClass] == 2)])/totalCount,
                                               len(inputdata[(inputdata[name_stabilityClass] == 3)])/totalCount, len(inputdata[(inputdata[name_stabilityClass] == 4)])/totalCount,
                                               len(inputdata[(inputdata[name_stabilityClass] == 5)])/totalCount]

    elif 'WindCube' in RSDtype['Selection']:
        # convert to radians
        dir_cols = [s for s in inputdata.columns.to_list() if 'Direction' in s]
        if len(dir_cols)==0:
            stabilityClass = None
            stabilityMetric = None
            regimeBreakdown = None
            print ('Warning: Could not find direction columns in configuration key.  TKE derived stability, check data.')
            sys.exit()
        else:
            for c in dir_cols:
                name_radians = str(c + '_radians')
                inputdata[name_radians] = inputdata[c] * (math.pi/180)
                if name_radians.split('_')[2] == 'radians':
                    name_u_std = str(name_radians.split('_')[0] + '_u_std')
                    name_v_std = str(name_radians.split('_')[0] + '_v_std')
                else:
                    name_u_std = str(name_radians.split('_')[0] + '_' +  name_radians.split('_')[2] + '_u_std')
                    name_v_std = str(name_radians.split('_')[0] + '_' + name_radians.split('_')[2] + '_v_std')
                name_dispersion = None
                name_std = c.replace('Direction','SD')
                inputdata[name_u_std] = inputdata[name_std] * np.cos(inputdata[name_radians])
                inputdata[name_v_std] = inputdata[name_std] * np.sin(inputdata[name_radians])
                name_tke = str(name_u_std.split('_u')[0] + '_LidarTKE')
                inputdata[name_tke] = 0.5 * (inputdata[name_u_std]**2 + inputdata[name_v_std]**2 + inputdata[name_std]**2)
                name_stabilityClass = str(name_tke + '_class')
                inputdata[name_stabilityClass] = inputdata[name_tke]
                inputdata.loc[(inputdata[name_tke] <= 0.4), name_stabilityClass] = 1
                inputdata.loc[(inputdata[name_tke] > 0.4) & (inputdata[name_tke] <= 0.7), name_stabilityClass] = 2
                inputdata.loc[(inputdata[name_tke] > 0.7) & (inputdata[name_tke] <= 1.0), name_stabilityClass] = 3
                inputdata.loc[(inputdata[name_tke] > 1.0) & (inputdata[name_tke] <= 1.4), name_stabilityClass] = 4
                inputdata.loc[(inputdata[name_tke] > 1.4), name_stabilityClass] = 5

                # get count and percent of data in each class
                numNans = inputdata[name_tke].isnull().sum()
                totalCount = len(inputdata) - numNans
                name_class = str(name_u_std.split('_u')[0] + '_class')
                regimeBreakdown[name_class] = ['1 (strongly stable)', '2 (stable)', '3 (near-neutral)', '4 (convective)', '5 (strongly convective)']
                name_count = str(name_u_std.split('_u')[0] + '_count')
                regimeBreakdown[name_count] = [len(inputdata[(inputdata[name_stabilityClass] == 1)]), len(inputdata[(inputdata[name_stabilityClass] == 2)]),
                                               len(inputdata[(inputdata[name_stabilityClass] == 3)]), len(inputdata[(inputdata[name_stabilityClass] == 4)]),
                                               len(inputdata[(inputdata[name_stabilityClass] == 5)])]
                name_percent = str(name_u_std.split('_u')[0] + '_percent')
                regimeBreakdown[name_percent] = [len(inputdata[(inputdata[name_stabilityClass] == 1)])/totalCount, len(inputdata[(inputdata[name_stabilityClass] == 2)])/totalCount,
                                               len(inputdata[(inputdata[name_stabilityClass] == 3)])/totalCount, len(inputdata[(inputdata[name_stabilityClass] == 4)])/totalCount,
                                               len(inputdata[(inputdata[name_stabilityClass] == 5)])/totalCount]

    else:
        print ('Warning: Due to senor type, TKE is not being calculated.')
        stabilityClass = None
        stabilityMetric = None
        regimeBreakdown = None

    classCols = [s for s in inputdata.columns.to_list() if '_class' in s]
    stabilityClass = inputdata[classCols]
    tkeCols = [s for s in inputdata.columns.to_list() if '_LidarTKE' in s or 'TKE' in s or 'tke' in s]
    tkeCols = [s for s in tkeCols if '_class' not in s]
    stabilityMetric = inputdata[tkeCols]

    return stabilityClass, stabilityMetric, regimeBreakdown


def initialize_resultsLists(appendString):
    resultsLists = {}
    resultsLists[str('TI_MBEList' + '_' + appendString)] = []
    resultsLists[str('TI_DiffList' + '_' + appendString)] = []
    resultsLists[str('TI_DiffRefBinsList' + '_' + appendString)] = []
    resultsLists[str('TI_RMSEList' + '_' + appendString)] = []
    resultsLists[str('RepTI_MBEList' + '_' + appendString)] = []
    resultsLists[str('RepTI_DiffList' + '_' + appendString)] = []
    resultsLists[str('RepTI_DiffRefBinsList' + '_' + appendString)] = []
    resultsLists[str('RepTI_RMSEList' + '_' + appendString)] = []
    resultsLists[str('rep_TI_results_1mps_List' + '_' + appendString)] = []
    resultsLists[str('rep_TI_results_05mps_List' + '_' + appendString)] = []
    resultsLists[str('TIBinList' + '_' + appendString)] = []
    resultsLists[str('TIRefBinList' + '_' + appendString)] = []
    resultsLists[str('total_StatsList' + '_' + appendString)] = []
    resultsLists[str('belownominal_statsList' + '_' + appendString)] = []
    resultsLists[str('abovenominal_statsList' + '_' + appendString)] = []
    resultsLists[str('lm_CorrList' + '_' + appendString)] = []
    resultsLists[str('correctionTagList' + '_' + appendString)] = []
    resultsLists[str('Distribution_statsList' + '_' + appendString)] = []
    resultsLists[str('sampleTestsLists' + '_' + appendString)] = []
    return resultsLists


def train_test_split(trainPercent, inputdata, stepOverride = False):
    '''
    train is 'split' == True
    '''
    import copy
    import numpy as np

    _inputdata = pd.DataFrame(columns=inputdata.columns, data=copy.deepcopy(inputdata.values))

    if stepOverride:
        msk = [False] * len(inputdata)
        _inputdata['split'] = msk
        _inputdata.loc[stepOverride[0]:stepOverride[1], 'split'] =  True

    else:
        msk = np.random.rand(len(_inputdata)) < float(trainPercent/100)
        train = _inputdata[msk]
        test = _inputdata[~msk]
        _inputdata['split'] = msk

    return _inputdata


def quick_metrics(inputdata, results_df, lm_corr_dict, testID):
    """"""
    from TACT.computation.adjustments import Adjustments

    _adjuster = Adjustments(raw_data=inputdata)

    inputdata_train = inputdata[inputdata['split'] == True].copy()
    inputdata_test = inputdata[inputdata['split'] == False].copy()

    # baseline results
    results_ = get_all_regressions(inputdata_test, title='baselines')
    results_RSD_Ref = results_.loc[results_['baselines'].isin(['TI_regression_Ref_RSD'])].reset_index()
    results_Ane2_Ref = results_.loc[results_['baselines'].isin(['TI_regression_Ref_Ane2'])].reset_index()
    results_RSD_Ref_SD = results_.loc[results_['baselines'].isin(['SD_regression_Ref_RSD'])].reset_index()
    results_Ane2_Ref_SD = results_.loc[results_['baselines'].isin(['SD_regression_Ref_Ane2'])].reset_index()
    results_RSD_Ref_WS = results_.loc[results_['baselines'].isin(['WS_regression_Ref_RSD'])].reset_index()
    results_Ane2_Ref_WS = results_.loc[results_['baselines'].isin(['WS_regression_Ref_Ane2'])].reset_index()
    results_RSD_Ref.loc[0,'testID'] = [testID]
    results_Ane2_Ref.loc[0,'testID'] = [testID]
    results_RSD_Ref_SD.loc[0,'testID'] = [testID]
    results_Ane2_Ref_SD.loc[0,'testID'] = [testID]
    results_RSD_Ref_WS.loc[0,'testID'] = [testID]
    results_Ane2_Ref_WS.loc[0,'testID'] = [testID]
    results_df = pd.concat([results_df,results_RSD_Ref,results_Ane2_Ref,results_RSD_Ref_SD,results_Ane2_Ref_SD,
                            results_RSD_Ref_WS,results_Ane2_Ref_WS],axis = 0)

    # Run a few corrections with this timing test aswell
    inputdata_corr, lm_corr, m, c = _adjuster.perform_SS_S_correction(inputdata.copy())
    lm_corr_dict[str(str(testID) + ' :SS_S' )] = lm_corr
    inputdata_corr, lm_corr, m, c = _adjuster.perform_SS_SF_correction(inputdata.copy())
    lm_corr_dict[str(str(testID) + ' :SS_SF' )] = lm_corr
    inputdata_corr, lm_corr, m, c = perform_SS_WS_correction(inputdata.copy())
    lm_corr_dict[str(str(testID) + ' :SS_WS-Std' )] = lm_corr
    inputdata_corr, lm_corr = perform_Match(inputdata.copy())
    lm_corr_dict[str(str(testID) + ' :Match' )] = lm_corr
    inputdata_corr, lm_corr = perform_Match_input(inputdata.copy())
    lm_corr_dict[str(str(testID) + ' :SS_Match_erforminput' )] = lm_corr
    override = False
    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(inputdata.copy(),override)
    lm_corr_dict[str(str(testID) + ' :SS_G_SFa' )] = lm_corr

    return results_df, lm_corr_dict


def block_print():
    '''
    disable print statements
    '''
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    '''
    restore printing statements
    '''
    sys.stdout = sys.__stdout__


def record_TIadj(correctionName, inputdata_corr, Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False):

    if isinstance(inputdata_corr, pd.DataFrame) == False:
        pass
    else:
        corr_cols = [s for s in inputdata_corr.columns.to_list() if 'corr' in s]
        corr_cols = [s for s in corr_cols if not ('diff' in s or 'Diff' in s or 'error' in s)]
        for c in corr_cols:
            TI_10minuteAdjusted[str(c + '_' + method)] = inputdata_corr[c]

    return TI_10minuteAdjusted


def populate_resultsLists(resultDict, appendString, correctionName, lm_corr, inputdata_corr,
                            Timestamps, method, emptyclassFlag = False):
    """"""

    if isinstance(inputdata_corr, pd.DataFrame) == False:
        emptyclassFlag = True
    elif inputdata_corr.empty:
        emptyclassFlag = True
    else:
        try:
            TI_MBE_j_, TI_Diff_j_, TI_RMSE_j_, RepTI_MBE_j_, RepTI_Diff_j_, RepTI_RMSE_j_ = get_TI_MBE_Diff_j(inputdata_corr)
            TI_Diff_r_, RepTI_Diff_r_ = get_TI_Diff_r(inputdata_corr)
            rep_TI_results_1mps, rep_TI_results_05mps = get_representative_TI(inputdata_corr) # char TI but at bin level
            TIbybin = get_TI_bybin(inputdata_corr)
            TIbyRefbin = get_TI_byTIrefbin(inputdata_corr)
            total_stats, belownominal_stats, abovenominal_stats = get_description_stats(inputdata_corr)

        except:
            emptyclassFlag = True

    if emptyclassFlag == True:
        resultDict[str('TI_MBEList' + '_' + appendString)].append(None)
        resultDict[str('TI_DiffList' + '_' + appendString)].append(None)
        resultDict[str('TI_DiffRefBinsList' + '_' + appendString)].append(None)
        resultDict[str('TI_RMSEList' + '_' + appendString)].append(None)
        resultDict[str('RepTI_MBEList' + '_' + appendString)].append(None)
        resultDict[str('RepTI_DiffList' + '_' + appendString)].append(None)
        resultDict[str('RepTI_DiffRefBinsList' + '_' + appendString)].append(None)
        resultDict[str('RepTI_RMSEList' + '_' + appendString)].append(None)
        resultDict[str('rep_TI_results_1mps_List' + '_' + appendString)].append(None)
        resultDict[str('rep_TI_results_05mps_List' + '_' + appendString)].append(None)
        resultDict[str('TIBinList' + '_' + appendString)].append(None)
        resultDict[str('TIRefBinList' + '_' + appendString)].append(None)
        resultDict[str('total_StatsList' + '_' + appendString)].append(None)
        resultDict[str('belownominal_statsList' + '_' + appendString)].append(None)
        resultDict[str('abovenominal_statsList' + '_' + appendString)].append(None)
        resultDict[str('lm_CorrList' + '_' + appendString)].append(lm_corr)
        resultDict[str('correctionTagList' + '_' + appendString)].append(method)
        resultDict[str('Distribution_statsList' + '_' + appendString)].append(None)
        resultDict[str('sampleTestsLists' + '_' + appendString)].append(None)

    else:
        resultDict[str('TI_MBEList' + '_' + appendString)].append(TI_MBE_j_)
        resultDict[str('TI_DiffList' + '_' + appendString)].append(TI_Diff_j_)
        resultDict[str('TI_DiffRefBinsList' + '_' + appendString)].append(TI_Diff_r_)
        resultDict[str('TI_RMSEList' + '_' + appendString)].append(TI_RMSE_j_)
        resultDict[str('RepTI_MBEList' + '_' + appendString)].append(RepTI_MBE_j_)
        resultDict[str('RepTI_DiffList' + '_' + appendString)].append(RepTI_Diff_j_)
        resultDict[str('RepTI_DiffRefBinsList' + '_' + appendString)].append(RepTI_Diff_r_)
        resultDict[str('RepTI_RMSEList' + '_' + appendString)].append(RepTI_RMSE_j_)
        resultDict[str('rep_TI_results_1mps_List' + '_' + appendString)].append(rep_TI_results_1mps)
        resultDict[str('rep_TI_results_05mps_List' + '_' + appendString)].append(rep_TI_results_05mps)
        resultDict[str('TIBinList' + '_' + appendString)].append(TIbybin)
        resultDict[str('TIRefBinList' + '_' + appendString)].append(TIbyRefbin)
        resultDict[str('total_StatsList' + '_' + appendString)].append(total_stats)
        resultDict[str('belownominal_statsList' + '_' + appendString)].append(belownominal_stats)
        resultDict[str('abovenominal_statsList' + '_' + appendString)].append(abovenominal_stats)
        resultDict[str('lm_CorrList' + '_' + appendString)].append(lm_corr)
        resultDict[str('correctionTagList' + '_' + appendString)].append(method)
    try:
        Distribution_stats, sampleTests = Dist_stats(inputdata_corr, Timestamps,correctionName)
        resultDict[str('Distribution_statsList' + '_' + appendString)].append(Distribution_stats)
        resultDict[str('sampleTestsLists' + '_' + appendString)].append(sampleTests)

    except:
        resultDict[str('Distribution_statsList' + '_' + appendString)].append(None)
        resultDict[str('sampleTestsLists' + '_' + appendString)].append(None)

    return resultDict


def populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, appendString):

    ResultsLists_stability[str('TI_MBEList_stability' + '_' + appendString)].append(ResultsLists_class[str('TI_MBEList_class_' + appendString)])
    ResultsLists_stability[str('TI_DiffList_stability' + '_' + appendString)].append(ResultsLists_class[str('TI_DiffList_class_'  + appendString)])
    ResultsLists_stability[str('TI_DiffRefBinsList_stability' + '_' + appendString)].append(ResultsLists_class[str('TI_DiffRefBinsList_class_' + appendString)])
    ResultsLists_stability[str('TI_RMSEList_stability' + '_' + appendString)].append(ResultsLists_class[str('TI_RMSEList_class_' + appendString)])
    ResultsLists_stability[str('RepTI_MBEList_stability' + '_' + appendString)].append(ResultsLists_class[str('RepTI_MBEList_class_' + appendString)])
    ResultsLists_stability[str('RepTI_DiffList_stability' + '_' + appendString)].append(ResultsLists_class[str('RepTI_DiffList_class_' + appendString)])
    ResultsLists_stability[str('RepTI_DiffRefBinsList_stability' + '_' + appendString)].append(ResultsLists_class[str('RepTI_DiffRefBinsList_class_'  + appendString)])
    ResultsLists_stability[str('RepTI_RMSEList_stability' + '_' + appendString)].append(ResultsLists_class[str('RepTI_RMSEList_class_' + appendString)])
    ResultsLists_stability[str('rep_TI_results_1mps_List_stability' + '_' + appendString)].append(ResultsLists_class[str('rep_TI_results_1mps_List_class_' + appendString)])
    ResultsLists_stability[str('rep_TI_results_05mps_List_stability' + '_' + appendString)].append(ResultsLists_class[str('rep_TI_results_05mps_List_class_' + appendString)])
    ResultsLists_stability[str('TIBinList_stability' + '_' + appendString)].append(ResultsLists_class[str('TIBinList_class_' + appendString)])
    ResultsLists_stability[str('TIRefBinList_stability' + '_' + appendString)].append(ResultsLists_class[str('TIRefBinList_class_' + appendString)])
    ResultsLists_stability[str('total_StatsList_stability' + '_' + appendString)].append(ResultsLists_class[str('total_StatsList_class_' + appendString)])
    ResultsLists_stability[str('belownominal_statsList_stability' + '_' + appendString)].append(ResultsLists_class[str('belownominal_statsList_class_' + appendString)])
    ResultsLists_stability[str('abovenominal_statsList_stability' + '_' + appendString)].append(ResultsLists_class[str('abovenominal_statsList_class_' + appendString)])
    ResultsLists_stability[str('lm_CorrList_stability' + '_' + appendString)].append(ResultsLists_class[str('lm_CorrList_class_' + appendString)])
    ResultsLists_stability[str('correctionTagList_stability' + '_' + appendString)].append(ResultsLists_class[str('correctionTagList_class_' + appendString)])
    ResultsLists_stability[str('Distribution_statsList_stability' + '_' + appendString)].append(ResultsLists_class[str('Distribution_statsList_class_' + appendString)])
    ResultsLists_stability[str('sampleTestsLists_stability' + '_' + appendString)].append(ResultsLists_class[str('sampleTestsLists_class_' + appendString)])

    return ResultsLists_stability




if __name__ == '__main__':
    # Python 2 caveat: Only working for Python 3 currently
    if sys.version_info[0] < 3:
        raise Exception("Tool will not run at this time. You must be using Python 3, as running on Python 2 will encounter errors.")
    # ------------------------
    # set up and configuration
    # ------------------------
    """parser get_input_files"""
    config = Config()

    input_filename = config.input_filename
    config_file = config.config_file
    rtd_files = config.rtd_files
    results_filename = config.results_file
    saveModel = config.save_model_location
    timetestFlag = config.time_test_flag
    globalModel  = config.global_model

    """config object assignments"""
    outpath_dir = config.outpath_dir
    outpath_file = config.outpath_file

    """metadata parser"""
    config.get_site_metadata()
    siteMetadata = config.site_metadata

    config.get_filtering_metadata()
    filterMetadata = config.config_metadata

    config.get_adjustments_metadata()
    correctionsMetadata = config.adjustments_metadata
    RSDtype = config.RSDtype
    extrap_metadata = config.extrap_metadata
    extrapolation_type = config.extrapolation_type

    """data object assignments"""

    data=Data(input_filename, config_file)
    data.get_inputdata()
    data.get_refTI_bins()      # >> to data_file.py
    data.check_for_alphaConfig()

    inputdata = data.inputdata
    Timestamps = data.timestamps
    a = data.a
    lab_a = data.lab_a
    RSD_alphaFlag = data.RSD_alphaFlag
    Ht_1_rsd = data.Ht_1_rsd
    Ht_2_rsd = data.Ht_2_rsd

    """sensor, height"""
    sensor = config.model
    height = config.height


    print ('%%%%%%%%%%%%%%%%%%%%%%%%% Processing Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # -------------------------------
    # special handling for data types
    # -------------------------------
    stabilityFlag = False
    if RSDtype['Selection'][0:4] == 'Wind':
        stabilityFlag = True
    if RSDtype['Selection']=='ZX':
        stabilityFlag = True
        TI_computed = inputdata['RSD_SD']/inputdata['RSD_WS']
        RepTI_computed = TI_computed + 1.28 * inputdata['RSD_SD']
        inputdata = inputdata.rename(columns={'RSD_TI':'RSD_TI_instrument'})
        inputdata = inputdata.rename(columns={'RSD_RepTI':'RSD_RepTI_instrument'})
        inputdata['RSD_TI'] = TI_computed
        inputdata['RSD_RepTI'] = RepTI_computed
    elif RSDtype['Selection']=='Triton':
        print ('RSD type is triton, not that output uncorrected TI is instrument corrected')
    # ------------------------
    # Baseline Results
    # ------------------------
    # Get all regressions available
    reg_results = get_all_regressions(inputdata, title='Full comparison')

    stabilityClass_tke, stabilityMetric_tke, regimeBreakdown_tke = calculate_stability_TKE(inputdata)
    cup_alphaFlag, stabilityClass_ane, stabilityMetric_ane, regimeBreakdown_ane, Ht_1_ane, Ht_2_ane, stabilityClass_rsd, stabilityMetric_rsd, regimeBreakdown_rsd = calculate_stability_alpha(inputdata, config_file, RSD_alphaFlag, Ht_1_rsd, Ht_2_rsd)
    #------------------------
    # Time Sensivity Analysis
    #------------------------
    # TimeTestA = pd.DataFrame()
    # TimeTestB = pd.DataFrame()
    # TimeTestC = pd.DataFrame()

    if timetestFlag == True:
        # A) increase % of test train split -- check for convergence --- basic metrics recorded baseline but also for every corrections
        splitList = np.linspace(0.0, 100.0, num = 20, endpoint =False)
        print ('Testing model generation time period sensitivity...% of data')
        TimeTestA_corrections_df = {}
        TimeTestA_baseline_df = pd.DataFrame()

        for s in splitList[1:]:

            sys.stdout.write("\r")
            sys.stdout.write(f"{str(s).rjust(10, ' ')} %      ")

            inputdata_test = train_test_split(s,inputdata.copy())
            TimeTestA_baseline_df, TimeTestA_corrections_df = quick_metrics(inputdata_test, TimeTestA_baseline_df, TimeTestA_corrections_df,str(100-s))

        sys.stdout.flush()
        print()

        # B) incrementally Add days to training set sequentially -- check for convergence
        numberofObsinOneDay = 144
        numberofDaysInTest = int(round(len(inputdata)/numberofObsinOneDay))
        print ('Testing model generation time period sensitivity...days to train model')
        print ('Number of days in the study ' + str(numberofDaysInTest))
        TimeTestB_corrections_df = {}
        TimeTestB_baseline_df = pd.DataFrame()

        for i in range(0,numberofDaysInTest):

            sys.stdout.write("\r")
            sys.stdout.write(f"{str(i).rjust(10, ' ')} of {str(numberofDaysInTest)} days   ")

            windowEnd = (i+1)*(numberofObsinOneDay)
            inputdata_test = train_test_split(i,inputdata.copy(), stepOverride = [0,windowEnd])
            TimeTestB_baseline_df, TimeTestB_corrections_df = quick_metrics(inputdata_test,TimeTestB_baseline_df, TimeTestB_corrections_df,str(numberofDaysInTest-i))

        sys.stdout.flush()
        print()

        # C) If experiment is greater than 3 months, slide a 6 week window (1 week step)
        if len(inputdata) > (numberofObsinOneDay*90): # check to see if experiment is greater than 3 months
            print ('Testing model generation time period sensitivity...6 week window pick')
            windowStart = 0
            windowEnd = (numberofObsinOneDay*42)
            TimeTestC_corrections_df = {}
            TimeTestC_baseline_df = pd.DataFrame()

            while windowEnd < len(inputdata):
                print (str('After observation #' + str(windowStart) + ' ' + 'Before observation #' + str(windowEnd)))
                windowStart += numberofObsinOneDay*7
                windowEnd = windowStart + (numberofObsinOneDay*42)
                inputdata_test = train_test_split(i,inputdata.copy(), stepOverride = [windowStart,windowEnd])
                TimeTestC_baseline_df, TimeTestC_corrections_df = quick_metrics(inputdata_test, TimeTestC_baseline_df, TimeTestC_corrections_df,
                                                                              str('After_' + str(windowStart) + '_' + 'Before_' + str(windowEnd)))
    else:
        TimeTestA_baseline_df = pd.DataFrame()
        TimeTestB_baseline_df = pd.DataFrame()
        TimeTestC_baseline_df = pd.DataFrame()
        TimeTestA_corrections_df = {}
        TimeTestB_corrections_df = {}
        TimeTestC_corrections_df = {}

    #-----------------------
    # Test - Train split
    #-----------------------
    # random 80-20 split
    inputdata = train_test_split(80.0, inputdata.copy())

    inputdata_train = inputdata[inputdata['split'] == True].copy().join(Timestamps)
    inputdata_test = inputdata[inputdata['split'] == False].copy().join(Timestamps)

    timestamp_train = inputdata_train['Timestamp']
    timestamp_test = inputdata_test['Timestamp']

    #-----------------------------
    # stability class subset lists
    #-----------------------------
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

    if RSDtype['Selection'][0:4] == 'Wind' or 'ZX' in RSDtype['Selection']:

        inputdata_class1 = []
        inputdata_class2 = []
        inputdata_class3 = []
        inputdata_class4 = []
        inputdata_class5 = []
        RSD_h = []

        Alldata_inputdata = inputdata.copy()
        for h in stabilityClass_tke.columns.to_list():
            RSD_h.append(h)
            inputdata_class1.append(Alldata_inputdata[Alldata_inputdata[h] == 1])
            inputdata_class2.append(Alldata_inputdata[Alldata_inputdata[h] == 2])
            inputdata_class3.append(Alldata_inputdata[Alldata_inputdata[h] == 3])
            inputdata_class4.append(Alldata_inputdata[Alldata_inputdata[h] == 4])
            inputdata_class5.append(Alldata_inputdata[Alldata_inputdata[h] == 5])

        All_class_data = [inputdata_class1,inputdata_class2, inputdata_class3,
                          inputdata_class4, inputdata_class5]
        All_class_data_clean = [inputdata_class1, inputdata_class2, inputdata_class3,
                                inputdata_class4, inputdata_class5]

        for h in RSD_h:
            idx = RSD_h.index(h)
            df = inputdata_class1[idx]
            reg_results_class1.append(get_all_regressions(df, title = str('TKE_stability_' + h + 'class1')))
            df = inputdata_class2[idx]
            reg_results_class2.append(get_all_regressions(df, title = str('TKE_stability_' + h + 'class2')))
            df = inputdata_class3[idx]
            reg_results_class3.append(get_all_regressions(df, title = str('TKE_stability_' + h + 'class3')))
            df = inputdata_class4[idx]
            reg_results_class4.append(get_all_regressions(df, title = str('TKE_stability_' + h + 'class4')))
            df = inputdata_class5[idx]
            reg_results_class5.append(get_all_regressions(df, title = str('TKE_stability_' + h + 'class5')))

    if RSD_alphaFlag:
        del inputdata_class1, inputdata_class2, inputdata_class3, inputdata_class4, inputdata_class5

        Alldata_inputdata = inputdata.copy()
        colName = stabilityClass_rsd.name
        Alldata_inputdata[colName] = stabilityClass_rsd.values

        inputdata_class1=Alldata_inputdata[Alldata_inputdata[stabilityClass_rsd.name] == 1.0]
        inputdata_class2=Alldata_inputdata[Alldata_inputdata[stabilityClass_rsd.name] == 2.0]
        inputdata_class3=Alldata_inputdata[Alldata_inputdata[stabilityClass_rsd.name] == 3.0]
        inputdata_class4=Alldata_inputdata[Alldata_inputdata[stabilityClass_rsd.name] == 4.0]
        inputdata_class5=Alldata_inputdata[Alldata_inputdata[stabilityClass_rsd.name] == 5.0]

        All_class_data_alpha_RSD = [inputdata_class1,inputdata_class2, inputdata_class3,
                                    inputdata_class4, inputdata_class5]
        All_class_data_alpha_RSD_clean = [inputdata_class1.copy(),inputdata_class2.copy(), inputdata_class3.copy(),
                                    inputdata_class4.copy(), inputdata_class5.copy()]

        reg_results_class1_alpha['RSD'] = get_all_regressions(inputdata_class1, title = str('alpha_stability_RSD' + 'class1'))
        reg_results_class2_alpha['RSD'] = get_all_regressions(inputdata_class2, title = str('alpha_stability_RSD' + 'class2'))
        reg_results_class3_alpha['RSD'] = get_all_regressions(inputdata_class3, title = str('alpha_stability_RSD' + 'class3'))
        reg_results_class4_alpha['RSD'] = get_all_regressions(inputdata_class4, title = str('alpha_stability_RSD' + 'class4'))
        reg_results_class5_alpha['RSD'] = get_all_regressions(inputdata_class5, title = str('alpha_stability_RSD' + 'class5'))

    if cup_alphaFlag:
        del inputdata_class1, inputdata_class2, inputdata_class3, inputdata_class4, inputdata_class5

        Alldata_inputdata = inputdata.copy()
        colName = stabilityClass_ane.name
        Alldata_inputdata[colName] = stabilityClass_ane.values

        inputdata_class1 = Alldata_inputdata[Alldata_inputdata[stabilityClass_ane.name] == 1.0]
        inputdata_class2 = Alldata_inputdata[Alldata_inputdata[stabilityClass_ane.name] == 2.0]
        inputdata_class3 = Alldata_inputdata[Alldata_inputdata[stabilityClass_ane.name] == 3.0]
        inputdata_class4 = Alldata_inputdata[Alldata_inputdata[stabilityClass_ane.name] == 4.0]
        inputdata_class5 = Alldata_inputdata[Alldata_inputdata[stabilityClass_ane.name] == 5.0]

        All_class_data_alpha_Ane = [inputdata_class1,inputdata_class2, inputdata_class3,
                                    inputdata_class4, inputdata_class5]
        All_class_data_alpha_Ane_clean = [inputdata_class1.copy(),inputdata_class2.copy(), inputdata_class3.copy(),
                                          inputdata_class4.copy(), inputdata_class5.copy()]

        reg_results_class1_alpha['Ane'] = get_all_regressions(inputdata_class1, title = str('alpha_stability_Ane' + 'class1'))
        reg_results_class2_alpha['Ane'] = get_all_regressions(inputdata_class2, title = str('alpha_stability_Ane' + 'class2'))
        reg_results_class3_alpha['Ane'] = get_all_regressions(inputdata_class3, title = str('alpha_stability_Ane' + 'class3'))
        reg_results_class4_alpha['Ane'] = get_all_regressions(inputdata_class4, title = str('alpha_stability_Ane' + 'class4'))
        reg_results_class5_alpha['Ane'] = get_all_regressions(inputdata_class5, title = str('alpha_stability_Ane' + 'class5'))

    # ------------------------
    # TI AdjuAdjustments.perform_SS_S_correctionstments
    # ------------------------
    from TACT.computation.adjustments import Adjustments

    baseResultsLists = initialize_resultsLists('')

    # get number of observations in each bin
    count_1mps, count_05mps = get_count_per_WSbin(inputdata, 'RSD_WS')

    inputdata_train = inputdata[inputdata['split'] == True].copy().join(Timestamps)
    inputdata_test = inputdata[inputdata['split'] == False].copy().join(Timestamps)

    timestamp_train = inputdata_train['Timestamp']
    timestamp_test = inputdata_test['Timestamp']

    count_1mps_train, count_05mps_train = get_count_per_WSbin(inputdata_train, 'RSD_WS')
    count_1mps_test, count_05mps_test = get_count_per_WSbin(inputdata_test, 'RSD_WS')

    if RSDtype['Selection'][0:4] == 'Wind' or 'ZX' in RSDtype['Selection']:
        primary_c = [h for h in RSD_h if 'Ht' not in h]
        primary_idx = RSD_h.index(primary_c[0])
        ResultsLists_stability = initialize_resultsLists('stability_')
    if cup_alphaFlag:
        ResultsLists_stability_alpha_Ane = initialize_resultsLists('stability_alpha_Ane')
    if  RSD_alphaFlag:
        ResultsLists_stability_alpha_RSD = initialize_resultsLists('stability_alpha_RSD')

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

    for c in range(0,len(All_class_data)):
        name_1mps_tke.append(str('count_1mps_class_' + str(c) + '_tke'))
        name_1mps_alpha_Ane.append(str('count_1mps_class_' + str(c) + '_alpha_Ane'))
        name_1mps_alpha_RSD.append(str('count_1mps_class_' + str(c) + '_alpha_RSD'))
        name_05mps_tke.append(str('count_05mps_class_' + str(c) + '_tke'))
        name_05mps_alpha_Ane.append(str('count_05mps_class_' + str(c) + '_alpha_Ane'))
        name_05mps_alpha_RSD.append(str('count_05mps_class_' + str(c) + '_alpha_RSD'))

        try:
            c_1mps_tke, c_05mps_tke = get_count_per_WSbin(All_class_data[c][primary_idx], 'RSD_WS')
            count_1mps_tke.append(c_1mps_tke)
            count_05mps_tke.append(c_05mps_tke)
        except:
            count_1mps_tke.append(None)
            count_05mps_tke.append(None)
        try:
            c_1mps_alpha_Ane, c_05mps_alpha_Ane = get_count_per_WSbin(All_class_data_alpha_Ane[c], 'RSD_WS')
            count_1mps_alpha_Ane.append(c_1mps_alpha_Ane)
            count_05mps_alpha_Ane.append(c_05mps_alpha_Ane)
        except:
            count_1mps_alpha_Ane.append(None)
            count_05mps_alpha_Ane.append(None)
        try:
            c_1mps_alpha_RSD, c_05mps_alpha_RSD = get_count_per_WSbin(All_class_data_alpha_RSD[c], 'RSD_WS')
            count_1mps_alpha_RSD.append(c_1mps_alpha_RSD)
            count_05mps_alpha_RSD.append(c_05mps_alpha_RSD)
        except:
            count_1mps_alpha_RSD.append(None)
            count_05mps_alpha_RSD.append(None)


    # intialize 10 minute output
    TI_10minuteAdjusted = pd.DataFrame()

    # initialize Adjustments object
    adjuster = Adjustments(inputdata.copy(), correctionsMetadata, baseResultsLists)

    for method in correctionsMetadata:

        # ************************************ #
        # Site Specific Simple Correction (SS-S)
        if method != 'SS-S':
            pass
        elif method == 'SS-S' and correctionsMetadata['SS-S'] == False:
            pass
        else:
            print('Applying Correction Method: SS-S')
            logger.info('Applying Correction Method: SS-S')
            inputdata_corr, lm_corr, m, c = adjuster.perform_SS_S_correction(inputdata.copy())
            print("SS-S: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS-S'
            correctionName = 'SS_S'

            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-S by stability class (TKE)')
                logger.info('Applying Correction Method: SS-S by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c = adjuster.perform_SS_S_correction(item[primary_idx].copy())
                    print("SS-S: y = " + str(m) + " * x + " + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-S' + '_TKE_' + 'class_' + str(className))
                    correctionName = str('SS-S'+ '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: SS-S by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-S by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                print (str('class ' + str(className)))
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = adjuster.perform_SS_S_correction(item.copy())
                    print ("SS-S: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-S' + '_' + 'class_' + str(className))
                    correctionName = str('SS-S' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: SS-S by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-S by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = adjuster.perform_SS_S_correction(item.copy())
                    print ("SS-S: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-S' + '_alphaCup_' + 'class_' + str(className))
                    correctionName = str('SS-S' + '_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ********************************************** #
        # Site Specific Simple + Filter Correction (SS-SF)
        if method != 'SS-SF':
            pass
        elif method == 'SS-SF' and correctionsMetadata['SS-SF'] == False:
            pass
        else:
            print('Applying Correction Method: SS-SF')
            logger.info('Applying Correction Method: SS-SF')
           # inputdata_corr, lm_corr, m, c = perform_SS_SF_correction(inputdata.copy())
            inputdata_corr, lm_corr, m, c = adjuster.perform_SS_SF_correction(inputdata.copy())
            print("SS-SF: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS-SF'
            correctionName = 'SS_SF'

            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind' or 'ZX' in RSDtype['Selection']:
                print('Applying Correction Method: SS-SF by stability class (TKE)')
                logger.info('Applying Correction Method: SS-SF by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c = adjuster.perform_SS_SF_correction(item[primary_idx].copy())
                    print("SS-SF: y = " + str(m) + " * x + " + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-SF' + '_' + 'class_' + str(className))
                    correctionName = str('SS_SF' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: SS-SF by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-SF by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = adjuster.perform_SS_SF_correction(item.copy())
                    print ("SS-SF: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-SF' + '_' + 'class_' + str(className))
                    correctionName = str('SS_SF' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD,
                                                                                   ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: SS-SF by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-SF by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = adjuster.perform_SS_SF_correction(item.copy())
                    print ("SS-SF: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-SF' + '_' + 'class_' + str(className))
                    correctionName = str('SS_SF' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane,
                                                                                   ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ************************************ #
        # Site Specific Simple Correction (SS-SS) combining stability classes adjusted differently
        if method != 'SS-SS':
            pass
        elif method == 'SS-SS' and correctionsMetadata['SS-SS'] == False:
            pass
        elif RSDtype['Selection'][0:4] != 'Wind' and 'ZX' not in RSDtype['Selection']:
            pass
        else:
            print('Applying Correction Method: SS-SS')
            logger.info('Applying Correction Method: SS-SS')
            inputdata_corr, lm_corr, m, c = perform_SS_SS_correction(inputdata.copy(),All_class_data,primary_idx)
            print("SS-SS: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS-SS'
            correctionName = 'SS_SS'

            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-SS by stability class (TKE). SAME as Baseline')
                logger.info('Applying Correction Method: SS-SS by stability class (TKE). SAME as Baseline')
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    print("SS-SS: y = " + str(m) + " * x + " + str(c))
                    correctionName = str('SS_SS' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
            if RSD_alphaFlag:
                print('Applying Correction Method: SS-SS by stability class Alpha w/ RSD. SAEM as Baseline')
                logger.info('Applying Correction Method: SS-SS by stability class Alpha w/ RSD. SAEM as Baseline')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    print ("SS-SS: y = " + str(m) + "* x +" + str(c))
                    correctionName = str('SS_SS' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
            if cup_alphaFlag:
                print('Applying correction Method: SS-SS by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-SS by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    print ("SS-SS: y = " + str(m) + "* x +" + str(c))
                    emptyclassFlag = False
                    correctionName = str('SS_SS' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ******************************************* #
        # Site Specific WindSpeed Correction (SS-WS)
        if method != 'SS-WS':
            pass
        elif method == 'SS-WS' and correctionsMetadata['SS-WS'] == False:
            pass
        else:
            print('Applying Correction Method: SS-WS')
            logger.info('Applying Correction Method: SS-WS')
            inputdata_corr, lm_corr, m, c = perform_SS_WS_correction(inputdata.copy())
            print("SS-WS: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS-WS'
            correctionName = 'SS_WS'

            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind' or 'ZX' in RSDtype['Selection']:
                print('Applying Correction Method: SS-WS by stability class (TKE)')
                logger.info('Applying Correction Method: SS-WS by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c = perform_SS_WS_correction(item[primary_idx].copy())
                    print("SS-WS: y = " + str(m) + " * x + " + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-WS' + '_' + 'class_' + str(className))
                    correctionName = str('SS_WS' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
            if RSD_alphaFlag:
                print('Applying Correction Method: SS-WS by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-WS by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_SS_WS_correction(item.copy())
                    print ("SS-WS: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-WS' + '_' + 'class_' + str(className))
                    correctionName = str('SS_WS' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
            if cup_alphaFlag:
                print('Applying correction Method: SS-WS by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-WS by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_SS_WS_correction(item.copy())
                    print ("SS-WS: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-WS' + '_' + 'class_' + str(className))
                    emptyclassFlag = False
                    correctionName = str('SS_WS' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ******************************************* #
        # Site Specific Comprehensive Correction (SS-WS-Std)
        if method != 'SS-WS-Std':
            pass
        elif method == 'SS-WS-Std' and correctionsMetadata['SS-WS-Std'] == False:
            pass
        else:
           print('Applying Correction Method: SS-WS-Std')
           logger.info('Applying Correction Method: SS-WS-Std')
           inputdata_corr, lm_corr, m, c = perform_SS_WS_Std_correction(inputdata.copy())
           print("SS-WS-Std: y = " + str(m) + " * x + " + str(c))
           lm_corr['sensor'] = sensor
           lm_corr['height'] = height
           lm_corr['correction'] = 'SS-WS-Std'
           correctionName = 'SS_WS_Std'

           baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                    Timestamps, method)
           TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

           if RSDtype['Selection'][0:4] == 'Wind' or 'ZX' in RSDtype['Selection']:
               print('Applying Correction Method: SS-WS-Std by stability class (TKE)')
               logger.info('Applying Correction Method: SS-WS-Std by stability class (TKE)')
               # stability subset output for primary height (all classes)
               ResultsLists_class = initialize_resultsLists('class_')
               className = 1
               for item in All_class_data:
                   inputdata_corr, lm_corr, m, c = perform_SS_WS_Std_correction(item[primary_idx].copy())
                   print("SS-WS-Std: y = " + str(m) + " * x + " + str(c))
                   lm_corr['sensor'] = sensor
                   lm_corr['height'] = height
                   lm_corr['correction'] = str('SS-WS-Std' + '_' + 'class_' + str(className))
                   correctionName = str('SS_WS_Std' + '_TKE_' + str(className))
                   ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                   className += 1
               ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
           if RSD_alphaFlag:
               print('Applying Correction Method: SS-WS-Std by stability class Alpha w/ RSD')
               logger.info('Applying Correction Method: SS-WS-Std by stability class Alpha w/ RSD')
               ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
               className = 1
               for item in All_class_data_alpha_RSD:
                   inputdata_corr, lm_corr, m, c = perform_SS_WS_Std_correction(item.copy())
                   print ("SS-WS-Std: y = " + str(m) + "* x +" + str(c))
                   lm_corr['sensor'] = sensor
                   lm_corr['height'] = height
                   lm_corr['correction'] = str('SS-WS-Std' + '_' + 'class_' + str(className))
                   correctionName = str('SS_WS_Std' + '_alphaRSD_' + str(className))
                   ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                        inputdata_corr, Timestamps, method)
                   className += 1
               ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
           if cup_alphaFlag:
               print('Applying correction Method: SS-WS-Std by stability class Alpha w/cup')
               logger.info('Applying correction Method: SS-WS-Std by stability class Alpha w/cup')
               ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
               className = 1
               for item in All_class_data_alpha_Ane:
                   inputdata_corr, lm_corr, m, c = perform_SS_WS_Std_correction(item.copy())
                   print ("SS-WS-Std: y = " + str(m) + "* x +" + str(c))
                   lm_corr['sensor'] = sensor
                   lm_corr['height'] = height
                   lm_corr['correction'] = str('SS-WS-Std' + '_' + 'class_' + str(className))
                   emptyclassFlag = False
                   correctionName = str('SS_WS_Std' + '_alphaCup_' + str(className))
                   ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                        inputdata_corr, Timestamps, method)
                   className += 1
               ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # **************************************************************** #
        # Site Specific LTERRA for WC 1HZ Data Correction (G-LTERRA_WC_1HZ)
        if method != 'SS-LTERRA-WC-1HZ':
            pass
        elif method == 'SS-LTERRA-WC-1HZ' and correctionsMetadata['SS-LTERRA-WC-1HZ'] == False:
            pass
        else:
           print('Applying Correction Method: SS-LTERRA-WC-1HZ')
           logger.info('Applying Correction Method: SS-LTERRA-WC-1HZ')


        # ******************************************************************* #
        # Site Specific LTERRA WC Machine Learning Correction (SS-LTERRA-MLa)
        # Random Forest Regression with now ancillary columns
        if method != 'SS-LTERRA-MLa':
            pass
        elif method == 'SS-LTERRA-MLa' and correctionsMetadata['SS-LTERRA-MLa'] == False:
            pass
        else:
            print('Applying Correction Method: SS-LTERRA-MLa')
            logger.info('Applying Correction Method: SS-LTERRA-MLa')

            inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_ML_correction(inputdata.copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS_LTERRA_MLa'
            correctionName = 'SS_LTERRA_MLa'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-LTERRA MLa by stability class (TKE)')
                logger.info('Applying Correction Method: SS-LTERRA MLa by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c= perform_SS_LTERRA_ML_correction(item[primary_idx].copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS_LTERRA_MLa' + '_' + 'class_' + str(className))
                    correctionName = str('SS_LTERRA_MLa' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
            if RSD_alphaFlag:
                print('Applying Correction Method: SS-LTERRA MLa by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-LTERRA MLa by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_ML_correction(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-LTERRA_MLa' + '_' + 'class_' + str(className))
                    correctionName = str('SS_LTERRA_ML' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
            if cup_alphaFlag:
                print('Applying correction Method: SS-LTERRA_MLa by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-LTERRA_MLa by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_ML_correction(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS_LTERRA_MLa' + '_' + 'class_' + str(className))
                    emptyclassFlag = False
                    correctionName = str('SS_LTERRA_MLa' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ************************************************************************************ #
        # Site Specific LTERRA WC (w/ stability) Machine Learning Correction (SS-LTERRA_MLc)
        if method != 'SS-LTERRA-MLc':
            pass
        elif method == 'SS-LTERRA-MLc' and correctionsMetadata['SS-LTERRA-MLc'] == False:
            pass
        else:
            print('Applying Correction Method: SS-LTERRA-MLc')
            logger.info('Applying Correction Method: SS-LTERRA-MLc')
            all_trainX_cols = ['x_train_TI', 'x_train_TKE','x_train_WS','x_train_DIR','x_train_Hour']
            all_trainY_cols = ['y_train']
            all_testX_cols = ['x_test_TI','x_test_TKE','x_test_WS','x_test_DIR','x_test_Hour']
            all_testY_cols = ['y_test']

            inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(inputdata.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS_LTERRA_MLc'
            correctionName = 'SS_LTERRA_MLc'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr, Timestamps, method)

            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-LTERRA_MLc by stability class (TKE)')
                logger.info('Applying Correction Method: SS-LTERRA_MLc by stability class (TKE)')

                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c= perform_SS_LTERRA_S_ML_correction(item[primary_idx].copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS_LTERRA_MLc' + '_' + 'class_' + str(className))
                    correctionName = str('SS_LTERRA_MLc' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
            if RSD_alphaFlag:
                print('Applying Correction Method: SS-LTERRA_MLc by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-LTERRA_MLc by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-LTERRA_MLc' + '_' + 'class_' + str(className))
                    correctionName = str('SS_LTERRA_S_ML' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
            if cup_alphaFlag:
                print('Applying correction Method: SS-LTERRA_MLc by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-LTERRA_MLc by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS_LTERRA_MLc' + '_' + 'class_' + str(className))
                    emptyclassFlag = False
                    correctionName = str('SS_LTERRA_MLc' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # *********************** #
        # Site Specific SS-LTERRA-MLb
        if method != 'SS-LTERRA-MLb':
            pass
        elif method == 'SS-LTERRA-MLb' and correctionsMetadata['SS-LTERRA-MLb'] == False:
            pass
        else:
            print('Applying Correction Method: SS-LTERRA-MLb')
            logger.info('Applying Correction Method: SS-LTERRA-MLb')
            all_trainX_cols = ['x_train_TI', 'x_train_TKE']
            all_trainY_cols = ['y_train']
            all_testX_cols = ['x_test_TI','x_test_TKE']
            all_testY_cols = ['y_test']

            inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(inputdata.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS_LTERRA_MLb'
            correctionName = 'SS_LTERRA_MLb'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr, Timestamps, method)

            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-LTERRA_MLb by stability class (TKE)')
                logger.info('Applying Correction Method: SS-LTERRA_MLb by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c= perform_SS_LTERRA_S_ML_correction(item[primary_idx].copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS_LTERRA_MLb' + '_' + 'class_' + str(className))
                    correctionName = str('SS_LTERRA_MLb' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')
            if RSD_alphaFlag:
                print('Applying Correction Method: SS-LTERRA_MLb by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-LTERRA_MLb by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-LTERRA_MLb' + '_' + 'class_' + str(className))
                    correctionName = str('SS_LTERRA_MLb' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
            if cup_alphaFlag:
                print('Applying correction Method: SS-LTERRA_MLb by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-LTERRA_MLb by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS_LTERRA_MLb' + '_' + 'class_' + str(className))
                    emptyclassFlag = False
                    correctionName = str('SS_LTERRA_MLb' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # *********************** #
        # TI Extrapolation (TI-Ext)
        if method != 'TI-Extrap':
            pass
        elif method == 'TI-Extrap' and correctionsMetadata['TI-Extrap'] == False:
            pass
        else:
            print ('Found enough data to perform extrapolation comparison')
            block_print()
            # Get extrapolation height
            height_extrap = float(extrap_metadata['height'][extrap_metadata['type'] == 'extrap'])
            # Extrapolate
            inputdata_corr, lm_corr, shearTimeseries= perform_TI_extrapolation(inputdata.copy(), extrap_metadata,
                                                                               extrapolation_type, height)
            correctionName = 'TI_EXTRAP'
            lm_corr['correction'] = correctionName

            inputdataEXTRAP = inputdata_corr.copy()
            inputdataEXTRAP, baseResultsLists = extrap_configResult(extrapolation_type, inputdataEXTRAP, baseResultsLists, method,lm_corr)

            if RSDtype['Selection'][0:4] == 'Wind':
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, shearTimeseries= perform_TI_extrapolation(item[primary_idx].copy(), extrap_metadata,
                                                                                       extrapolation_type, height)
                    lm_corr['correction'] = str('TI_EXT_class1' + '_TKE_' + 'class_' + str(className))
                    inputdataEXTRAP = inputdata_corr.copy()
                    inputdataEXTRAP, ResultsLists_class = extrap_configResult(extrapolation_type, inputdataEXTRAP, ResultsLists_class,
                                                                              method, lm_corr, appendString = 'class_')
                    className += 1

                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if cup_alphaFlag:
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, shearTimeseries= perform_TI_extrapolation(item.copy(), extrap_metadata,
                                                                                       extrapolation_type, height)
                    lm_corr['correction'] = str('TI_Ane_class1' + '_alphaCup_' + 'class_' + str(className))
                    inputdataEXTRAP = inputdata_corr.copy()
                    inputdataEXTRAP, ResultsLists_class_alpha_Ane = extrap_configResult(extrapolation_type, inputdataEXTRAP,
                                                                                        ResultsLists_class_alpha_Ane, method,
                                                                                        lm_corr, appendString = 'class_alpha_Ane')
                    className += 1

                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane,
                                                                                   ResultsLists_class_alpha_Ane, 'alpha_Ane')
            if RSD_alphaFlag:
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, shearTimeseries= perform_TI_extrapolation(item.copy(), extrap_metadata,
                                                                                       extrapolation_type, height)
                    lm_corr['correction'] = str('TI_RSD_class1' + '_alphaRSD_' + 'class_' + str(className))
                    inputdataEXTRAP = inputdata_corr.copy()
                    inputdataEXTRAP, ResultsLists_class_alpha_RSD = extrap_configResult(extrapolation_type, inputdataEXTRAP,
                                                                                        ResultsLists_class_alpha_RSD, method,
                                                                                        lm_corr, appendString = 'class_alpha_RSD')
                    className += 1

                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD,
                                                                                   ResultsLists_class_alpha_RSD, 'alpha_RSD')
            # Add extra info to meta data and reformat
            if extrapolation_type == 'simple':
                desc = 'No truth measurement at extrapolation height'
            else:
                desc = 'Truth measurement available at extrapolation height'
            extrap_metadata = (extrap_metadata
                              .append({'type': np.nan, 'height': np.nan, 'num': np.nan},
                                      ignore_index=True)
                              .append(pd.DataFrame([['extrapolation type', extrapolation_type, desc]],
                                                   columns=extrap_metadata.columns))
                              .rename(columns={'type': 'Type',
                                               'height': 'Height (m)',
                                               'num': 'Comparison Height Number'}))
            enable_print()

        # ************************************************** #
        # Histogram Matching
        if method != 'SS-Match':
            pass
        elif method == 'SS-Match' and correctionsMetadata['SS-Match'] == False:
            pass
        else:
            print('Applying Match algorithm: SS-Match')
            logger.info('Applying Match algorithm: SS-Match')
            inputdata_corr, lm_corr = perform_Match(inputdata.copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS-Match'
            correctionName = 'SS_Match'

            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-Match by stability class (TKE)')
                logger.info('Applying Correction Method: SS-Match by stability class (TKE)')
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr = perform_Match(item[primary_idx].copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-Match' + '_' + 'class_' + str(className))
                    correctionName = str('SS_Match' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: SS-Match by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-Match by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr = perform_Match(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-Match' + '_' + 'class_' + str(className))
                    correctionName = str('SS_Match' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: SS-Match by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-Match by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr = perform_Match(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-Match' + '_' + 'class_' + str(className))
                    correctionName = str('SS_Match' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ************************************************** #
        # Histogram Matching Input Corrected
        if method != 'SS-Match2':
            pass
        elif method == 'SS-Match2' and correctionsMetadata['SS-Match2'] == False:
            pass
        else:
            print('Applying input match algorithm: SS-Match2')
            logger.info('Applying input match algorithm: SS-Match2')
            inputdata_corr, lm_corr = perform_Match_input(inputdata.copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'SS-Match2'
            correctionName = 'SS_Match2'

            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: SS-Match2 by stability class (TKE)')
                logger.info('Applying Correction Method: SS-Match2 by stability class (TKE)')
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr = perform_Match_input(item[primary_idx].copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-Match2' + '_' + 'class_' + str(className))
                    correctionName = str('SS_Match2' + '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: SS-Match2 by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: SS-Match2 by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr = perform_Match_input(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-Match2' + '_' + 'class_' + str(className))
                    correctionName = str('SS_Match2' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: SS-Match2 by stability class Alpha w/cup')
                logger.info('Applying correction Method: SS-Match2 by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr = perform_Match_input(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('SS-Match2' + '_' + 'class_' + str(className))
                    correctionName = str('SS_Match2' + '_alphaCup_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')
        # ************************************************** #
        # Global Simple Phase II mean Linear Reressions (G-Sa) + project
        '''
            RSD_TI = .984993 * RSD_TI + .087916
        '''

        if method != 'G-Sa':
            pass
        elif method == 'G-Sa' and correctionsMetadata['G-Sa'] == False:
            pass
        else:
            print('Applying Correction Method: G-Sa')
            logger.info('Applying Correction Method: G-Sa')
            override = False
            inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(inputdata.copy(),override)
            print("G-Sa: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'G-Sa'
            correctionName = 'G_Sa'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: G-Sa by stability class (TKE)')
                logger.info('Applying Correction Method: G-Sa by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item[primary_idx].copy(),override)
                    print("G-Sa: y = " + str(m) + " * x + " + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-Sa' + '_TKE_' + 'class_' + str(className))
                    correctionName = str('G-Sa'+ '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: G-Sa by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: G-Sa by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
                    print ("G-Sa: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-Sa' + '_' + 'class_' + str(className))
                    correctionName = str('G-Sa' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: G-Sa by stability class Alpha w/cup')
                logger.info('Applying correction Method: G-Sa by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
                    print ("G-Sa: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-Sa' + '_alphaCup_' + 'class_' + str(className))
                    correctionName = str('G-Sa' + '_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ******************************************************** #
        # Global Simple w/filter Phase II Linear Regressions (G-SFa) + project
        # Check these values, but for WC m = 0.7086 and c = 0.0225
        if method != 'G-SFa':
            pass
        elif method == 'G-SFa' and correctionsMetadata['G-SFa'] == False:
            pass
        elif RSDtype['Selection'][0:4] != 'Wind':
            pass
        else:
            print('Applying Correction Method: G-SFa')
            logger.info('Applying Correction Method: G-SFa')
            override = [0.7086, 0.0225]
            inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(inputdata.copy(),override)
            print("G-SFa: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'G-SFa'
            correctionName = 'G_SFa'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: G-SFa by stability class (TKE)')
                logger.info('Applying Correction Method: G-SFa by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item[primary_idx].copy(),override)
                    print("G-SFa: y = " + str(m) + " * x + " + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-SFa' + '_TKE_' + 'class_' + str(className))
                    correctionName = str('G-SFa'+ '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: G-SFa by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: G-SFa by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
                    print ("G-SFa: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-Sa' + '_' + 'class_' + str(className))
                    correctionName = str('G-SFa' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: G-SFa by stability class Alpha w/cup')
                logger.info('Applying correction Method: G-SFa by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
                    print ("G-SFa: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-SFa' + '_alphaCup_' + 'class_' + str(className))
                    correctionName = str('G-SFa' + '_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

        # ************************************************ #
        # Global Standard Deviation and WS correction (G-Sc)
        if method != 'G-SFc':
            pass
        elif method == 'G-SFc' and correctionsMetadata['G-SFc'] == False:
            pass
        elif RSDtype['Selection'][0:4] != 'Wind':
            pass
        else:
            print('Applying Correction Method: G-Sc')
            logger.info('Applying Correction Method: G-Sc')
            inputdata_corr, lm_corr, m, c = perform_G_SFc_correction(inputdata.copy())
            print("G-SFc: y = " + str(m) + " * x + " + str(c))
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'G-SFc'
            correctionName = 'G_SFc'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: G-SFa by stability class (TKE)')
                logger.info('Applying Correction Method: G-SFa by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    inputdata_corr, lm_corr, m, c = perform_G_SFc_correction(item[primary_idx].copy())
                    print("G-SFc: y = " + str(m) + " * x + " + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-SFc' + '_TKE_' + 'class_' + str(className))
                    correctionName = str('G-SFc'+ '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: G-SFc by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: G-SFc by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    inputdata_corr, lm_corr, m, c = perform_G_SFc_correction(item.copy())
                    print ("G-SFc: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-SFc' + '_' + 'class_' + str(className))
                    correctionName = str('G-SFc' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

            if cup_alphaFlag:
                print('Applying correction Method: G-SFc by stability class Alpha w/cup')
                logger.info('Applying correction Method: G-SFc by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    inputdata_corr, lm_corr, m, c = perform_G_SFc_correction(item.copy())
                    print ("G-SFc: y = " + str(m) + "* x +" + str(c))
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-SFc' + '_alphaCup_' + 'class_' + str(className))
                    correctionName = str('G-SFc' + '_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')


        # ************************ #
        # Global Comprehensive (G-C)
        '''
        based on empirical calibrations by EON
        '''
        if method != 'G-C':
            pass
        elif method == 'G-C' and correctionsMetadata['G-C'] == False:
            pass
        else:
            print('Applying Correction Method: G-C')
            logger.info('Applying Correction Method: G-C')
            inputdata_corr, lm_corr, m, c = perform_G_C_correction(inputdata.copy())
            lm_corr['sensor'] = sensor
            lm_corr['height'] = height
            lm_corr['correction'] = 'G-C'
            correctionName = 'G_C'
            baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                     Timestamps, method)
            TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

            if RSDtype['Selection'][0:4] == 'Wind':
                print('Applying Correction Method: G-C by stability class (TKE)')
                logger.info('Applying Correction Method: G-C by stability class (TKE)')
                # stability subset output for primary height (all classes)
                ResultsLists_class = initialize_resultsLists('class_')
                className = 1
                for item in All_class_data:
                    print (str('class ' + str(className)))
                    inputdata_corr, lm_corr, m, c = perform_G_C_correction(item[primary_idx].copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-C' + '_TKE_' + 'class_' + str(className))
                    correctionName = str('G-C'+ '_TKE_' + str(className))
                    ResultsLists_class = populate_resultsLists(ResultsLists_class, 'class_', correctionName, lm_corr,
                                                               inputdata_corr, Timestamps, method)
                    className += 1
                ResultsList_stability = populate_resultsLists_stability(ResultsLists_stability, ResultsLists_class, '')

            if RSD_alphaFlag:
                print('Applying Correction Method: G-C by stability class Alpha w/ RSD')
                logger.info('Applying Correction Method: G-C by stability class Alpha w/ RSD')
                ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
                className = 1
                for item in All_class_data_alpha_RSD:
                    print (str('class ' + str(className)))
                    inputdata_corr, lm_corr, m, c = perform_G_C_correction(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-C' + '_' + 'class_' + str(className))
                    correctionName = str('G-C' + '_alphaRSD_' + str(className))
                    ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')


            if cup_alphaFlag:
                print('Applying correction Method: G-C by stability class Alpha w/cup')
                logger.info('Applying correction Method: G-C by stability class Alpha w/cup')
                ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
                className = 1
                for item in All_class_data_alpha_Ane:
                    print (str('class ' + str(className)))
                    inputdata_corr, lm_corr, m, c = perform_G_C_correction(item.copy())
                    lm_corr['sensor'] = sensor
                    lm_corr['height'] = height
                    lm_corr['correction'] = str('G-C' + '_alphaCup_' + 'class_' + str(className))
                    correctionName = str('G-C' + '_' + str(className))
                    ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                         inputdata_corr, Timestamps, method)
                    className += 1
                ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')


        # ************************ #
        # Global Comprehensive (G-Match)
        if method != 'G-Match':
            pass
        elif method == 'G-Match' and correctionsMetadata['G-Match'] == False:
            pass
        else:
            print('Applying Correction Method: G-Match')
            logger.info('Applying Correction Method: G-Match')

        # ************************ #
        # Global Comprehensive (G-Ref-S)
        if method != 'G-Ref-S':
            pass
        elif method == 'G-Ref-S' and correctionsMetadata['G-Ref-S'] == False:
            pass
        else:
            print('Applying Correction Method: G-Ref-S')
            logger.info('Applying Correction Method: G-Ref-S')

        # ************************ #
        # Global Comprehensive (G-Ref-Sf)
        if method != 'G-Ref-Sf':
            pass
        elif method == 'G-Ref-Sf' and correctionsMetadata['G-Ref-Sf'] == False:
            pass
        else:
            print('Applying Correction Method: G-Ref-Sf')
            logger.info('Applying Correction Method: G-Ref-Sf')

        # ************************ #
        # Global Comprehensive (G-Ref-SS)
        if method != 'G-Ref-SS':
            pass
        elif method == 'G-Ref-SS' and correctionsMetadata['G-Ref-SS'] == False:
            pass
        else:
            print('Applying Correction Method: G-Ref-SS')
            logger.info('Applying Correction Method: G-Ref-SS')
        # ************************ #
        # Global Comprehensive (G-Ref-SS-S)
        if method != 'G-Ref-SS-S':
            pass
        elif method == 'G-Ref-SS-S' and correctionsMetadata['G-Ref-SS-S'] == False:
            pass
        else:
            print('Applying Correction Method: G-Ref-SS-S')
            logger.info('Applying Correction Method: G-Ref-SS-S')
        # ************************ #
        # Global Comprehensive (G-Ref-WS-Std)
        if method != 'G-Ref-WS-Std':
            pass
        elif method == 'G-Ref-WS-Std' and correctionsMetadata['G-Ref-WS-Std'] == False:
            pass
        else:
            print('Applying Correction Method: G-Ref-WS-Std')
            logger.info('Applying Correction Method: G-Ref-WS-Std')

        # ***************************************** #
        # Global LTERRA WC 1Hz Data (G-LTERRA_WC_1Hz)
        if method != 'G-LTERRA_WC_1Hz':
            pass
        elif method == 'G-LTERRA_WC_1Hz' and correctionsMetadata['G-LTERRA_WC_1Hz'] == False:
            pass
        else:
            print('Applying Correction Method: G-LTERRA_WC_1Hz')
            logger.info('Applying Correction Method: G-LTERRA_WC_1Hz')

        # ************************************************ #
        # Global LTERRA ZX Machine Learning (G-LTERRA_ZX_ML)
        if method != 'G-LTERRA_ZX_ML':
            pass
        elif correctionsMetadata['G-LTERRA_ZX_ML'] == False:
            pass
        else:
            print('Applying Correction Method: G-LTERRA_ZX_ML')
            logger.info('Applying Correction Method: G-LTERRA_ZX_ML')

        # ************************************************ #
        # Global LTERRA WC Machine Learning (G-LTERRA_WC_ML)
        if method != 'G-LTERRA_WC_ML':
            pass
        elif correctionsMetadata['G-LTERRA_WC_ML'] == False:
            pass
        else:
            print('Applying Correction Method: G-LTERRA_WC_ML')
            logger.info('Applying Correction Method: G-LTERRA_WC_ML')

        # ************************************************** #
        # Global LTERRA WC w/Stability 1Hz (G-LTERRA_WC_S_1Hz)
        if method != 'G-LTERRA_WC_S_1Hz':
            pass
        elif method == 'G-LTERRA_WC_S_1Hz' and correctionsMetadata['G-LTERRA_WC_S_1Hz'] == False:
            pass
        else:
            print('Applying Correction Method: G-LTERRA_WC_S_1Hz')
            logger.info('Applying Correction Method: G-LTERRA_WC_S_1Hz')

        # ************************************************************** #
        # Global LTERRA WC w/Stability Machine Learning (G-LTERRA_WC_S_ML)
        if method != 'G-LTERRA_WC_S_ML':
            pass
        elif method == 'G-LTERRA_WC_S_ML' and correctionsMetadata['G-LTERRA_WC_S_ML'] == False:
            pass
        else:
            print('Applying Correction Method: G-LTERRA_WC_S_ML')
            logger.info('Applying Correction Method: G-LTERRA_WC_S_ML')

    if RSD_alphaFlag:
        pass
    else:
        ResultsLists_stability_alpha_RSD = ResultsList_stability

    if cup_alphaFlag:
        pass
    else:
        ResultsLists_stability_alpha_Ane = ResultsList_stability

    if RSDtype['Selection'][0:4] != 'Wind':
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
        lm_CorrList_stability = np.nan
        correctionTagList_stability = np.nan
        Distibution_statsList_stability = np.nan
        sampleTestsLists_stability = np.nan

    # Write 10 minute Adjusted data to a csv file
    outpath_dir = os.path.dirname(results_filename)
    outpath_file = os.path.basename(results_filename)
    outpath_file = str('TI_10minuteAdjusted_' + outpath_file.split('.xlsx')[0] + '.csv')
    out_dir = os.path.join(outpath_dir,outpath_file)

    TI_10minuteAdjusted.to_csv(out_dir)

    write_all_resultstofile(reg_results, baseResultsLists, count_1mps, count_05mps, count_1mps_train, count_05mps_train,
                            count_1mps_test, count_05mps_test, name_1mps_tke, name_1mps_alpha_Ane, name_1mps_alpha_RSD,
                            name_05mps_tke, name_05mps_alpha_Ane, name_05mps_alpha_RSD, count_05mps_tke, count_05mps_alpha_Ane, count_05mps_alpha_RSD,
                            count_1mps_tke, count_1mps_alpha_Ane, count_1mps_alpha_RSD,results_filename, siteMetadata, filterMetadata,
                            Timestamps,timestamp_train,timestamp_test,regimeBreakdown_tke, regimeBreakdown_ane, regimeBreakdown_rsd,
                            Ht_1_ane, Ht_2_ane, extrap_metadata, reg_results_class1, reg_results_class2, reg_results_class3,
                            reg_results_class4, reg_results_class5,reg_results_class1_alpha, reg_results_class2_alpha, reg_results_class3_alpha,
                            reg_results_class4_alpha, reg_results_class5_alpha, Ht_1_rsd, Ht_2_rsd, ResultsLists_stability, ResultsLists_stability_alpha_RSD,
                            ResultsLists_stability_alpha_Ane, stabilityFlag, cup_alphaFlag, RSD_alphaFlag, TimeTestA_baseline_df, TimeTestB_baseline_df,
                            TimeTestC_baseline_df,TimeTestA_corrections_df,TimeTestB_corrections_df,TimeTestC_corrections_df)
