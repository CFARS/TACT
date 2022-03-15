# -*- coding: utf-8 -*-
"""
TACT Adjustments

"""
import os
import datetime as dt
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.ioff()  # setting to non-interactive
import seaborn as sns
from dateutil import parser
import numpy as np
from pathlib import Path
import sys
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

class Adjustments():

    """
    document parameters
    """

    def __init__(self, raw_data, adjustments_list,baseResultsLists):
        self.raw_data = raw_data
        self.adjusted_data =pd.DataFrame()
        self.results_stats = [] # make this a dictionary of results with adjustment_list items as keys
        #self.Adjustments queue
        
    def get_regression(self, x, y):
        '''
        Compute linear regression of data -> need to deprecate this function for get_modelRegression..
        '''
        df = pd.DataFrame()
        df['x'] = x
        df['y'] = y
        df = df.dropna()
        if len(df) > 1:
            x = df['x']
            y = df['y']
            x = x.astype(float)
            y = y.astype(float)
            lm = linear_model.LinearRegression()
            lm.fit(x.to_frame(), y.to_frame())
            result = [lm.coef_[0][0], lm.intercept_[0]]         #slope and intercept?
            result.append(lm.score(x.to_frame(), y.to_frame())) #r score?
            result.append(abs((x - y).mean()))                  # mean diff?
            x = x.to_numpy().reshape(len(x), 1)
            y = y.to_numpy().reshape(len(y), 1)
            predict = lm.predict(x)
            mse = mean_squared_error(y, predict, multioutput='raw_values')
            rmse = np.sqrt(mse)
            result.append(mse[0])
            result.append(rmse[0])
        else:
            result = [None, None, None, None, None, None]
        # results order: m, c, r2, mean difference, mse, rmse

        return result

    def post_correction_stats(self,inputdata,results,ref_col,TI_col):

        if isinstance(inputdata, pd.DataFrame):
            fillEmpty = False
            if ref_col in inputdata.columns and TI_col in inputdata.columns:
                model_corrTI = self.get_regression(inputdata[ref_col], inputdata[TI_col])
                name1 = 'TI_regression_' + TI_col + '_' + ref_col
                results.loc[name1, ['m']] = model_corrTI[0]
                results.loc[name1, ['c']] = model_corrTI[1]
                results.loc[name1, ['rsquared']] = model_corrTI[2]
                results.loc[name1, ['difference']] = model_corrTI[3]
                results.loc[name1, ['mse']] = model_corrTI[4]
                results.loc[name1, ['rmse']] = model_corrTI[5]
            else:
                fillEmpty = True
        else:
            fillEmpty = True
        if fillEmpty:
            name1 = 'TI_regression_' + TI_col + '_' + ref_col
            results.loc[name1, ['m']] = 'NaN'
            results.loc[name1, ['c']] = 'NaN'
            results.loc[name1, ['rsquared']] = 'NaN'
            results.loc[name1, ['difference']] = 'NaN'
            results.loc[name1, ['mse']] = 'NaN'
            results.loc[name1, ['rmse']] = 'NaN'
        return results
    
    def perform_SS_S_correction(self,inputdata):
        '''
        Note: Representative TI computed with original RSD_SD
        '''
        results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
        inputdata_train = inputdata[inputdata['split'] == True].copy()
        inputdata_test = inputdata[inputdata['split'] == False].copy()

        if inputdata.empty or len(inputdata) < 2:
            results = self.post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
            if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
            if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
            if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
            m = np.NaN
            c = np.NaN
            inputdata = False
        else:
            full = pd.DataFrame()
            full['Ref_TI'] = inputdata_test['Ref_TI']
            full['RSD_TI'] = inputdata_test['RSD_TI']
            full = full.dropna()
            if len(full) < 2:
                results = self.post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
                m = np.NaN
                c = np.NaN
            else:
                model = self.get_regression(inputdata_train['RSD_TI'], inputdata_train['Ref_TI'])
                m = model[0]
                c = model[1]
                RSD_TI = inputdata_test['RSD_TI'].copy()
                RSD_TI = (model[0]*RSD_TI) + model[1]
                inputdata_test['corrTI_RSD_TI'] = RSD_TI
                results = self.post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
            if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
                full = pd.DataFrame()
                full['Ref_TI'] = inputdata_test['Ane_TI_Ht1']
                full['RSD_TI'] = inputdata_test['RSD_TI_Ht1']
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(inputdata_train['RSD_TI'], inputdata_train['Ref_TI'])
                    RSD_TI = inputdata_test['RSD_TI_Ht1'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht1'] = RSD_TI
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
            if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
                full = pd.DataFrame()
                full['Ref_TI'] = inputdata_test['Ane_TI_Ht2']
                full['RSD_TI'] = inputdata_test['RSD_TI_Ht2']
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(inputdata_train['RSD_TI_Ht2'],inputdata_train['Ane_TI_Ht2'])
                    RSD_TI = inputdata_test['RSD_TI_Ht2'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht2'] = RSD_TI
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
            if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
                full = pd.DataFrame()
                full['Ref_TI'] = inputdata_test['Ane_TI_Ht3']
                full['RSD_TI'] = inputdata_test['RSD_TI_Ht3']
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(inputdata_train['RSD_TI_Ht3'], inputdata_train['Ane_TI_Ht3'])
                    RSD_TI = inputdata_test['RSD_TI_Ht3'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht3'] = RSD_TI
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
            if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
                full = pd.DataFrame()
                full['Ref_TI'] = inputdata_test['Ane_TI_Ht4']
                full['RSD_TI'] = inputdata_test['RSD_TI_Ht4']
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                    m = np.NaN
                    c = np.NaN
                else:
                    model = self.get_regression(inputdata_train['RSD_TI_Ht4'], inputdata_train['Ane_TI_Ht4'])
                    RSD_TI = inputdata_test['RSD_TI_Ht4'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht4'] = RSD_TI
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

        results['correction'] = ['SS-S'] * len(results)
        results = results.drop(columns=['sensor','height'])
    
        return inputdata_test, results, m, c

    def perform_SS_SF_correction(self,inputdata):

        results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                    'c', 'rsquared', 'difference','mse', 'rmse'])
        inputdata_train = inputdata[inputdata['split'] == True].copy()
        inputdata_test = inputdata[inputdata['split'] == False].copy()

        if inputdata.empty or len(inputdata) < 2:
            results = self.post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI')
            if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
            if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
            if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
            if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
            m = np.NaN
            c = np.NaN
            inputdata = False
        else:
            filtered_Ref_TI = inputdata_train['Ref_TI'][inputdata_train['RSD_TI'] < 0.3]
            filtered_RSD_TI = inputdata_train['RSD_TI'][inputdata_train['RSD_TI'] < 0.3]
            full = pd.DataFrame()
            full['filt_Ref_TI'] = filtered_Ref_TI
            full['filt_RSD_TI'] = filtered_RSD_TI
            full = full.dropna()
            if len(full) < 2:
                results = self.post_correction_stats([None],results, 'Ref_TI','corrTI_RSD_TI',)
                m = np.NaN
                c = np.NaN
            else:
                model = self.get_regression(filtered_RSD_TI,filtered_Ref_TI)
                m = model[0]
                c = model[1]
                RSD_TI = inputdata_test['RSD_TI'].copy()
                RSD_TI = (float(model[0])*RSD_TI) + float(model[1])
                inputdata_test['corrTI_RSD_TI'] = RSD_TI
                inputdata_test['corrRepTI_RSD_RepTI'] = RSD_TI + 1.28 * inputdata_test['RSD_SD']
                results = self.post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')
            if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
                filtered_Ref_TI = inputdata_train['Ane_TI_Ht1'][inputdata_train['Ane_TI_Ht1'] < 0.3]
                filtered_RSD_TI = inputdata_train['RSD_TI_Ht1'][inputdata_train['RSD_TI_Ht1'] < 0.3]
                full = pd.DataFrame()
                full['filt_Ref_TI'] = filtered_Ref_TI
                full['filt_RSD_TI'] = filtered_RSD_TI
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                else:
                    model = self.get_regression(filtered_RSD_TI,filtered_Ref_TI)
                    RSD_TI = inputdata_test['RSD_TI_Ht1'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht1'] = RSD_TI
                    inputdata_test['corrRepTI_RSD_RepTI_Ht1'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht1']
                    results = self.post_correction_stats(inputdata,results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
            if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
                filtered_Ref_TI = inputdata_train['Ane_TI_Ht2'][inputdata_train['Ane_TI_Ht2'] < 0.3]
                filtered_RSD_TI = inputdata_train['RSD_TI_Ht2'][inputdata_train['RSD_TI_Ht2'] < 0.3]
                full = pd.DataFrame()
                full['filt_Ref_TI'] = filtered_Ref_TI
                full['filt_RSD_TI'] = filtered_RSD_TI
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                else:
                    model = self.get_regression(filtered_RSD_TI,filtered_Ref_TI)
                    RSD_TI = inputdata_test['RSD_TI_Ht2'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht2'] = RSD_TI
                    inputdata_test['corrRepTI_RSD_RepTI_Ht2'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht2']
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
            if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
                filtered_Ref_TI = inputdata_train['Ane_TI_Ht3'][inputdata_train['Ane_TI_Ht3'] < 0.3]
                filtered_RSD_TI = inputdata_train['RSD_TI_Ht3'][inputdata_train['RSD_TI_Ht3'] < 0.3]
                full = pd.DataFrame()
                full['filt_Ref_TI'] = filtered_Ref_TI
                full['filt_RSD_TI'] = filtered_RSD_TI
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                else:
                    model = self.get_regression(filtered_RSD_TI,filtered_Ref_TI)
                    RSD_TI = inputdata_test['RSD_TI_Ht3'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht3'] = RSD_TI
                    inputdata_test['corrRepTI_RSD_RepTI_Ht3'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht3']
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
            if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
                filtered_Ref_TI = inputdata_train['Ane_TI_Ht4'][inputdata_train['Ane_TI_Ht4'] < 0.3]
                filtered_RSD_TI = inputdata_train['RSD_TI_Ht4'][inputdata_train['RSD_TI_Ht4'] < 0.3]
                full = pd.DataFrame()
                full['filt_Ref_TI'] = filtered_Ref_TI
                full['filt_RSD_TI'] = filtered_RSD_TI
                full = full.dropna()
                if len(full) < 2:
                    results = self.post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                else:
                    model = self.get_regression(filtered_RSD_TI,filtered_Ref_TI)
                    RSD_TI = inputdata_test['RSD_TI_Ht4'].copy()
                    RSD_TI = (model[0]*RSD_TI) + model[1]
                    inputdata_test['corrTI_RSD_TI_Ht4'] = RSD_TI
                    inputdata_test['corrRepTI_RSD_RepTI_Ht4'] = RSD_TI + 1.28 * inputdata_test['RSD_SD_Ht4']
                    results = self.post_correction_stats(inputdata_test,results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')

        results['correction'] = ['SS-SF'] * len(results)
        results = results.drop(columns=['sensor','height'])
        return inputdata_test, results, m, c

    
