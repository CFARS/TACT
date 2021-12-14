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

class Adjustments():

    """
    document parameters
    """

    def __init__(self, adjustments):
        self.raw_data = raw_data
        

    def execute_GC_method(self):

        '''
        Note: comprehensive empirical correction from a dozen locations. Focuses on std. deviation
        '''
        results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm','c', 'rsquared', 'difference','mse', 'rmse'])

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
    
        return input_data, adjusted_data, results, m, c

    def method(self):
        
        '''
        simple site specific correction, but adjust each TKE class differently
        '''
        results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                        'c', 'rsquared', 'difference','mse', 'rmse'])
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
                        model = get_regression(inputdata_train['RSD_TI'], inputdata_train['Ref_TI'])
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
    
        return input_data, adjusted_data, results, m, c
    
    def method(self):

        return data_adjusted

       
         

