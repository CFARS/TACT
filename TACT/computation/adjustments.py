# -*- coding: utf-8 -*-
"""
TACT Adjustments

"""
try:
    from TACT import logger
except ImportError:
    pass
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.ioff()  # setting to non-interactive
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class Adjustments():

    """
    document parameters
    """

    def __init__(self, height, raw_data='', adjustments_list=''):
        self.raw_data = raw_data
        self.adjusted_data =pd.DataFrame()
        self.height = height # primary height of rsd and ane comparison
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

        feature_name = "x"
        target_name = "y"

        data, target = df[[feature_name]], df[target_name]

        if len(df) > 1:

            x = df['x'].astype(float)
            y = df['y'].astype(float)
 
            lm = LinearRegression()
            lm.fit(data, target)
            predict = lm.predict(data)

            result = [lm.coef_[0], lm.intercept_]         #slope and intercept?
            result.append(lm.score(data, target))         #r score?
            result.append(abs((x - y).mean()))            # mean diff?

            mse = mean_squared_error(target, predict, multioutput='raw_values')
            rmse = np.sqrt(mse)
            result.append(mse[0])
            result.append(rmse[0])

        else:
            result = [None, None, None, None, None, None]
            result = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        # results order: m, c, r2, mean difference, mse, rmse

        # logger.debug(result)

        return result

    def post_correction_stats(self, inputdata, results, ref_col, TI_col):

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
    
    def perform_SS_S_correction(self, inputdata):
        '''
        Site-specific method adjusting TI data based on resgressions slope and offset adjustments
        Note: Representative TI computed with original RSD_SD ---> delete this
        '''
        results = pd.DataFrame(columns=['sensor', 'height', 'correction', 'm',
                                        'c', 'rsquared', 'difference','mse', 'rmse'])
        
        # rely on test-train split so that method is not tested on the data that the
        #      model was generated from 
        inputdata_train = inputdata[inputdata['split'] == True].copy()
        inputdata_test = inputdata[inputdata['split'] == False].copy()

        if inputdata.empty or len(inputdata) < 2: # if there is a problem with input dataframe
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
        else: # isolate timestamps with data
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
                    model = self.get_regression(inputdata_train['RSD_TI_Ht1'], inputdata_train['Ane_TI_Ht1'])
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

    def perform_SS_SF_correction(self, inputdata):

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
                results = self.post_correction_stats([None], results, 'Ref_TI','corrTI_RSD_TI',)
                m = np.NaN
                c = np.NaN
            else:
                model = self.get_regression(filtered_RSD_TI, filtered_Ref_TI)
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

    def perform_ZX_correction(self, inputdata):
    

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
            m_h = 0.2073*self.height + 13.294
            c_h = 0.4707*self.height + 20.041
            m = None
            c = None
            
            temp = pd.DataFrame()
            temp['RSD_TI'] = inputdata_test['RSD_TI'].astype(float)
            temp['corrTI_RSD_TI'] = temp['RSD_TI'].astype(float)*(1-((m_h * np.log(temp['RSD_TI'].astype(float)))/100) - (c_h/100))
            temp.loc[temp['RSD_TI'] > 0.2, 'corrTI_RSD_TI'] = temp['RSD_TI']*(1-((m_h * np.log(0.2))/100) - (c_h/100))

            inputdata_test['corrTI_RSD_TI'] = temp['corrTI_RSD_TI']
            inputdata_test['corrRepTI_RSD_RepTI'] = inputdata_test['RSD_TI'] + 1.28 * inputdata_test['RSD_SD']
            results = self.post_correction_stats(inputdata_test,results, 'Ref_TI','corrTI_RSD_TI')

            if 'Ane_TI_Ht1' in inputdata.columns and 'RSD_TI_Ht1' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht1','corrTI_RSD_TI_Ht1')
                # can complete this when we have variables for diff RSD heights                                                                    

            if 'Ane_TI_Ht2' in inputdata.columns and 'RSD_TI_Ht2' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht2','corrTI_RSD_TI_Ht2')
                # can complete this when we have variables for diff RSD heights                                                                    


            if 'Ane_TI_Ht3' in inputdata.columns and 'RSD_TI_Ht3' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht3','corrTI_RSD_TI_Ht3')
                # can complete this when we have variables for diff RSD heights                                                                    

            if 'Ane_TI_Ht4' in inputdata.columns and 'RSD_TI_Ht4' in inputdata.columns:
                results = self.post_correction_stats([None],results, 'Ane_TI_Ht4','corrTI_RSD_TI_Ht4')
                # can complete this when we have variables for diff RSD heights                                                                    

        results['correction'] = ['Zx'] * len(results)
        results = results.drop(columns=['sensor','height'])

        return inputdata_test, results, m, c

    
