try:
    from TACT import logger
except ImportError:
    pass
from TACT.readers.config import Config
import argparse
from future.utils import itervalues, iteritems
import os
import pandas as pd
import re
import sys
from string import printable
import numpy as np

class Data(Config):

    def read_timeseries(self):
        
        mm_filename=self.met_mast_timeseries_filepath
        self.input_met_mast_timeseries = pd.read_csv(self.met_mast_timeseries_filepath)
        
        self.input_rsd_timeseries= pd.read_csv(self.rsd_timeseries_filepath)
  
        
        

    def format_timeseries(self):
        
        ###########################
        #Format Mast Timeseries
        ###########################
        
        self.name_array=['Ane_1','Ane_2','Ane_3','Ane_4','Ane_5','Ane_6']
        self.std_name_array=['Ane_1_SD','Ane_2_SD','Ane_3_SD','Ane_4_SD','Ane_5_SD','Ane_6_SD']
        self.TI_name_array=['Ane_1_TI','Ane_2_TI','Ane_3_TI','Ane_4_TI','Ane_5_TI','Ane_6_TI']
        
        #Change wspd name
        for columns in self.input_met_mast_timeseries:
            for idx,names in enumerate(self.ane_array):
                if columns==names:
                    self.input_met_mast_timeseries.rename({names:self.name_array[idx]}, inplace=True,axis='columns')

        #Change sd name
        for columns in self.input_met_mast_timeseries:
            for idx_2,points in enumerate(self.get_std_dev(self.mast_data_model)):
                if columns==points:
                    self.input_met_mast_timeseries.rename({points:self.std_name_array[idx_2]}, inplace=True,axis='columns')
         
        #Get TI
        for columns in self.input_met_mast_timeseries:
            for idx_3,points in enumerate(self.get_TI(self.mast_data_model)):
                if columns==points:
                    self.input_met_mast_timeseries.rename({points:self.TI_name_array[idx_3]}, inplace=True,axis='columns')
                    
        ###########################            
        #Format RSD Timeseries
        ###########################
        
        self.rsd_name_array=['RSD_1','RSD_2','RSD_3','RSD_4','RSD_5','RSD_6']
        self.rsd_std_name_array=['RSD_1_SD','RSD_2_SD','RSD_3_SD','RSD_4_SD','RSD_5_SD','RSD_6_SD']
        self.rsd_TI_name_array=['RSD_1_TI','RSD_2_TI','RSD_3_TI','RSD_4_TI','RSD_5_TI','RSD_6_TI']
        
        #Change wspd name
        for columns in self.input_rsd_timeseries:
            for idx,names in enumerate(self.rsd_array):
                if columns==names:
                    self.input_rsd_timeseries.rename({names:self.rsd_name_array[idx]}, inplace=True,axis='columns')

        #Change sd name
        for columns in self.input_rsd_timeseries:
            for idx_2,points in enumerate(self.get_std_dev(self.rsd_data_model)):
                if columns==points:
                    self.input_rsd_timeseries.rename({points:self.rsd_std_name_array[idx_2]}, inplace=True,axis='columns')
         
        #Get TI
        for columns in self.input_rsd_timeseries:
            for idx_3,points in enumerate(self.get_TI(self.rsd_data_model)):
                if columns==points:
                    self.input_rsd_timeseries.rename({points:self.rsd_TI_name_array[idx_3]}, inplace=True,axis='columns')
                    
    
    def calculate_TI(self):
        
        def calc(wspd, wspd_std):
            wspd = _convert_df_to_series(wspd).dropna()
            wspd_std = _convert_df_to_series(wspd_std).dropna()
            ti = pd.concat([wspd[wspd > 3].rename('wspd'), wspd_std.rename('wspd_std')], axis=1, join='inner')
            return ti['wspd_std'] / ti['wspd']
        
        def _convert_df_to_series(df):
            """
            Convert a pd.DataFrame to a pd.Series.
            If more than 1 column is in the DataFrame then it will raise a TypeError.
            If the sent argument is not a DataFrame it will return itself.
            :param df:
            :return:
            """
            if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
                return df.iloc[:, 0]
            elif isinstance(df, pd.DataFrame) and df.shape[1] > 1:
                raise TypeError('DataFrame cannot be converted to a Series as it contains more than 1 column.')
            return df
    
        turbulence=[]
        for position in range(0,len(self.name_array)):
            ti=calc(self.input_met_mast_timeseries[self.name_array[position]],self.input_met_mast_timeseries[self.std_name_array[position]])
            turbulence.append(ti)
        
        self.t=turbulence
        ti_name=['Ane_1_TI','Ane_2_TI','Ane_3_TI','Ane_4_TI','Ane_5_TI','Ane_6_TI']
        for idx, ti in enumerate(self.t):
            self.input_met_mast_timeseries[ti_name[idx]]=ti
            
        
    
    
                
       
            
            
        