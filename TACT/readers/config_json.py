try:
    from TACT import logger
except ImportError:
    pass
import argparse
from future.utils import itervalues, iteritems
import os
import pandas as pd
import re
import sys

class Config(object):
    def __init__(self, json_filename=''):
        logger.info(f"initiated Config object")
        
        self.json_filename=json_filename

    def read_config(self):
        logger.info(f"parsing arguments from commandline call")
        
        #Get the config filepath from the command window
        parser = argparse.ArgumentParser()
        parser.add_argument("-json","--input_json_filename", help="print this requires the filename for the input json")
        args = parser.parse_args()
        
        logger.info('input data file: {}'.format(args.json_filename))
        print('the input data file is {}'.format(args.json_filename))
        
        #Read JSON file
        self.json_filename = args.json_filename
        json_open= open(self.json_filename)
        
        #Read timeseries filepaths from the json
        self.met_mast_timeseries_filepath=json_open["Met Mast"]["Timeseries_Data_Filepath"]
        if json_open["Met Mast"]["Timeseries_Data_Filepath"] is False:
            raise('No met mast timeseries availible')

        self.rsd_timeseries_filepath=json_open["RSD"]["Timeseries_Data_Filepath"]
        if json_open["RSD"]["Timeseries_Data_Filepath"] is False:
            raise('No RSD timeseries raised')
        ##data model filepath
        
        
        #Read mast column names from the JSON
        self.ane_1=json_open['Met Mast']['Ane_1']
        self.ane_2=json_open['Met Mast']['Ane_2']
        self.ane_3=json_open['Met Mast']['Ane_3']
        self.ane_4=json_open['Met Mast']['Ane_4']
        self.ane_5=json_open['Met Mast']['Ane_5']
        self.ane_6=json_open['Met Mast']['Ane_6']
        
        #Read the RSD column names from the JSON
        self.rsd_1=json_open['RSD']['RSD_1']
        self.rsd_2=json_open['RSD']['RSD_2']
        self.rsd_3=json_open['RSD']['RSD_3']
        self.rsd_4=json_open['RSD']['RSD_4']
        self.rsd_5=json_open['RSD']['RSD_5']
        self.rsd_6=json_open['RSD']['RSD_6']
        
        json_open.close()
        
    def read_data_model(self, model_filepath):
        json_open= open(model_filepath)
        if json_open['measurement_location']['measurement_station_type_id']=='lidar':
            self.rsd_data_model=json_open
        
        if json_open['measurement_location']['measurement_station_type_id']=='mast':
            self.mast_data_model=json_open
        json_open.close()
                        
    def get_heights(self, data_model):
        if data_model['measurement_location']['measurement_station_type_id']=='mast':
            ane_wspd_height=[]

            for point in data_model['measurement_location']['measurement_point']:
                if point['measurement_type_id']=='wind_speed':
                    #Find heights for the meas point
                    lidar_wspd_height.append(point['height_m'])
        
        if data_model['measurement_location']['measurement_station_type_id']=='lidar':
            lidar_wspd_height=[] #Create an empty array to populate with the annemometer heights. These values can then be assigned to their respective naming conventions.

            for point in data_model['measurement_location']['measurement_point']:
                if point['measurement_type_id']=='wind_speed':
                    #Find heights for the meas point
                    lidar_wspd_height.append(point['height_m'])

    def get_std_dev(self, data_model):
        if data_model['measurement_location']['measurement_station_type_id']=='mast':
            ane_sd_name=[]

            for point in data_model['measurement_location']['measurement_point']:
                if point['measurement_type_id']=='wind_speed':
                    #Find std dev names for each height
                    ane_sd_name.append(point['logger_measurement_config']['column_name'][1]['column_name'])

        if data_model['measurement_location']['measurement_station_type_id']=='lidar':
             lidar_sd_name=[]

             for point in data_model['measurement_location']['measurement_point']:
                if point['measurement_type_id']=='wind_speed':
                    #Find std dev names for each height
                    lidar_sd_name.append(point['logger_measurement_config']['column_name'][1]['column_name'])

    def get_TI(self, data_model):
        if data_model['measurement_location']['measurement_station_type_id']=='mast':
            ane_TI_name=[]

            for point in data_model['measurement_location']['measurement_point']:
                if point['measurement_type_id']=='wind_speed':
                    #Find the TI for each height
                    ane_TI_name.append(point['logger_measurement_config']['column_name'][5]['column_name'])## Need to find the correct location

            return ane_TI_name

        if data_model['measurement_location']['measurement_station_type_id']=='lidar':
            lidar_TI_name=[]

            for point in data_model['measurement_location']['measurement_point']:
                if point['measurement_type_id']=='wind_speed':
                    lidar_TI_name.append(point['logger_measurement_config']['column_name'][5]['column_name'])## Need to find the correct location

            return lidar_TI_name


    def get_wspd_dir(self, data_model):
        lidar_dir_name=[]
        if data_model['measurement_type_id']=='wind_direction':
            #Find the direction for each height
            lidar_dir_name.append(data_model['name'])## Need to find the correct location

        return lidar_dir_name