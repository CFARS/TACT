try:
    from TACT import logger
except ImportError:
    pass
import argparse
from future.utils import itervalues, iteritems
import os
import pandas as pd
import re


class Config(object):

    def __init__(self, input_filename='', config_file='', rtd_files='', 
                    results_file='', save_model_location='', time_test_flag='', 
                    global_model='', outpath_dir='', outpath_file='',   ):

        logger.info(f"initiated Config object")

        self.input_filename = input_filename
        self.config_file = config_file
        self.rtd_files = rtd_files
        self.results_file = results_file
        self.save_model_location = save_model_location
        self.time_test_flag = time_test_flag
        self.global_model = global_model

        self.outpath_dir = os.path.dirname(results_file)
        self.outpath_file = os.path.basename(results_file)

    def get_input_files(self):
        """ Used when running tool as script with command line arguments
        
        Parses arguments from command line. 
        """

        logger.info(f"parsing arguments from commandline call")

        parser = argparse.ArgumentParser()
        parser.add_argument("-in","--input_filename", help="print this requires the input filename")
        parser.add_argument("-config","--config_file", help="this requires the excel configuration file")
        parser.add_argument("-globalModel", "--global_model_to_test", help="specify the global model to test on the data",default='RF_model_1_SS_LTERRA_MLb.pkl')
        parser.add_argument("-rtd","--rtd_files", help="this requires input directory for wincube rtd files",default=False)
        parser.add_argument("-res","--results_file", help="this requires the excel results file")
        parser.add_argument("-saveModel", "--save_model_location", help="this argument specifies the location to save output global model", default=False)
        parser.add_argument("-timetestFlag", "--timetestFlag", action="store_true",
                            help = "initiates timing tests for model generation")
        args = parser.parse_args()

        logger.info('input data file: {}'.format(args.input_filename))
        logger.info('input configuration file: {}'.format(args.config_file))
        logger.info('results file: {}'.format(args.results_file))
        logger.info('1 HZ files directory: {}'.format(args.rtd_files))
        logger.info('global model: {}'.format(args.global_model_to_test))

        print('the input data file is {}'.format(args.input_filename))
        print('the input configuration file is {}'.format(args.config_file))
        print('results will output to {}'.format(args.results_file))
        print('windcube 1 HZ rtd. files are located {}'.format(args.rtd_files))
        print('Testing {} as global model'.format(args.global_model_to_test))

        self.input_filename = args.input_filename
        self.config_file = args.config_file
        self.rtd_files = args.rtd_files
        self.results_file = args.results_file
        self.save_model_location = args.save_model_location
        self.time_test_flag = args.timetestFlag
        self.global_model = args.global_model_to_test

        self.outpath_dir = os.path.dirname(self.results_file)
        self.outpath_file = os.path.basename(self.results_file)

    def get_site_metadata(self):
        """
        Parameters
        ----------
        config_file: str or DataFrame
            path to input configuration file or Pandas DataFrame object
            
        Returns
        -------
        DataFrame
            metadata containing information about site
        """
        if isinstance(self.config_file, pd.DataFrame):
            self.site_metadata = self.config_file

        else:
            self.site_metadata = pd.read_excel(self.config_file, usecols=[3, 4, 5], nrows=20)

    def get_filtering_metadata(self):
        """ Create DataFrame Metadata containing information about filtering applied to data
        """

        self.config_metadata = pd.read_excel(self.config_file, usecols=[7, 8, 9], nrows=8)

    def get_adjustments_metadata(self):
        '''create metadata object containing information about which corrections will be applied
                to this data set and why
        '''
        # get list of available data columns and available ancillary data
        self.available_data = pd.read_excel(self.config_file, usecols=[1],nrows=1000).dropna()['Header_CFARS_Python'].to_list()
        # check for .rtd files
        rtd = False
        mainPath = os.path.split(self.config_file)[0]
        if os.path.isdir(os.path.join(mainPath,'rtd_files')):
            rtd = True
        # read height data
        availableHtData = [s for s in self.available_data if 'Ht' in s]
        configHtData = pd.read_excel(self.config_file, usecols=[3, 4], nrows=17).iloc[[3,12,13,14,15]]
        primaryHeight = configHtData['Selection'].to_list()[0]
        # read RSD Type
        self.RSDtype = pd.read_excel(self.config_file, usecols=[4], nrows=8).iloc[6]
        # read NDA status
        ndaStatus = pd.read_excel(self.config_file, usecols=[12], nrows=3).iloc[1]
        # check argument that specifies global model
        globalModel = self.global_model
        # check ability to compute extrapolated TI
        all_heights, ane_heights, RSD_heights, ane_cols, RSD_cols = check_for_additional_heights(self.config_file, primaryHeight)
        self.extrapolation_type = check_for_extrapolations(ane_heights, RSD_heights)

        if self.extrapolation_type is not None:
            self.extrap_metadata = get_extrap_metadata(ane_heights, RSD_heights, self.extrapolation_type)

        else:
            self.extrap_metadata = pd.DataFrame([['extrapolation type', 'None',
                                            "No extrapolation due to insufficient anemometer heights"]],
                                        columns=['Type', 'Height (m)', 'Comparison Height Number'])

        # Make dictionary of potential methods, Note: SS-LTERRA-WC-1HZ, G-LTERRA-WC-1HZ, and G-Std are windcube only (but we want to test on zx) so they are false until we know sensor
        correctionsManager = {'SS-SF':True,'SS-S':True,'SS-SS':True,'SS-Match2':True,'SS-WS':True,'SS-WS-Std':True,
                            'SS-LTERRA-WC-1HZ':False,'SS-LTERRA-MLa':True,'SS-LTERRA-MLb':True,'SS-LTERRA-MLc':True,'TI-Extrap':False,
                            'G-Sa':True,'G-SFa':True,'G-Sc':True,'G-SFc':True,'G-Std':False,'G-Match':True,'G-Ref-S':True,
                            'G-Ref-SF':True, 'G-Ref-SS':True,'G-Ref-WS-Std':True}
        # input data checking
        subset = ['Ref_TI','RSD_TI']
        result = all(elem in self.available_data for elem in subset)
        if result:
            pass
        else:
            if self.RSDtype['Selection']!='No RSD':
                print('Error encountered. Input data does not, but should have TI from reference and/or TI from RSD')
                sys.exit()
            else:
                subset2 = ['Ref_TI','Ane2_TI']
                result2 = all(elem in self.available_data for elem in subset2)
                if result2:
                    pass
                else:
                    print('Error encountered. Input data does not have enough TI data (second Anemometer) to compare. Check input and config')
                    sys.exit()

        # enable methods
        if self.RSDtype['Selection'][0:4] == 'Wind': # if rsd is windcube
            correctionsManager['G-C']=True
            if self.rtd_files == False:
                print ('Rtd file location not specified. Not running 1Hz adjustment. To change this behavior, use argument -rtd_files')
            else:
                correctionsManager['SS-LTERRA-WC-1HZ']=True
                correctionsManager['G-LTERRA-WC-1HZ'] = True
        subset3 = ['RSD_SD']
        result3 = all(elem in self.available_data for elem in subset3)
        if result3:
            pass
        else:
            print ('Error encountered. Input data does not include RSD standard deviation and cannot utilize some corrections methods. Please modify input data to include standard deviation.')
            sys.exit()

        if self.extrapolation_type is not None:
            correctionsManager['TI-Extrap']=True
            correctionsManager['Name global model'] = globalModel

        self.adjustments_metadata = correctionsManager


def get_extrap_metadata(ane_heights, RSD_heights, extrapolation_type):
    """
    get metadata for TI extrapolation
    :param ane_heights: dictionary of height labels and values for anemometers
    :param RSD_heights: dictionary of height labels and values for RSD
    :param extrapolation_type: str to decide what type of extrapolation to perform
    :return extrap_metadata: DataFrame with metadata required for TI extrapolation
    """
    unique_ane_hts = set(ane_heights.values()).difference(set(['unknown']))
    unique_RSD_hts = set(RSD_heights.values()).difference(set(['unknown']))
    overlapping_hts = unique_ane_hts.intersection(unique_RSD_hts)

    # Get extrapolation height and number/label
    if extrapolation_type == 'simple':
        # At least two anemometer heights exist, but no "truth" measurement at extrapolated height
        extrap_height = max(unique_RSD_hts)
    elif extrapolation_type == 'truth':
        # At least three anemometer heights exist, one of which is the same as RSD at extrap ht
        extrap_height = max(overlapping_hts)
    extrap_height_num = [num for num, ht in iteritems(RSD_heights) if ht == extrap_height][0]

    # Get anemometer heights and numbers/labels
    ane_hts_input_num = [num for num, ht in iteritems(ane_heights)
                         if ht in unique_ane_hts.difference(set([extrap_height]))]
    ane_hts_input = [ane_heights[num] for num in ane_hts_input_num]

    # Combine into DataFrame
    extrap_metadata = pd.DataFrame({'height': ane_hts_input, 'num': ane_hts_input_num})
    extrap_metadata['type'] = 'input'
    extrap_metadata = extrap_metadata.append(
        pd.DataFrame([[extrap_height, extrap_height_num, 'extrap']],
                     columns=extrap_metadata.columns),
        ignore_index=True)
    extrap_metadata = extrap_metadata.loc[:, ['type', 'height', 'num']]

    return extrap_metadata


def check_for_extrapolations(ane_heights, RSD_heights):
    """
    Check if columns are specified for other anemometer heights, and extract the column names.
    :param ane_heights: dictionary of height labels and values for anemometers
    :param RSD_heights: dictionary of height labels and values for RSD
    :return extrapolation_type: None or str to decide what type of extrapolation to perform
    Notes on what we need for extrapolation analysis
    """
    unique_ane_hts = set(ane_heights.values()).difference(set(['unknown']))
    unique_RSD_hts = set(RSD_heights.values()).difference(set(['unknown']))
    overlapping_hts = unique_ane_hts.intersection(unique_RSD_hts)

    extrapolation_type = None
    if len(unique_ane_hts) == 2 and (max(unique_RSD_hts) > max(unique_ane_hts)):
        print ('simple')
        extrapolation_type = 'simple'
    elif len(unique_ane_hts) > 2:
        # We still need at least two ane heights that are lower than one RSD height
        tmp = [sum([ht > a for a in unique_ane_hts]) >= 2 for ht in unique_RSD_hts]
        if any(tmp):
            extrapolation_type = 'simple'
        # Otherwise we can have two ane heights that are lower than one overlapping height
        tmp = [sum([ht > a for a in unique_ane_hts]) >= 2 for ht in overlapping_hts]
        if any(tmp):
            extrapolation_type = 'truth'

    return extrapolation_type
