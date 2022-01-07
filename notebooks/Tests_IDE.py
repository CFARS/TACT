# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:25:35 2021

@author: AHBL
"""
import os
import openpyxl
from TACT.readers.config import Config
from TACT.readers.data import Data

"""parser get_input_files"""

example_path = "C:/GitHub/site_suitability_tool/Example/"

config = Config(
    input_filename=example_path+"example_project.csv",
    config_file=example_path+"configuration_example_project.xlsx",
    rtd_files='', 
    results_file=example_path+'Test_IDE_results.xlsx',
    save_model_location='',
    time_test_flag=False, 
)

config.outpath_dir = os.path.dirname(config.results_file)
config.outpath_file = os.path.basename(config.results_file)

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
data = Data(input_filename=example_path+"example_project.csv",
    config_file=example_path+"configuration_example_project.xlsx")

data.get_inputdata()
data.get_refTI_bins()
data.check_for_alphaConfig()


#%%
"""
config_object

- site_suitability_tool/readers/config_file.py
- parameters
    -
- object includes
    - config_file
    - input_filename
    - rtd_files
    - results_filename
    - saveModel
    - time_test_flag
    - global_model
    -
    - siteMetaData
    - filterMetaData
    - correctionsMetaData
    - RSDtype
    - extrap_metadata
    - extrapolation_type

data_object
    - config_object
    - inputdata
    - Timestamps
    - a
    - lab_a
    - RSD_alphaFlag
    - Ht_1_rsd
    - Ht_2_rsd

"""


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
reg_results = get_all_regressions(inputdata, title = 'Full comparison')
sensor, height = get_phaseiii_metadata(config_file)
stabilityClass_tke, stabilityMetric_tke, regimeBreakdown_tke = calculate_stability_TKE(inputdata)
cup_alphaFlag, stabilityClass_ane, stabilityMetric_ane, regimeBreakdown_ane, Ht_1_ane, Ht_2_ane, stabilityClass_rsd, stabilityMetric_rsd, regimeBreakdown_rsd = calculate_stability_alpha(inputdata, config_file, RSD_alphaFlag, Ht_1_rsd, Ht_2_rsd)
#------------------------
# Time Sensivity Analysis
#------------------------
TimeTestA = pd.DataFrame()
TimeTestB = pd.DataFrame()
TimeTestC = pd.DataFrame()

if timetestFlag == True:
    # A) increase % of test train split -- check for convergence --- basic metrics recorded baseline but also for every corrections
    splitList = np.linspace(0.0, 100.0, num = 20, endpoint =False)
    print ('Testing model generation time period sensitivity...% of data')
    TimeTestA_corrections_df = {}
    TimeTestA_baseline_df = pd.DataFrame()
    for s in splitList[1:]:
        print (str(str(s) + '%'))
        inputdata_test = train_test_split(s,inputdata.copy())
        TimeTestA_baseline_df, TimeTestA_corrections_df = QuickMetrics(inputdata_test,TimeTestA_baseline_df,TimeTestA_corrections_df,str(100-s))

    # B) incrementally Add days to training set sequentially -- check for convergence
    numberofObsinOneDay = 144
    numberofDaysInTest = int(round(len(inputdata)/numberofObsinOneDay))
    print ('Testing model generation time period sensitivity...days to train model')
    print ('Number of days in the study ' + str(numberofDaysInTest))
    TimeTestB_corrections_df = {}
    TimeTestB_baseline_df = pd.DataFrame()
    for i in range(0,numberofDaysInTest):
        print (str(str(i) + 'days'))
        windowEnd = (i+1)*(numberofObsinOneDay)
        inputdata_test = train_test_split(i,inputdata.copy(), stepOverride = [0,windowEnd])
        TimeTestB_baseline_df, TimeTestB_corrections_df = QuickMetrics(inputdata_test,TimeTestB_baseline_df, TimeTestB_corrections_df,str(numberofDaysInTest-i))

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
            TimeTestC_baseline_df, TimeTestC_corrections_df = QuickMetrics(inputdata_test, TimeTestC_baseline_df, TimeTestC_corrections_df,
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
# TI Adjustments
# ------------------------
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

for method in correctionsMetadata:

    # ************************************ #
    # Site Specific Simple Correction (SS-S)
    if method != 'SS-S':
        pass
    elif method == 'SS-S' and correctionsMetadata['SS-S'] == False:
        pass
    else:
        print ('Applying Correction Method: SS-S')
        inputdata_corr, lm_corr, m, c = perform_SS_S_correction(inputdata.copy())
        print("SS-S: y = " + str(m) + " * x + " + str(c))
        lm_corr['sensor'] = sensor
        lm_corr['height'] = height
        lm_corr['correction'] = 'SS-S'
        correctionName = 'SS_S'

        baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                 Timestamps, method)
        TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

        if RSDtype['Selection'][0:4] == 'Wind':
            print ('Applying Correction Method: SS-S by stability class (TKE)')
            # stability subset output for primary height (all classes)
            ResultsLists_class = initialize_resultsLists('class_')
            className = 1
            for item in All_class_data:
                inputdata_corr, lm_corr, m, c = perform_SS_S_correction(item[primary_idx].copy())
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
            print ('Applying Correction Method: SS-S by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            print (str('class ' + str(className)))
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_SS_S_correction(item.copy())
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
            print ('Applying correction Method: SS-S by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_SS_S_correction(item.copy())
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
        print ('Applying Correction Method: SS-SF')
        inputdata_corr, lm_corr, m, c = perform_SS_SF_correction(inputdata.copy())
        print("SS-SF: y = " + str(m) + " * x + " + str(c))
        lm_corr['sensor'] = sensor
        lm_corr['height'] = height
        lm_corr['correction'] = 'SS-SF'
        correctionName = 'SS_SF'

        baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                 Timestamps, method)
        TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)
        
        if RSDtype['Selection'][0:4] == 'Wind' or 'ZX' in RSDtype['Selection']:
            print ('Applying Correction Method: SS-SF by stability class (TKE)')
            # stability subset output for primary height (all classes)
            ResultsLists_class = initialize_resultsLists('class_')
            className = 1
            for item in All_class_data:
                inputdata_corr, lm_corr, m, c = perform_SS_SF_correction(item[primary_idx].copy())
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
            print ('Applying Correction Method: SS-SF by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_SS_SF_correction(item.copy())
                print ("SS-SF: y = " + str(m) + "* x +" + str(c))
                lm_corr['sensor'] = sensor
                lm_corr['height'] = height
                lm_corr['correction'] = str('SS-SF' + '_' + 'class_' + str(className))
                correctionName = str('SS_SF' + '_alphaRSD_' + str(className))
                ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                     inputdata_corr, Timestamps, method)
                className += 1
            ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')

        if cup_alphaFlag:
            print ('Applying correction Method: SS-SF by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_SS_SF_correction(item.copy())
                print ("SS-SF: y = " + str(m) + "* x +" + str(c))
                lm_corr['sensor'] = sensor
                lm_corr['height'] = height
                lm_corr['correction'] = str('SS-SF' + '_' + 'class_' + str(className))
                correctionName = str('SS_SF' + '_alphaCup_' + str(className))
                ResultsLists_class_alpha_Ane = populate_resultsLists(ResultsLists_class_alpha_Ane, 'class_alpha_Ane', correctionName, lm_corr,
                                                                     inputdata_corr, Timestamps, method)
                className += 1
            ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, 'alpha_Ane')

    # ************************************ #
    # Site Specific Simple Correction (SS-SS) combining stability classes adjusted differently
    if method != 'SS-SS':
        pass
    elif method == 'SS-SS' and correctionsMetadata['SS-SS'] == False:
        pass
    elif RSDtype['Selection'][0:4] != 'Wind' and 'ZX' not in RSDtype['Selection']:
        pass
    else:
        print ('Applying Correction Method: SS-SS')
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
            print ('Applying Correction Method: SS-SS by stability class (TKE). SAME as Baseline')
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
            print ('Applying Correction Method: SS-SS by stability class Alpha w/ RSD. SAEM as Baseline')
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
            print ('Applying correction Method: SS-SS by stability class Alpha w/cup')
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
        print ('Applying Correction Method: SS-WS')
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
            print ('Applying Correction Method: SS-WS by stability class (TKE)')
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
            print ('Applying Correction Method: SS-WS by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_SS_WS_correction(item.copy())
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
            print ('Applying correction Method: SS-WS by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_SS_WS_correction(item.copy())
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
       print ('Applying Correction Method: SS-WS-Std')
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
           print ('Applying Correction Method: SS-WS-Std by stability class (TKE)')
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
           print ('Applying Correction Method: SS-WS-Std by stability class Alpha w/ RSD')
           ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
           className = 1
           for item in All_class_data_alpha_RSD:
               iputdata_corr, lm_corr, m, c = perform_SS_WS_Std_correction(item.copy())
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
           print ('Applying correction Method: SS-WS-Std by stability class Alpha w/cup')
           ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
           className = 1
           for item in All_class_data_alpha_Ane:
               iputdata_corr, lm_corr, m, c = perform_SS_WS_Std_correction(item.copy())
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
       print ('Applying Correction Method: SS-LTERRA-WC-1HZ')


    # ******************************************************************* #
    # Site Specific LTERRA WC Machine Learning Correction (SS-LTERRA-MLa)
    # Random Forest Regression with now ancillary columns
    if method != 'SS-LTERRA-MLa':
        pass
    elif method == 'SS-LTERRA-MLa' and correctionsMetadata['SS-LTERRA-MLa'] == False:
        pass
    else:
        print ('Applying Correction Method: SS-LTERRA-MLa')
        inputdata_corr, lm_corr, m, c = perform_SS_LTERRA_ML_correction(inputdata.copy())
        lm_corr['sensor'] = sensor
        lm_corr['height'] = height
        lm_corr['correction'] = 'SS_LTERRA_MLa'
        correctionName = 'SS_LTERRA_MLa'
        baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                 Timestamps, method)
        TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)
        
        if RSDtype['Selection'][0:4] == 'Wind':
            print ('Applying Correction Method: SS-LTERRA MLa by stability class (TKE)')
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
            print ('Applying Correction Method: SS-LTERRA MLa by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_SS_LTERRA_ML_correction(item.copy())
                lm_corr['sensor'] = sensor
                lm_corr['height'] = height
                lm_corr['correction'] = str('SS-LTERRA_MLa' + '_' + 'class_' + str(className))
                correctionName = str('SS_LTERRA_ML' + '_alphaRSD_' + str(className))
                ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                     inputdata_corr, Timestamps, method)
                className += 1
            ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
        if cup_alphaFlag:
            print ('Applying correction Method: SS-LTERRA_MLa by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_SS_LTERRA_ML_correction(item.copy())
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
        print ('Applying Correction Method: SS-LTERRA-MLc')
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
            print ('Applying Correction Method: SS-LTERRA_MLc by stability class (TKE)')
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
            print ('Applying Correction Method: SS-LTERRA_MLc by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                lm_corr['sensor'] = sensor
                lm_corr['height'] = height
                lm_corr['correction'] = str('SS-LTERRA_MLc' + '_' + 'class_' + str(className))
                correctionName = str('SS_LTERRA_S_ML' + '_alphaRSD_' + str(className))
                ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                     inputdata_corr, Timestamps, method)
                className += 1
            ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
        if cup_alphaFlag:
            print ('Applying correction Method: SS-LTERRA_MLc by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
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
        print ('Applying Correction Method: SS-LTERRA-MLb')
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
            print ('Applying Correction Method: SS-LTERRA_MLb by stability class (TKE)')
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
            print ('Applying Correction Method: SS-LTERRA_MLb by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
                lm_corr['sensor'] = sensor
                lm_corr['height'] = height
                lm_corr['correction'] = str('SS-LTERRA_MLb' + '_' + 'class_' + str(className))
                correctionName = str('SS_LTERRA_MLb' + '_alphaRSD_' + str(className))
                ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                     inputdata_corr, Timestamps, method)
                className += 1
            ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
        if cup_alphaFlag:
            print ('Applying correction Method: SS-LTERRA_MLb by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_SS_LTERRA_S_ML_correction(item.copy(),all_trainX_cols,all_trainY_cols,all_testX_cols,all_testY_cols)
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
        blockPrint()
        # Get extrapolation height
        height_extrap = float(extrap_metadata['height'][extrap_metadata['type'] == 'extrap'])
        # Extrapolate
        inputdata_corr, lm_corr, shearTimeseries= perform_TI_extrapolation(inputdata.copy(), extrap_metadata,
                                                                           extrapolation_type, height)
        correctionName = 'TI_EXTRAP'
        lm_corr['correction'] = correctionName

        inputdataEXTRAP = inputdata_corr.copy()
        inputdataEXTRAP, baseResultsLists = extrap_configResult(inputdataEXTRAP, baseResultsLists, method,lm_corr)

        if RSDtype['Selection'][0:4] == 'Wind':
            # stability subset output for primary height (all classes)
            ResultsLists_class = initialize_resultsLists('class_')
            className = 1
            for item in All_class_data:
                inputdata_corr, lm_corr, shearTimeseries= perform_TI_extrapolation(item[primary_idx].copy(), extrap_metadata,
                                                                                   extrapolation_type, height)
                lm_corr['correction'] = str('TI_EXT_class1' + '_TKE_' + 'class_' + str(className))
                inputdataEXTRAP = inputdata_corr.copy()
                inputdataEXTRAP, ResultsLists_class = extrap_configResult(inputdataEXTRAP, ResultsLists_class,
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
                inputdataEXTRAP, ResultsLists_class_alpha_Ane = extrap_configResult(inputdataEXTRAP,
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
                inputdataEXTRAP, ResultsLists_class_alpha_RSD = extrap_configResult(inputdataEXTRAP,
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
        enablePrint()

    # ************************************************** #
    # Histogram Matching
    if method != 'SS-Match':
        pass
    elif method == 'SS-Match' and correctionsMetadata['SS-Match'] == False:
        pass
    else:
        print ('Applying Match algorithm: SS-Match')
        inputdata_corr, lm_corr = perform_Match(inputdata.copy())
        lm_corr['sensor'] = sensor
        lm_corr['height'] = height
        lm_corr['correction'] = 'SS-Match'
        correctionName = 'SS_Match'

        baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                 Timestamps, method)
        TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

        if RSDtype['Selection'][0:4] == 'Wind':
            print ('Applying Correction Method: SS-Match by stability class (TKE)')
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
            print ('Applying Correction Method: SS-Match by stability class Alpha w/ RSD')
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
            print ('Applying correction Method: SS-Match by stability class Alpha w/cup')
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
        print ('Applying input match algorithm: SS-Match2')
        inputdata_corr, lm_corr = perform_Match_input(inputdata.copy())
        lm_corr['sensor'] = sensor
        lm_corr['height'] = height
        lm_corr['correction'] = 'SS-Match2'
        correctionName = 'SS_Match2'

        baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                 Timestamps, method)
        TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)

        if RSDtype['Selection'][0:4] == 'Wind':
            print ('Applying Correction Method: SS-Match2 by stability class (TKE)')
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
            print ('Applying Correction Method: SS-Match2 by stability class Alpha w/ RSD')
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
            print ('Applying correction Method: SS-Match2 by stability class Alpha w/cup')
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
        print ('Applying Correction Method: G-Sa')
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
            print ('Applying Correction Method: G-Sa by stability class (TKE)')
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
            print ('Applying Correction Method: G-Sa by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
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
            print ('Applying correction Method: G-Sa by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
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
        print ('Applying Correction Method: G-SFa')
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
            print ('Applying Correction Method: G-SFa by stability class (TKE)')
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
            print ('Applying Correction Method: G-SFa by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
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
            print ('Applying correction Method: G-SFa by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_G_Sa_correction(item.copy(),override)
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
        print ('Applying Correction Method: G-Sc')
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
            print ('Applying Correction Method: G-SFa by stability class (TKE)')
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
            print ('Applying Correction Method: G-SFc by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                iputdata_corr, lm_corr, m, c = perform_G_SFc_correction(item.copy())
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
            print ('Applying correction Method: G-SFc by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                iputdata_corr, lm_corr, m, c = perform_G_SFc_correction(item.copy())
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
        print ('Applying Correction Method: G-C')
        inputdata_corr, lm_corr, m, c = perform_G_C_correction(inputdata.copy())
        lm_corr['sensor'] = sensor
        lm_corr['height'] = height
        lm_corr['correction'] = 'G-C'
        correctionName = 'G_C'
        baseResultsLists = populate_resultsLists(baseResultsLists, '', correctionName, lm_corr, inputdata_corr,
                                                 Timestamps, method)
        TI_10minuteAdjusted = record_TIadj(correctionName,inputdata_corr,Timestamps, method, TI_10minuteAdjusted, emptyclassFlag=False)
       
        if RSDtype['Selection'][0:4] == 'Wind':
            print ('Applying Correction Method: G-C by stability class (TKE)')
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
            print ('Applying Correction Method: G-C by stability class Alpha w/ RSD')
            ResultsLists_class_alpha_RSD = initialize_resultsLists('class_alpha_RSD')
            className = 1
            for item in All_class_data_alpha_RSD:
                print (str('class ' + str(className)))
                iputdata_corr, lm_corr, m, c = perform_G_C_correction(item.copy())
                lm_corr['sensor'] = sensor
                lm_corr['height'] = height
                lm_corr['correction'] = str('G-C' + '_' + 'class_' + str(className))
                correctionName = str('G-C' + '_alphaRSD_' + str(className))
                ResultsLists_class_alpha_RSD = populate_resultsLists(ResultsLists_class_alpha_RSD, 'class_alpha_RSD', correctionName, lm_corr,
                                                                     inputdata_corr, Timestamps, method)
                className += 1
            ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, 'alpha_RSD')
            

        if cup_alphaFlag:
            print ('Applying correction Method: G-C by stability class Alpha w/cup')
            ResultsLists_class_alpha_Ane = initialize_resultsLists('class_alpha_Ane')
            className = 1
            for item in All_class_data_alpha_Ane:
                print (str('class ' + str(className)))
                iputdata_corr, lm_corr, m, c = perform_G_C_correction(item.copy())
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
        print ('Applying Correction Method: G-Match')

    # ************************ #
    # Global Comprehensive (G-Ref-S)
    if method != 'G-Ref-S':
        pass
    elif method == 'G-Ref-S' and correctionsMetadata['G-Ref-S'] == False:
        pass
    else:
        print ('Applying Correction Method: G-Ref-S')

    # ************************ #
    # Global Comprehensive (G-Ref-Sf)
    if method != 'G-Ref-Sf':
        pass
    elif method == 'G-Ref-Sf' and correctionsMetadata['G-Ref-Sf'] == False:
        pass
    else:
        print ('Applying Correction Method: G-Ref-Sf')
        
    # ************************ #
    # Global Comprehensive (G-Ref-SS)
    if method != 'G-Ref-SS':
        pass
    elif method == 'G-Ref-SS' and correctionsMetadata['G-Ref-SS'] == False:
        pass
    else:
        print ('Applying Correction Method: G-Ref-SS')
    # ************************ #
    # Global Comprehensive (G-Ref-SS-S)
    if method != 'G-Ref-SS-S':
        pass
    elif method == 'G-Ref-SS-S' and correctionsMetadata['G-Ref-SS-S'] == False:
        pass
    else:
        print ('Applying Correction Method: G-Ref-SS-S')
    # ************************ #
    # Global Comprehensive (G-Ref-WS-Std)
    if method != 'G-Ref-WS-Std':
        pass
    elif method == 'G-Ref-WS-Std' and correctionsMetadata['G-Ref-WS-Std'] == False:
        pass
    else:
        print ('Applying Correction Method: G-Ref-WS-Std')

    # ***************************************** #
    # Global LTERRA WC 1Hz Data (G-LTERRA_WC_1Hz)
    if method != 'G-LTERRA_WC_1Hz':
        pass
    elif method == 'G-LTERRA_WC_1Hz' and correctionsMetadata['G-LTERRA_WC_1Hz'] == False:
        pass
    else:
        print ('Applying Correction Method: G-LTERRA_WC_1Hz')

    # ************************************************ #
    # Global LTERRA ZX Machine Learning (G-LTERRA_ZX_ML)
    if method != 'G-LTERRA_ZX_ML':
        pass
    elif correctionsMetadata['G-LTERRA_ZX_ML'] == False:
        pass
    else:
        print ('Applying Correction Method: G-LTERRA_ZX_ML')

    # ************************************************ #
    # Global LTERRA WC Machine Learning (G-LTERRA_WC_ML)
    if method != 'G-LTERRA_WC_ML':
        pass
    elif correctionsMetadata['G-LTERRA_WC_ML'] == False:
        pass
    else:
        print ('Applying Correction Method: G-LTERRA_WC_ML')

    # ************************************************** #
    # Global LTERRA WC w/Stability 1Hz (G-LTERRA_WC_S_1Hz)
    if method != 'G-LTERRA_WC_S_1Hz':
        pass
    elif method == 'G-LTERRA_WC_S_1Hz' and correctionsMetadata['G-LTERRA_WC_S_1Hz'] == False:
        pass
    else:
        print ('Applying Correction Method: G-LTERRA_WC_S_1Hz')

    # ************************************************************** #
    # Global LTERRA WC w/Stability Machine Learning (G-LTERRA_WC_S_ML)
    if method != 'G-LTERRA_WC_S_ML':
        pass
    elif method == 'G-LTERRA_WC_S_ML' and correctionsMetadata['G-LTERRA_WC_S_ML'] == False:
        pass
    else:
        print ('Applying Correction Method: G-LTERRA_WC_S_ML')

if  RSD_alphaFlag:
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

