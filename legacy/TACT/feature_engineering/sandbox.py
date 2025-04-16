import pandas as pd



def get_representative_TI_15mps(inputdata):
    # this is the represetative TI, this is currently only done at a 1m/s bins not sure if this needs to be on .5m/s
    # TODO: find out if this needs to be on the 1m/s bin or the .5m/s bin
    inputdata_TI15 = inputdata[inputdata['bins'] == 15]
    listofcols = ['Ref_TI']
    if 'RSD_TI' in inputdata.columns:
        listofcols.append('RSD_TI')
    if 'Ane2_TI' in inputdata.columns:
        listofcols.append('Ane2_TI')
    if 'adjTI_RSD_WS' in inputdata.columns:
        listofcols.append('adjTI_RSD_WS')
    results = inputdata_TI15[listofcols].describe()
    results.loc['Rep_TI', :] = results.loc['mean'] + 1.28 * results.loc['std']
    results = results.loc[['mean', 'std', 'Rep_TI'], :].T
    results.columns = ['mean_15mps', 'std_15mps', 'Rep_TI']
    return results


def check_for_alphaConfig(config_file,inputdata):
    """
    checks to see if the configurations are there to compute alpha from cup
    checks to see if the configurations are there to compute alpha from RSD
    """
    RSD_alphaFlag = False

    # get list of available data columns and available ancillary data
    availableData = pd.read_excel(config_file, usecols=[1], nrows=1000).dropna()['Header_CFARS_Python'].to_list()
    if 'RSD_alpha_lowHeight' in availableData and 'RSD_alpha_highHeight' in availableData:
        RSD_alphaFlag = True
        configHtData = pd.read_excel(config_file, usecols=[3, 4], nrows=25).iloc[[20,21]]
        Ht_1_rsd = configHtData['Selection'].to_list()[0]
        Ht_2_rsd = configHtData['Selection'].to_list()[1]
    else:
        print ('%%%%%%%%% Warning: No alpha calculation. To compute alpha check config file settings. %%%%%%%%%%%')
        Ht_1_rsd = None
        Ht_2_rsd = None

    return RSD_alphaFlag, Ht_1_rsd, Ht_2_rsd


def calculate_stability_alpha(inputdata, config_file, RSD_alphaFlag, Ht_1_rsd, Ht_2_rsd):
    '''
    Computation of wind shear exponent. From Wharton and Lundquist 2012
    
    Stability class from shear exponent categories:
        [1]     strongly stable -------- alpha > 0.3
        [2]              stable -------- 0.2 < alpha < 0.3
        [3]        near-neutral -------- 0.1 < alpha < 0.2
        [4]          convective -------- 0.0 < alpha < 0.1
        [5] strongly convective -------- alpha < 0.0
    
    Parameters
    ----------
    inputdata: data struct
        all RSD, anemometer, and wind vane data
    config_file: filename
        *NB: replace with Config() class object*
        Configuration and metadata of dataset
    RSD_alphaFlag: Boolean
        Should shear be computed using RSD data (or anem data) ?
    Ht_1_rsd: float
        Lower height, in meters, from RSD to use in computing shear
    Ht_2_rsd: float
        Upper height, in meters, from RSD to use in computing shear
    
    Returns
    --------
    cup_alphaFlag: Boolean
        Is stability computed from Anemometer data?
    stabilityClass_ane: DataFrame
        DataFrame of integer Anem stability classes
    stabilityMetric_ane: DataFrame
        DataFrame of integer Anem stability metric values
    regimeBreakdown_ane: DataFrame
        Counts and percentages of each stability class computed from RSD data
    Ht_1_ane: float
        Minimum anemometer height in data
    Ht_2_ane: float
        Maximum anemometer height in data
    stabilityClass_rsd: DataFrame
        DataFrame of integer RSD stability classes
    stabilityMetric_rsd: DataFrame
        DataFrame of float RSD stability metric values
    regimeBreakdown_rsd: DataFrame
        Counts and percentages of each stability class computed from RSD data
    
    Notes: Can this be broken into multiple functions? One for Anems, one for RSDs
           Both could have a switch on whether to report the statistics of stability
           or at least varoutput. Simplest: input data Output: stability values and/or categories
    TO DO:
    - Change config_file to use config_data
    
    '''

    regimeBreakdown_ane = pd.DataFrame()

    #check for 2 anemometer heights (use furthest apart) for cup alpha calculation
    configHtData = pd.read_excel(config_file, usecols=[3, 4], nrows=17).iloc[[3,12,13,14,15]]
    primaryHeight = configHtData['Selection'].to_list()[0]
    all_heights, ane_heights, RSD_heights, ane_cols, RSD_cols = check_for_additional_heights(config_file, primaryHeight)
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


def get_all_heights(config_file, primary_height):
    all_heights = {'primary': primary_height}

    df = pd.read_excel(config_file, usecols=[3, 4]).dropna()

    ix = ["Additional Comparison Height" in h for h in df['Site Metadata']]
    additional_heights = df.loc[ix, :]
    if len(additional_heights) > 0:
        tmp = {i: additional_heights['Site Metadata'].str.contains(str(i)) for i in range(1, 5)}
        for ht in tmp:
            if not any(tmp[ht]):
                continue
            all_heights[ht] = float(additional_heights['Selection'][tmp[ht].values])
    return all_heights


def check_for_additional_heights(config_file, height):
    '''
    Check if columns are specified for other heights, and extract the column names.
    :param config_file: Input configuration file
    :param height: Primary comparison height (m)
    :return all_heights: dictionary of all height labels and values
    :return ane_heights: dictionary of height labels and values for anemometers
    :return RSD_heights: dictionary of height labels and values for RSD
    :return ane_cols: list of column names corresponding to additional anemometer heights
    :return RSD_cols: list of column names corresponding to additional RSD heights
    '''
    # Get dictionary of all heights
    all_heights = get_all_heights(config_file, height)

    df = pd.read_excel(config_file, usecols=[0, 1]).dropna()

    # Run a quick check to make sure program exits gracefully if user makes a mistake in config
    #  (i.e, not having necessary variables, duplicate variables etc.)
    cfarsColList = df.Header_CFARS_Python.tolist()

    # Check to see if we have additional heights
    optional_height_names = get_optional_height_names()
    all_cols = []
    for cols in optional_height_names:
        col_present = [col in cfarsColList for col in cols]
        if not any(col_present):
            continue
        if (set(cols).issubset(set(cfarsColList))) == False:
            missing = set(cols).difference(set(cfarsColList))
            sys.exit(
                'You have not specified all variables for each measurement at an additional height.\n'
                + 'You are missing the following variables in the Header_CFARS_Python that are necessary:\n'
                + str(missing)
                + '\n Please fix and restart to run')
        all_cols.extend(cols)

    RSD_cols = [col for col in all_cols if 'RSD' in col]
    ane_cols = [col for col in all_cols if 'Ane' in col]

    # Get the height numbers of any Additional Comparison Heights
    regexp = re.compile('[a-zA-Z_]+(?P<ht>\d+)')
    ane_hts = [regexp.match(c) for c in ane_cols]
    ane_hts = [c.groupdict()['ht'] for c in ane_hts if c is not None]
    ane_hts = [int(h) for h in set(ane_hts)]
    RSD_hts = [regexp.match(c) for c in RSD_cols]
    RSD_hts = [c.groupdict()['ht'] for c in RSD_hts if c is not None]
    RSD_hts = [int(h) for h in set(RSD_hts)]

    # Check for Additional Comparison Heights not specified in the config file
    missing_ane = set(ane_hts).difference(set(all_heights))
    missing_RSD = set(RSD_hts).difference(set(all_heights))
    missing = missing_ane.union(missing_RSD)
    if len(missing) > 0:
        sys.exit(
            'You have not specified the Additional Comparison Height (m) for each height.\n'
            + 'You are missing the following heights in the "Selection" column that are necessary:\n\n'
            + 'Additional Comparison Height ' + str(missing) + '\n\n'
            + 'Please fix and restart to run')

    # Create dictionaries with all anemometer and RSD heights
    # NOTE : here we assume it's okay if we don't have anemometer AND RSD data for EACH additional
    #        comparison height
    ane_heights = {ht: all_heights[ht] for ht in all_heights if ht in ane_hts}
    ane_heights.update(primary=all_heights['primary'])
    RSD_heights = {ht: all_heights[ht] for ht in all_heights if ht in RSD_hts}
    RSD_heights.update(primary=all_heights['primary'])

    # Do a final check in case there are any heights that were specified but no matching columns
    missing = set(all_heights.values()).difference(set(RSD_heights.values()).
                                                   union(set(ane_heights.values())))
    if len(missing) > 0:
        sys.exit(
            'You have specified an Additional Comparison Height (m) that has no corresponding data columns.\n'
            + 'The following heights in the "Selection" column have no data columns specified:\n\n'
            + 'Additional Comparison Height ' + str(missing) + '\n\n'
            + 'Please fix and restart to run')

    return all_heights, ane_heights, RSD_heights, ane_cols, RSD_cols



# Which functions are FeatureEngineering?

# 6750
# This is computing the TI for ZX, and throwing a note if Sodar 3.0 is implemented in Triton
# Computing TI = LABEL engineering :)
# 'calculate_ti' with subfunctions for ZX, WindCube, Triton, et al.

# calculate_ti, calculate_shear, calculate_tke, create_bins, all support functions

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
# 6763

# 6769
# Computing stability from TKE. Definitely feature engineering
# 'calculate_stability_TKE' and 'calculate_stability_alpha'
stabilityClass_tke, stabilityMetric_tke, regimeBreakdown_tke = \
    calculate_stability_TKE(inputdata)

cup_alphaFlag, stabilityClass_ane, stabilityMetric_ane, regimeBreakdown_ane, Ht_1_ane, Ht_2_ane, \
    stabilityClass_rsd, stabilityMetric_rsd, regimeBreakdown_rsd = \
        calculate_stability_alpha(inputdata, config_file, RSD_alphaFlag, Ht_1_rsd, Ht_2_rsd)

# Skip all Time Sensitivity Analysis 6772 - 6823

# 6824 Test Train Split

# 6836

"""
- Compute TI from data for each RSD type
- Compute TKE from data
- Compute wind shear from data
- Attach TKE and shear 'groups' to inputdata
- Set up inputdata with stability classes for S-A-C with function 'get_all_regressions'
- Test train split
- How is All_class_data being used??
- ResultsLists functions need similar structure
"""

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
