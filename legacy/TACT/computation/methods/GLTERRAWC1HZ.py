import datetime
import numpy as np
import os
import pandas as pd
from TACT.computation.adjustments import Adjustments, empirical_stdAdjustment


def perform_G_LTERRA_WC_1HZ_adjustment(inputdata):
    """
    In development
    """

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
    dir_rtd = os.path.join("/Users/aearntsen/L-TERRA", "WindcubeRTDs")
    dir_npz = os.path.join(
        "/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/lidar_directory",
        "ProcessedRTDData",
    )

    # Height where variance and wind speed should be extracted
    #    height_needed = 55
    # Specify the time period for data extraction
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
    # Extract raw off-vertical radial wind speed components
    #                            [vr_n,vr_e,vr_s,vr_w] = WC_processing_standard(jj,"vr",height_needed)
    #                        except:
    #                            pass
    # [vr_n,vr_n_dispersion_interp,vr_e,vr_e_dispersion_interp,vr_s,vr_s_dispersion_interp,vr_w,vr_w_dispersion_interp,
    #  vert_beam_rot,vert_beam_dispersion_interp,heights,U,time_datenum_10min,hub_height_index,SNR_n_interp,SNR_e_interp,
    #  SNR_s_interp,SNR_w_interp,SNR_vert_beam_interp] = WC_processing_standard(jj,"vr",height_needed)

    # Write output to 10-min. files
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

    # test adjustments
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

    # Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
    # For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
    # not be used in a power performance test.

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
    #    vol_opts = ["none","spectral_adjustment_fit","acf"]
    #    contamination_opts = ["none","taylor_ws","taylor_var"]

    #    for ii in ws_opts:
    #        print (str('ws opts' + ii))
    #        for jj in noise_opts:
    #            print (str('noise opts' + jj))
    #            for kk in vol_opts:
    #                print (str('vol opts' + kk))
    #                for mm in contamination_opts:
    #                    print (str('contam opts' + mm))
    #                    #Initialize arrays for the new lidar TI after each adjustment has been applied
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

    #                            #Apply noise adjustment and calculate variance
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

    # Corrected TI is the value of the TI after it has been through all adjustment modules
    #                    TI_WC_adjected_all = TI_WC_var_contamination_all
    #
    #                    opts = 'WS_' + mode_ws +'_N_' + mode_noise + '_V_' + mode_vol + '_C_' + mode_contamination

    #                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_adjected_all)]
    #                    mask = functools.reduce(np.logical_and, mask)
    #                    MAE_all = MAE(TI_ref_all[mask],TI_WC_adjected_all[mask])

    #                    #Classify stability by shear parameter, p. A different stability metric could be used if available.
    #                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_adjected_all),p_all >= 0.2]
    #                    mask = functools.reduce(np.logical_and, mask)
    #                    MAE_s = MAE(TI_ref_all[mask],TI_WC_adjected_all[mask])

    #                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_adjected_all),p_all >= 0.1,p_all < 0.2]
    #                    mask = functools.reduce(np.logical_and, mask)
    #                    MAE_n = MAE(TI_ref_all[mask],TI_WC_adjected_all[mask])

    #                    mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_adjected_all),p_all < 0.1]
    #                    mask = functools.reduce(np.logical_and, mask)
    #                    MAE_u = MAE(TI_ref_all[mask],TI_WC_adjected_all[mask])

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

    # Write out final MAE values after all adjustments have been applied for this model combination
    #                    opts_temp = 'WS_' + mode_ws +'_N_' + mode_noise + '_V_' + mode_vol + '_C_' + mode_contamination
    #                    with open(main_directory + '/'+'L_TERRA_combination_summary_WC.csv', 'a') as fp:
    #                        a = csv.writer(fp, delimiter=',')
    #                        data = [[opts_temp,'{:0.2f}'.format(MAE_all),'{:0.2f}'.format(MAE_s),\
    #                        '{:0.2f}'.format(MAE_n),'{:0.2f}'.format(MAE_u)]]
    #                        a.writerows(data)

    # Write out minimum MAE values for each stability class and model options associated with these minima
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

    # apply adjustments
    from itertools import compress

    ##########################################################
    # USER INPUTS GO HERE
    # Directory where CSV output file will be stored
    main_directory = os.path.join("/Volumes/New P/DataScience/CFARS", "NRGSeptLTERRA")
    # Directory where lidar data are saved
    lidar_directory = os.path.join(
        "/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA/lidar_directory",
        "ProcessedRTDData",
    )

    # Height were TI values will be compared
    height_needed = 55
    # Time period for data extraction
    years = [2019, 2020]
    years = [2019]
    months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    months = [9]
    days = np.arange(1, 31)
    days = np.arange(25, 27)
    # Wind speed minimum and maximum thresholds. If threshold is not needed, put 'none'.
    U_min = "none"
    U_max = "none"
    # Wind direction sectors to exclude. If exclusion sector is not needed, put 'none' for wd_min and wd_max values.
    wd_sector_min1 = "none"
    wd_sector_max1 = "none"
    wd_sector_min2 = "none"
    wd_sector_max2 = "none"
    # Model options to use for different stability conditions. If adjustment is not needed, put 'none'.
    # Options for stable conditions (p >= 0.2)
    mode_ws_s = "raw_WC"
    mode_noise_s = "spike"
    mode_vol_s = "acf"
    mode_contamination_s = "taylor_ws"
    # Options for neutral conditions (0.1 <= p < 0.2)
    mode_ws_n = "raw_WC"
    mode_noise_n = "spike"
    mode_vol_n = "acf"
    mode_contamination_n = "taylor_ws"
    # Options for unstable conditions (p < 0.1)
    mode_ws_u = "raw_WC"
    mode_noise_u = "none"
    mode_vol_u = "none"
    mode_contamination_u = "none"
    ##########################################################
    if "none" in str(U_min):
        U_min = 0
    if "none" in str(U_max):
        U_max = np.inf
    if "none" in str(wd_sector_min1):
        wd_sector_min1 = np.nan
    if "none" in str(wd_sector_max1):
        wd_sector_max1 = np.nan
    if "none" in str(wd_sector_min2):
        wd_sector_min2 = np.nan
    if "none" in str(wd_sector_max2):
        wd_sector_max2 = np.nan

    files_WC_DBS = []
    files_WC_VAD = []
    files_WC_vr = []

    for i in years:
        print(i)
        for j in months:
            print(j)
            for k in days:
                print(k)
                # Find all lidar files that correspond to this date
                files_found_WC = glob.glob(
                    lidar_directory
                    + "*"
                    + str(i)
                    + str(j).zfill(2)
                    + str(k).zfill(2)
                    + "*"
                )
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

    # Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
    # For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
    # not be used in a power performance test. In this case, all measurements come from the lidar data.
    for i in range(len(files_WC_DBS)):
        print(i)
        wd = np.load(files_WC_DBS[i], allow_pickle=True)["wd"]
        U = np.load(files_WC_DBS[i], allow_pickle=True)["U"]
        time_all.append(np.load(files_WC_DBS[i], allow_pickle=True)["time"].item())
        if (
            ~(wd.any() >= wd_sector_min1 and wd.any() < wd_sector_max1)
            and ~(wd.any() >= wd_sector_min2 and wd.any() < wd_sector_max2)
            and U.any() >= U_min
            and U.any() < U_max
        ):
            TI_WC_orig_all[i] = (
                np.sqrt(np.load(files_WC_DBS[i], allow_pickle=True)["u_var"])
                / np.load(files_WC_DBS[i], allow_pickle=True)["U"]
            ) * 100

    TI_WC_orig_all = np.array(TI_WC_orig_all)

    # Convert time from datetime format to a normal timestamp.
    # Timestamp is in UTC and corresponds to start of 10-min. averaging period.
    timestamp_10min_all = []

    for i in time_all:
        timestamp_10min_all.append(datetime.datetime.strftime(i, "%Y/%m/%d %H:%M"))

    with open(main_directory + "L_TERRA_adjected_TI_WC.csv", "w") as fp:
        a = csv.writer(fp, delimiter=",")
        data = [["Timestamp (UTC)", "Original TI (%)", "Corrected TI (%)"]]
        a.writerows(data)

    # Initialize arrays for the new lidar TI after each adjustment has been applied
    TI_WC_noise_all = np.zeros(len(files_WC_DBS))
    TI_WC_noise_all[:] = np.nan
    TI_WC_vol_avg_all = np.zeros(len(files_WC_DBS))
    TI_WC_vol_avg_all[:] = np.nan
    TI_WC_var_contamination_all = np.zeros(len(files_WC_DBS))
    TI_WC_var_contamination_all[:] = np.nan

    p_all = np.zeros(len(files_WC_DBS))
    p_all[:] = np.nan

    for i in range(len(files_WC_DBS)):
        print(i)
        if ~np.isnan(TI_WC_orig_all[i]):
            p_all[i] = np.load(files_WC_DBS[i], allow_pickle=True)["p"]
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
                frequency = 1.0
                file_temp = files_WC_DBS[i]
            else:
                frequency = 1.0 / 4
                file_temp = files_WC_VAD[i]

            u_rot = np.load(file_temp, allow_pickle=True)["u_rot"]
            u_var = np.load(file_temp, allow_pickle=True)["u_var"]
            U = np.load(file_temp)["U"]

            vr_n = np.load(files_WC_vr[i], allow_pickle=True)["vr_n"]
            vr_e = np.load(files_WC_vr[i], allow_pickle=True)["vr_e"]
            vr_s = np.load(files_WC_vr[i], allow_pickle=True)["vr_s"]
            vr_w = np.load(files_WC_vr[i], allow_pickle=True)["vr_w"]
            vert_beam = np.load(files_WC_VAD[i], allow_pickle=True)["w_rot"]

            wd = np.load(files_WC_DBS[i], allow_pickle=True)["wd"]

            # Apply noise adjustment and calculate variance
            if "none" in mode_noise:
                u_var_noise = u_var
            else:
                u_var_noise = lidar_processing_noise(
                    u_rot, frequency, mode_ws, mode_noise
                )

            TI_WC_noise_all[i] = (np.sqrt(u_var_noise) / U) * 100

            # Estimate loss of variance due to volume averaging
            if "none" in mode_vol:
                u_var_diff = 0.0
            else:
                try:
                    u_var_diff = lidar_processing_vol_averaging(
                        u_rot, frequency, mode_ws, mode_vol
                    )
                except:
                    u_var_diff = 0.0

            u_var_vol = u_var_noise + u_var_diff
            TI_WC_vol_avg_all[i] = (np.sqrt(u_var_vol) / U) * 100

            # Estimate increase in variance due to variance contamination
            if "none" in mode_contamination:
                u_var_diff = 0.0
            else:
                u_var_diff = lidar_processing_var_contam(
                    vr_n,
                    vr_e,
                    vr_s,
                    vr_w,
                    vert_beam,
                    wd,
                    U,
                    height_needed,
                    1.0 / 4,
                    62.0,
                    mode_contamination,
                )

                try:
                    if np.isnan(u_var_diff).any():
                        u_var_diff = 0.0
                except:
                    u_var_diff = 0.0

            u_var_contam = u_var_vol - u_var_diff
            TI_WC_var_contamination_all[i] = (np.sqrt(u_var_contam) / U) * 100

    # Extract TI values and timestamps for all times when corrected TI value is valid
    mask = ~np.isnan(TI_WC_var_contamination_all)

    timestamp_10min_all = list(compress(timestamp_10min_all, mask))
    TI_WC_orig_all = TI_WC_orig_all[mask]
    TI_WC_var_contamination_all = TI_WC_var_contamination_all[mask]

    # Reduce number of decimal places in output TI data
    TI_orig_temp = ["%0.2f" % i for i in TI_WC_orig_all]
    TI_adjected_temp = ["%0.2f" % i for i in TI_WC_var_contamination_all]

    # Write out timestamp, original lidar TI, and corrected lidar TI
    with open(main_directory + "L_TERRA_adjected_TI_WC.csv", "a") as fp:
        a = csv.writer(fp, delimiter=",")
        data = np.vstack(
            [timestamp_10min_all, TI_orig_temp, TI_adjected_temp]
        ).transpose()
        a.writerows(data)

    print("finished")
    sys.exit()
    # Merge ^ LTERRA TI result with timestamps, inputdata on timestamp
    lterraCorrected = pd.read_csv(
        os.path.join(
            "/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA",
            "L_Terra_adjected_TI_WC.csv",
        )
    )
    fullData = inputdata
    fullData["Timestamp"] = Timestamp
    fullData = pd.merge(fullData, lterraCorrected, on="Timestamp")
    fullData = fullData.drop(columns=["Timestamp"])

    # save orig TI to inject into input data
    fullData["orig_RSD_TI"] = inputdata["RSD_TI"]
    fullData.to_csv(
        os.path.join(
            "/Volumes/New P/DataScience/CFARS/NRGSeptLTERRA", "DataWithOrigTI.csv"
        )
    )
    # make new TI RSD_TI, make old RSD_TI the Orig_RSD_TI
    # inputdata['RSD_TI'] =

    # apply ML model
    print("Applying Adjustment Method: SS-LTERRA_1HZ")
    inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(inputdata.copy())
    lm_adj["sensor"] = sensor
    lm_adj["height"] = height
    lm_adj["adjustment"] = "SS_LTERRA_1HZ"
    adjustment_name = "SS_LTERRA_1HZ"
    baseResultsLists = populate_resultsLists(
        baseResultsLists, "", adjustment_name, lm_adj, inputdata_adj, Timestamps, method
    )

    if RSDtype["Selection"][0:4] == "Wind":
        print("Applying Adjustment Method: SS-LTERRA_1HZ by stability class (TKE)")
        # stability subset output for primary height (all classes)
        ResultsLists_class = initialize_resultsLists("class_")
        className = 1
        for item in All_class_data:
            inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(
                item[primary_idx].copy()
            )
            lm_adj["sensor"] = sensor
            lm_adj["height"] = height
            lm_adj["adjustment"] = str(
                "SS_LTERRA_1HZ" + "_" + "class_" + str(className)
            )
            adjustment_name = str("SS_LTERRA_1HZ" + "_TKE_" + str(className))
            ResultsLists_class = populate_resultsLists(
                ResultsLists_class,
                "class_",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                Timestamps,
                method,
            )
            className += 1
        ResultsList_stability = populate_resultsLists_stability(
            ResultsLists_stability, ResultsLists_class, ""
        )
    if RSD_alphaFlag:
        print(
            "Applying Adjustment Method: SS-LTERRA_1HZ by stability class Alpha w/ RSD"
        )
        ResultsLists_class_alpha_RSD = initialize_resultsLists("class_alpha_RSD")
        className = 1
        for item in All_class_data_alpha_RSD:
            inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(item.copy())
            lm_adj["sensor"] = sensor
            lm_adj["height"] = height
            lm_adj["adjustment"] = str(
                "SS-LTERRA_1HZ" + "_" + "class_" + str(className)
            )
            adjustment_name = str("SS_LTERRA_1HZ" + "_alphaRSD_" + str(className))
            ResultsLists_class_alpha_RSD = populate_resultsLists(
                ResultsLists_class_alpha_RSD,
                "class_alpha_RSD",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                Timestamps,
                method,
            )
            className += 1
        ResultsLists_stability_alpha_RSD = populate_resultsLists_stability(
            ResultsLists_stability_alpha_RSD, ResultsLists_class_alpha_RSD, "alpha_RSD"
        )
    if cup_alphaFlag:
        print(
            "Applying Adjustment Method: SS-LTERRA_1HZ by stability class Alpha w/cup"
        )
        ResultsLists_class_alpha_Ane = initialize_resultsLists("class_alpha_Ane")
        className = 1
        for item in All_class_data_alpha_Ane:
            inputdata_adj, lm_adj, m, c = perform_SS_LTERRA_S_ML_adjustment(item.copy())
            lm_adj["sensor"] = sensor
            lm_adj["height"] = height
            lm_adj["adjustment"] = str(
                "SS_LTERRA_1HZ" + "_" + "class_" + str(className)
            )
            emptyclassFlag = False
            adjustment_name = str("SS_LTERRA_1HZ" + "_alphaCup_" + str(className))
            ResultsLists_class_alpha_Ane = populate_resultsLists(
                ResultsLists_class_alpha_Ane,
                "class_alpha_Ane",
                adjustment_name,
                lm_adj,
                inputdata_adj,
                Timestamps,
                method,
            )
            className += 1
        ResultsLists_stability_alpha_Ane = populate_resultsLists_stability(
            ResultsLists_stability_alpha_Ane, ResultsLists_class_alpha_Ane, "alpha_Ane"
        )
