   # -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:57:30 2015

@author: jnewman
"""

#These functions enable processing of WINDCUBE data in each module of L-TERRA

def WC_processing_standard(filename,option,height_needed):
    #Reads in WC data, performs a temporal interpolation, and outputs streamwise wind speed, 10-min. mean wind speeds, 
    #and 10-min. shear parameter (for raw and VAD processing) and radial wind speeds and 10-min. mean wind speeds (for vr 
    #processing).

    #Inputs
    #filename: WINDCUBE v2 .rtd file to read
    #option: raw, VAD, or vr. Raw option reads in 1 Hz output u, v, w values from .rtd file. VAD option performs a VAD fit for each 
    #scan. vr option extracts radial velocity data from off-vertical beam positions. 
    #height_needed: Height of interest where data should be extracted

    #Outputs (varies based on option chosen)
    #u_rot: Time series of streamwise velocity 
    #U: 10-min. Mean horizontal wind speed
    #wd: 10-min. Mean wind direction
    #p: Shear exponent, calculated for every 10 min. of data
    #time_datenum_10min: Timestamp in datetime format
    #w_rot: Time series of vertical wind speed after tilt adjustment has been applied
    #vr_n_interp,vr_e_interp,vr_s_interp, and vr_w_interp: Time series of radial velocity from north-, east-, south-, 
    #and west-pointing beams, respectively.

    encoding_from='iso-8859-1'
    encoding_to='UTF-8'
    
    from lidar_preprocessing_functions import import_WC_file, import_WC_file_VAD,\
    import_WC_file_vr,rotate_ws, get_10min_mean_ws_wd, get_10min_shear_parameter, interp_ts, min_diff
    import numpy as np

    if not "vr" in option:
        if "raw" in option:
            frequency = 1.
            [u,v,w,heights,time_datenum] = import_WC_file(filename)
            
            #Perform a linear interpolation of the u, v, and w wind speed data using a time interval defined by
            #the inverse of the frequency. 
            u_interp = []
            v_interp = []
            w_interp = []
        
        
            for i in range(len(heights)):
                [u_interp_temp,time_interp] = interp_ts(u[:,i],time_datenum,1./frequency)
                [v_interp_temp,time_interp] = interp_ts(v[:,i],time_datenum,1./frequency)
                [w_interp_temp,time_interp] = interp_ts(w[:,i],time_datenum,1./frequency)
                u_interp.append(u_interp_temp)
                v_interp.append(v_interp_temp)
                w_interp.append(w_interp_temp)
            
            u_interp = np.transpose(np.array(u_interp))
            v_interp = np.transpose(np.array(v_interp))
            w_interp = np.transpose(np.array(w_interp))
            
            height_needed_index = min_diff(heights,height_needed,6.1)
            
            #Get the 10-min. mean wind speed, wind direction, and 10-min. timestamps from the dataset
            [U,wd,time_datenum_10min] = get_10min_mean_ws_wd(u_interp,v_interp,time_interp,frequency)
            
            #If the dataset is valid, calculate the 10-min. shear parameter and extract the 10-min. mean wind speed and
            #wind direction at the desired height
            if len(time_datenum_10min) != 0:
                p = get_10min_shear_parameter(U,heights,height_needed)
                U = U[:,height_needed_index][:,0]
                wd = wd[:,height_needed_index][:,0]
            else:
                p = []
                U = []
                wd = []


            [u_rot,v_rot,w_rot] = rotate_ws(u_interp[:,height_needed_index],v_interp[:,height_needed_index],\
            w_interp[:,height_needed_index],frequency)
            
            return u_rot,U,wd,p,time_datenum_10min

        elif "VAD" in option:
            frequency = 1./4
            #Extract the u, v, and w components at desired height calculated from the VAD technique, in addition to values 
            #from the vertically pointing beam. Values are only calculated at one height due to the long processing time for 
            #the VAD technique.              
            [u,v,w,vert_beam,time_datenum,time_datenum_vert_beam] = import_WC_file_VAD(filename,height_needed) 
            
            
            #Perform temporal interpolation
            [u_interp,time_interp] = interp_ts(u,time_datenum,1./frequency)
            [v_interp,time_interp] = interp_ts(v,time_datenum,1./frequency)
            [w_interp,time_interp] = interp_ts(w,time_datenum,1./frequency)
            [vert_beam_interp,time_interp] = interp_ts(vert_beam,time_datenum_vert_beam,1./frequency)

            
            u_interp = np.transpose(np.array(u_interp)).reshape(len(u_interp),1)
            v_interp = np.transpose(np.array(v_interp)).reshape(len(v_interp),1)
            w_interp = np.transpose(np.array(w_interp)).reshape(len(w_interp),1)
            vert_beam_interp = np.transpose(np.array(vert_beam_interp)).reshape(len(vert_beam_interp),1)
            
            #Get 10-min. mean wind speed and mean wind direction at desired height 
            [U,wd,time_datenum_10min] = get_10min_mean_ws_wd(u_interp,v_interp,time_interp,frequency)
            
            #Rotate raw wind speeds, using the vertical beam  output as the w velocity
            [u_rot,v_rot,w_rot] = rotate_ws(u_interp,v_interp,vert_beam_interp,frequency)
            
            return u_rot,U,w_rot
               

    else:
        frequency = 1./4
        #Extract raw data from the off-vertical beams
        [vr_n,vr_e,vr_s,vr_w,time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w] = \
        import_WC_file_vr(filename,height_needed)

        vr_n_interp = []
        vr_e_interp = []
        vr_s_interp = []
        vr_w_interp = []
        
        #Perform temporal interpolation
        [vr_n_interp,time_interp] = interp_ts(vr_n,time_datenum_n,1./frequency)
        [vr_e_interp,time_interp] = interp_ts(vr_e,time_datenum_e,1./frequency)
        [vr_s_interp,time_interp] = interp_ts(vr_s,time_datenum_s,1./frequency)
        [vr_w_interp,time_interp] = interp_ts(vr_w,time_datenum_w,1./frequency)

            
        return vr_n_interp,vr_e_interp,vr_s_interp,vr_w_interp
        
def ZephIR_processing_standard(filename,height_needed):
    #Reads in ZephIR data, performs a temporal interpolation, and outputs streamwise wind speed, 10-min. mean wind speeds, 
    #and 10-min. shear parameter


    #Inputs
    #filename: High-frequency ZephIR .ZPH file to read
    #height_needed: Height of interest where data should be extracted

    #Outputs
    #u_rot: Time series of streamwise velocity 
    #U: 10-min. Mean horizontal wind speed
    #wd: 10-min. Mean wind direction
    #p: Shear exponent, calculated for every 10-min. of data
    #time_datenum_10min: Timestamp in datetime format

    from lidar_preprocessing_functions import import_ZephIR_file, rotate_ws, get_10min_mean_ws_wd, get_10min_shear_parameter, interp_ts,min_diff
    import numpy as np

    frequency = 1./15
    [u,v,w,heights,time_datenum] = import_ZephIR_file(filename)
    
    #Perform a linear interpolation of the u, v, and w wind speed data using a time interval defined by
    #the inverse of the frequency. 
    u_interp = []
    v_interp = []
    w_interp = []


    for i in range(len(heights)):
        [u_interp_temp,time_interp] = interp_ts(u[:,i],time_datenum,1./frequency)
        [v_interp_temp,time_interp] = interp_ts(v[:,i],time_datenum,1./frequency)
        [w_interp_temp,time_interp] = interp_ts(w[:,i],time_datenum,1./frequency)
        u_interp.append(u_interp_temp)
        v_interp.append(v_interp_temp)
        w_interp.append(w_interp_temp)
    
    u_interp = np.transpose(np.array(u_interp))
    v_interp = np.transpose(np.array(v_interp))
    w_interp = np.transpose(np.array(w_interp))
    
    height_needed_index = min_diff(heights,[height_needed],5)
    
    #Get the 10-min. mean wind speed, wind direction, and 10-min. timestamps from the dataset
    [U,wd,time_datenum_10min] = get_10min_mean_ws_wd(u_interp,v_interp,time_interp,frequency)
        
        
    #If the dataset is valid, calculate the 10-min. shear parameter and extract the 10-min. mean wind speed and
    #wind direction at the desired height
    if len(time_datenum_10min) != 0:
        p = get_10min_shear_parameter(U,heights,height_needed)
        U = U[:,height_needed_index][:,0]
        wd = wd[:,height_needed_index][:,0]
    else:
        p = []
        U = []
        wd = []


    [u_rot,v_rot,w_rot] = rotate_ws(u_interp[:,height_needed_index],v_interp[:,height_needed_index],\
    w_interp[:,height_needed_index],frequency)
    
    return u_rot,U,wd,p,time_datenum_10min 
    
def lidar_processing_noise(ts,frequency,mode_ws,mode_noise):
    #Function to apply noise adjustment to time series. Outputs new variance after
    #noise adjustment has been applied. 

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data
    #mode_ws: raw_WC, VAD, or raw_ZephIR 
    #mode_noise: Type of noise adjustment to be applied. Options are spike, lenschow_linear, lenschow_subrange, and lenschow_spectrum.

    #Outputs
    #new_ts_var: New 10-min. variance values after noise adjustment has been applied

    from lidar_noise_removal_functions import lenschow_technique,spike_filter 
    from lidar_preprocessing_functions import get_10min_var

        
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

    
def lidar_processing_vol_averaging(u,frequency,mode_ws,mode_vol):
    #Function to estimate variance lost due to volume/temporal averaging

    #Inputs
    #u: Time series of streamwise wind speed
    #frequency: Sampling frequency of time series
    #mode_ws: raw_WC, VAD, or raw_ZephIR 
    #mode_vol: Type of volume averaging adjustment to be applied. Options are spectral_adjustment_fit and acf.

    #Outputs
    #var_diff: Estimate of loss of streamwise variance due to volume averaging

    from lidar_volume_averaging_functions import spectral_adjustment

    var_diff = spectral_adjustment(u,frequency,mode_ws,mode_vol)

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
    #mode: Type of variance contamination adjustment to be applied. Options are taylor_ws and taylor_var. 

    #Outputs
    #var_diff: Estimate of increase in streamwise variance due to variance contamination

    from lidar_var_contamination_functions import var_adjustment
    var_diff = var_adjustment(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode)
        
    #Set negative values of var_diff to 0 as they would increase the corrected variance
    #Note: This is not the best procedure and should probably be fixed at some point. 
    #It's possible that at times, the change in w across the scanning circle could
    #decrease, rather than increase, the u and v variance. 
    if var_diff < 0:
        var_diff = 0.    
    return var_diff    
    
    
