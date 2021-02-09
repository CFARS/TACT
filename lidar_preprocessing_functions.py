# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:06:11 2015

@author: jnewman
"""

#Functions for initial processing of WindCube data before TI correction is applied.
from functools import reduce
def import_WC_file(filename):
    encoding_from='iso-8859-1'
    encoding_to='UTF-8'
    #Reads in WINDCUBE .rtd file and outputs raw u, v, and w components, measurement heights, and timestamp.

    #Inputs
    #filename: WINDCUBE v2 .rtd file to read

    #Outputs
    #u_sorted, v_sorted, w_sorted: Raw u, v, and w values from all measurement heights
    #heights: Measurement heights from file
    #time_datenum_sorted: All timestamps in datetime format    
    
    import numpy as np
    from datetime import datetime
    
    #Read in row containing heights (either row 38 or 39) and convert heights to a set of integers. 
    inp = open(filename,encoding=encoding_from).readlines()
    height_array = str.split(inp[38])
    heights_temp = height_array[2:]
    
    if len(heights_temp) == 0:
        height_array = str.split(inp[39])
        heights_temp = height_array[2:]
        
    
    heights = [int(i) for i in heights_temp]

    #Read in timestamps. There will be either 41 or 42 headerlines.    
    num_rows = 41
    timestamp = np.loadtxt(filename,encoding=encoding_from, delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
    
    try:
        datetime.strptime(timestamp[0],"%Y/%m/%d %H:%M:%S.%f")
    except:
        num_rows = 42
        timestamp = np.loadtxt(filename, encoding=encoding_from,delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
        
    #Convert timestamps to Python datetime format. Some timestamps may be blank and will raise an error during the
    #datetime conversion. The rows corresponding to these bad timestamps are recorded. 
    time_datenum_temp = []
    bad_rows = []
    for i in range(0,len(timestamp)):
        try:
            time_datenum_temp.append(datetime.strptime(timestamp[i],"%Y/%m/%d %H:%M:%S.%f")) 
        except:
            bad_rows.append(i)
    
    #If bad timestamps are detected, an error message is output to the screen and all timestamps including and following
    #the bad timestamp are deleted. The rows corresponding to these timestamps are categorized as footer lines and are not 
    #used when reading in the velocity data.        
    if(bad_rows):
        print(filename,': Issue reading timestamp')
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
    else:
        footer_lines = 0


        
    #Create column of NaNs for measurement heights that raise an error.
    v_nan = np.empty(len(time_datenum_temp))
    v_nan[:] = np.nan
    
    u = []
    v = []
    w = []
    
    #Read in values of u, v, and w one measurement height at a time. Definitions of the wind components are as follows:
    #u is east-west wind (u > 0 means wind is coming from the west)
    #v is north-south wind (v > 0 means wind is coming from the south)
    #w is vertical wind (w > 0 means wind is upward)
    for i in range(1,len(heights)+1):
        try:
            u.append(-np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t',usecols=(i*9 + 1),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
            v.append(-np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(i*9),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
            w.append(-np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(i*9 + 2),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
        except:
            u.append(v_nan)
            v.append(v_nan)
            w.append(v_nan)
            
            
    u  = np.array(u).transpose()
    v  = np.array(v).transpose()
    w  = np.array(w).transpose()
  
  
    #Check to make sure all timestamps follow the initial timestamp. If a particular timestamp is earlier than the first 
    #timestamp in the data file, this row is marked as a bad row and removed from the data.
    bad_rows = []
    for i in range(1,len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)
            
    if(bad_rows):
        print(filename,': Issue with timestamp order')
        u = np.delete(u,bad_rows,axis=0)
        v = np.delete(v,bad_rows,axis=0)
        w = np.delete(w,bad_rows,axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,axis=0)
        timestamp = np.delete(timestamp,axis=0)
    
    #Sort data by timestamp to ensure that variables are in correct temporal order.        
    time_datenum_sorted = np.array([time_datenum_temp[i] for i in np.argsort(time_datenum_temp)])
    u_sorted = np.array([u[i,:] for i in np.argsort(time_datenum_temp)])
    v_sorted = np.array([v[i,:] for i in np.argsort(time_datenum_temp)])
    w_sorted = np.array([w[i,:] for i in np.argsort(time_datenum_temp)])
    
    return u_sorted,v_sorted,w_sorted,heights,time_datenum_sorted
    
def import_WC_file_VAD(filename,height_needed):
    #Reads in WINDCUBE .rtd file and performs VAD technique at desired height. Outputs u, v, and w values from VAD technique,
    #w values from vertical beam, measurement heights, and timestamp.

    #Inputs
    #filename: WINDCUBE v2 .rtd file to read
    #height_needed: Height where VAD analysis should be performed

    #Outputs
    #u_VAD, v_VAD, w_VAD: u, v, and w values from VAD fit at height_needed
    #vert_beam: Radial velocity from vertical beam at height_needed
    #time_datenum: Timestamps corresponding to the start of each scan in datetime format    
    #time_datenum_vert_beam: Timestamps corresponding to vertical beam position in datetime format

    import numpy as np
    from scipy.optimize import curve_fit
    from lidar_preprocessing_functions import VAD_func
    from datetime import datetime
    from lidar_preprocessing_functions import min_diff
    inp = open(filename, encoding='iso-8859-1').readlines()
    height_array = str.split(inp[38])
    heights_temp = height_array[2:]
    
    if len(heights_temp) == 0:
        height_array = str.split(inp[39])
        heights_temp = height_array[2:]
    
    heights = [int(i) for i in heights_temp]
        
    height_needed_index = min_diff(heights,height_needed,6.1)  
    
    num_rows = 41
    timestamp = np.loadtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
    
    try:
        datetime.strptime(timestamp[0],"%Y/%m/%d %H:%M:%S.%f")
    except:
        num_rows = 42
        timestamp = np.loadtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
    
    time_datenum_temp = []
    bad_rows = []
    #Create list of rows where timestamp cannot be converted to datetime 
    for i in range(0,len(timestamp)):
        try:
            time_datenum_temp.append(datetime.strptime(timestamp[i],"%Y/%m/%d %H:%M:%S.%f")) 
        except:
            bad_rows.append(i)
    
    #Delete all timestamp and datetime values from first bad row to end of dataset         
    if(bad_rows):
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
    else:
        footer_lines = 0
    
    #Skip lines that correspond to bad data
    az_angle = np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(1,),dtype=str, unpack=True,skip_header=num_rows,skip_footer=footer_lines)
    
    vr_nan = np.empty(len(time_datenum_temp))
    vr_nan[:] = np.nan
    
    vr = []
    for i in range(1,len(heights)+1):
        try:
            vr.append(-np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(i*9 -4),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
        except:
            vr.append(vr_nan)

                
        
    vr  = np.array(vr)    
    vr = vr.transpose()   
    
    bad_rows = []
    #Find rows where time decreases instead of increasing 
    for i in range(1,len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)
    
    #Delete rows where time decreases instead of increasing         
    if(bad_rows):
        vr = np.delete(vr,bad_rows,axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,bad_rows,axis=0)
        timestamp = np.delete(timestamp,bad_rows,axis=0)
        az_angle = np.delete(az_angle,bad_rows,axis=0)
    
    #Sort timestamp, vr, and az angle in order of ascending datetime value        
    timestamp_sorted = [timestamp[i] for i in np.argsort(time_datenum_temp)]
    vr_sorted = np.array([vr[i,:] for i in np.argsort(time_datenum_temp)])
    az_angle_sorted = np.array([az_angle[i] for i in np.argsort(time_datenum_temp)])

    
    
    vert_beam = []
    vr_temp = []
    az_temp = []
    timestamp_az = []
    timestamp_vert_beam = []
    #Separate vertical beam values (where az angle = "V") from off-vertical beam values
    for i in range(0,len(az_angle_sorted)):
        if "V" in az_angle_sorted[i]:
            vert_beam.append(vr_sorted[i,height_needed_index])
            timestamp_vert_beam.append(timestamp_sorted[i])
        else:
            vr_temp.append(vr_sorted[i,height_needed_index])
            az_temp.append(float(az_angle_sorted[i]))
            timestamp_az.append(timestamp_sorted[i])
            
    vr_temp = np.array(vr_temp)     
    elevation = 62
    u_VAD = []
    v_VAD = []
    w_VAD = []
    timestamp_VAD = []


    
    #Perform a VAD fit on each full scan
    print(len(az_temp)/4)
    for i in range(0,int(len(az_temp)/4)):
            x_vals = np.array(az_temp[i*4 + 1:i*4 + 5])
            y_vals= np.array(vr_temp[i*4 + 1:i*4 + 5])
            if len(y_vals[np.isnan(y_vals)]) == 0:
                #Initial guesses for the VAD fit parameters 
                p0 = np.array([(np.max(y_vals)-np.min(y_vals))/2,2*np.pi,np.nanmean(y_vals)])
                popt, pcov = curve_fit(VAD_func, x_vals.ravel(), y_vals.ravel(),p0.ravel())
                ws_temp = popt[0]/np.cos(np.radians(elevation))
                wd_temp = np.degrees(popt[1]-np.pi)
                if wd_temp > 360:
                    wd_temp-= 360
                u_VAD.append(np.sin(np.radians(wd_temp) - np.pi)*ws_temp)
                v_VAD.append(np.cos(np.radians(wd_temp) - np.pi)*ws_temp)
                w_VAD.append(popt[2]/np.sin(np.radians(elevation)))
            else:
                u_VAD.append(np.nan)
                v_VAD.append(np.nan)
                w_VAD.append(np.nan)
            timestamp_VAD.append(timestamp_az[i*4 +1])
    
    #Convert VAD and vertical beam timestamps to datetime format     
    
    time_datenum = []
      
    for i in range(0,len(timestamp_VAD)):
        time_datenum.append(datetime.strptime(timestamp_VAD[i],"%Y/%m/%d %H:%M:%S.%f")) 
        
    time_datenum_vert_beam = []
      
    for i in range(0,len(timestamp_vert_beam)):
        time_datenum_vert_beam.append(datetime.strptime(timestamp_vert_beam[i],"%Y/%m/%d %H:%M:%S.%f"))        
       
            
    return np.array(u_VAD),np.array(v_VAD),np.array(w_VAD),np.array(vert_beam)[:,0],\
    np.array(time_datenum),np.array(time_datenum_vert_beam)
        
            
            
def VAD_func(az, x1, x2, x3):
    import numpy as np
    return np.array(x3+x1*np.cos(np.radians(az)-x2))   
    
def import_WC_file_vr(filename,height_needed):
    #Reads in WINDCUBE .rtd file and extracts off-vertical radial wind speed components at desired height.

    #Inputs
    #filename: WINDCUBE v2 .rtd file to read
    #height_needed: Height where off-vertical measurements should be extracted

    #Outputs
    #vr_n,vr_e,vr_s,vr_w: Time series from north-, east-, south-, and west-pointing beams, 
    #respectively, at height_needed
    #time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w: Timestamps corresponding to 
    #north-, east-, south-, and west-pointing beams, respectively, in datetime format

    import numpy as np
    from datetime import datetime
    from lidar_preprocessing_functions import min_diff
    inp = open(filename,encoding='iso-8859-1').readlines()
    height_array = str.split(inp[38])
    heights_temp = height_array[2:]
    
    if len(heights_temp) == 0:
        height_array = str.split(inp[39])
        heights_temp = height_array[2:]
    
    heights = [int(i) for i in heights_temp]
        
    height_needed_index = min_diff(heights,height_needed,6.1)
    
    num_rows = 41
    timestamp = np.loadtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
    
    try:
        datetime.strptime(timestamp[0],"%Y/%m/%d %H:%M:%S.%f")
    except:
        num_rows = 42
        timestamp = np.loadtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(0,),dtype=str, unpack=True,skiprows=num_rows)
         
    
    time_datenum_temp = []
    bad_rows = []
    #Create list of rows where timestamp cannot be converted to datetime 
    for i in range(0,len(timestamp)):
        try:
            time_datenum_temp.append(datetime.strptime(timestamp[i],"%Y/%m/%d %H:%M:%S.%f")) 
        except:
            bad_rows.append(i)
    
    #Delete all timestamp and datetime values from first bad row to end of dataset         
    if(bad_rows):
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
    else:
        footer_lines = 0
    
    #Skip lines that correspond to bad data
    az_angle = np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(1,),dtype=str, unpack=True,skip_header=num_rows,skip_footer=footer_lines)
    
    vr_nan = np.empty(len(time_datenum_temp))
    vr_nan[:] = np.nan
    
    vr = []
    for i in range(1,len(heights)+1):
        try:
            vr.append(-np.genfromtxt(filename,encoding='iso-8859-1', delimiter='\t', usecols=(i*9 -4),dtype=None,skip_header=num_rows,skip_footer=footer_lines))
        except:
            vr.append(vr_nan)

                
        
    vr  = np.array(vr)    
    vr = vr.transpose()   
    
    bad_rows = []
    #Find rows where time decreases instead of increasing 
    for i in range(1,len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)
    
    #Delete rows where time decreases instead of increasing         
    if(bad_rows):
        vr = np.delete(vr,bad_rows,axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,bad_rows,axis=0)
        timestamp = np.delete(timestamp,bad_rows,axis=0)
        az_angle = np.delete(az_angle,bad_rows,axis=0)
    
    #Sort timestamp, vr, and az angle in order of ascending datetime value        
    timestamp_sorted = [timestamp[i] for i in np.argsort(time_datenum_temp)]
    vr_sorted = np.array([vr[i,:] for i in np.argsort(time_datenum_temp)])
    az_angle_sorted = np.array([az_angle[i] for i in np.argsort(time_datenum_temp)])
        
    vr_sorted  = np.array(vr_sorted)         

    vr_temp = []
    az_temp = []  
    timestamp_az = []
    vert_beam = []
    timestamp_vert_beam = []
    #Separate vertical beam values (where az angle = "V") from off-vertical beam values
    for i in range(0,len(az_angle_sorted)):
        if "V" in az_angle_sorted[i]:
            vert_beam.append(vr_sorted[i,:])
            timestamp_vert_beam.append(timestamp_sorted[i])
        else:
            vr_temp.append(vr_sorted[i,:])
            az_temp.append(float(az_angle_sorted[i]))
            timestamp_az.append(timestamp_sorted[i])
            
    vr_temp = np.array(vr_temp)
    az_temp = np.array(az_temp)
    
    #Extract data for north-, east-, south-, and west-pointing beams at height of interest 
    vr_n = vr_temp[az_temp==0,height_needed_index]
    vr_e = vr_temp[az_temp==90,height_needed_index]
    vr_s = vr_temp[az_temp==180,height_needed_index]
    vr_w = vr_temp[az_temp==270,height_needed_index]
    

    #Convert timestamps to datetime format     
    time_datenum = []
      
    for i in range(0,len(timestamp_az)):
        time_datenum.append(datetime.strptime(timestamp_az[i],"%Y/%m/%d %H:%M:%S.%f"))
        
        
    time_datenum = np.array(time_datenum)
    time_datenum_n = time_datenum[az_temp==0]   
    time_datenum_e = time_datenum[az_temp==90] 
    time_datenum_s = time_datenum[az_temp==180] 
    time_datenum_w = time_datenum[az_temp==270] 
            
    return vr_n,vr_e,vr_s,vr_w,time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w   
    
def import_ZephIR_file(filename):
    #Reads in ZephIR .ZPH file and outputs raw u, v, and w components, measurement heights, and timestamp.

    #filename: High-resolution ZephIR. ZPH file to read

    #Outputs
    #u_sorted, v_sorted, w_sorted: u, v, and w values from all measurement heights
    #heights: Measurement heights from file
    #time_datenum_sorted: All timestamps in datetime format    
    
    import numpy as np
    from datetime import datetime
    inp = open(filename,'iso-8859-1').readlines()
    #Read in measurement heights and convert to integers
    height_array = str.split(inp[0])[33:44]
    heights = [int(i[:-1]) for i in height_array]
    
    #Read in timestamps
    num_rows = 2
    timestamp = np.loadtxt(filename,encoding='iso-8859-1', delimiter=',', usecols=(1,),dtype=str, unpack=True,skiprows=num_rows)
    
        
    
    time_datenum_temp = []
    bad_rows = []
    #Create list of rows where timestamp cannot be converted to datetime
    for i in range(0,len(timestamp)):
        try:
            time_datenum_temp.append(datetime.strptime(timestamp[i],"%d/%m/%Y %H:%M:%S")) 
        except:
            bad_rows.append(i)
            
    #Delete all timestamp and datetime values from first bad row to end of dataset          
    if(bad_rows):
        print(filename,': Issue reading timestamp')
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp,range(bad_rows[0],len(timestamp)),axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,range(bad_rows[0],len(time_datenum_temp)),axis=0)
    else:
        footer_lines = 0


        
    
    v_nan = np.empty(len(time_datenum_temp))
    v_nan[:] = np.nan
    
    #Read in wind direction, horizontal wind speed, and vertical wind speed estimated from VAD fit at each measurement height
    for i in range(0,len(heights)):
        try:
            if i == 0:
                wd = np.genfromtxt(filename,encoding='iso-8859-1', delimiter=',',usecols=(19 + i*3),dtype=None,skip_header=num_rows,skip_footer=footer_lines)
                ws = np.genfromtxt(filename,encoding='iso-8859-1', delimiter=',', usecols=(20 + i*3),dtype=None,skip_header=num_rows,skip_footer=footer_lines)
                w = np.genfromtxt(filename,encoding='iso-8859-1', delimiter=',', usecols=(21 + i*3),dtype=None,skip_header=num_rows,skip_footer=footer_lines)
            else:
                wd = np.vstack((wd,np.genfromtxt(filename,encoding='iso-8859-1', delimiter=',',usecols=(19 + i*3),dtype=None,skip_header=num_rows,skip_footer=footer_lines)))
                ws = np.vstack((ws,np.genfromtxt(filename,encoding='iso-8859-1', delimiter=',', usecols=(20 + i*3),dtype=None,skip_header=num_rows,skip_footer=footer_lines)))
                w = np.vstack((w,np.genfromtxt(filename,encoding='iso-8859-1', delimiter=',', usecols=(21 + i*3),dtype=None,skip_header=num_rows,skip_footer=footer_lines)))
        except:
            if i == 0:
                wd = v_nan
                ws = v_nan
                w = v_nan
            else:
                wd = np.vstack((wd,v_nan))
                ws = np.vstack((ws,v_nan))
                w = np.vstack((w,v_nan))   
            print(filename,': Issue with wind speed')
            
    wd[wd==9999] = np.nan    
    ws[ws==9999] = np.nan  
    w[w==9999] = np.nan   
    
    u = np.array(np.sin(np.radians(wd - 180))*ws).transpose()
    v = np.array(np.cos(np.radians(wd - 180))*ws).transpose() 
    w  = np.array(w).transpose()     

    bad_rows = []
    #Find rows where time decreases instead of increasing 
    for i in range(1,len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)
            
    #Delete rows where time decreases instead of increasing        
    if(bad_rows):
        print(filename,': Issue with timestamp order')
        u = np.delete(u,bad_rows,axis=0)
        v = np.delete(v,bad_rows,axis=0)
        w = np.delete(w,bad_rows,axis=0)
        time_datenum_temp = np.delete(time_datenum_temp,axis=0)
        timestamp = np.delete(timestamp,axis=0)
    
    #Sort data in order of ascending datetime value        
    time_datenum_sorted = np.array([time_datenum_temp[i] for i in np.argsort(time_datenum_temp)])
    u_sorted = np.array([u[i,:] for i in np.argsort(time_datenum_temp)])
    v_sorted = np.array([v[i,:] for i in np.argsort(time_datenum_temp)])
    w_sorted = np.array([w[i,:] for i in np.argsort(time_datenum_temp)])
    
    return u_sorted,v_sorted,w_sorted,heights,time_datenum_sorted
    
def interp_ts(ts,time_datenum,interval): 
    #Interpolates time series ts with timestamps time_datenum to a grid with constant temporal spacing of "interval"

    #Inputs
    #ts: Time series for interpolation
    #time_datenum: Original timestamps for time series in datetime format
    #interval: Temporal interval to use for interpolation

    #Outputs
    #ts_interp: Interpolated time series
    #time_interp: Timestamps of interpolated time series in datetime format

    import numpy as np
    from datetime import datetime
    import calendar as cal

    #Convert timestamps to unix time (seconds after 1970 01-01) as it's easier to perform the interpolation
    unix_time = []
    for i in range(0,len(time_datenum)):
        unix_time.append(cal.timegm(datetime.timetuple(time_datenum[i])) + (time_datenum[i].microsecond/1e6))
        
    unix_time = np.array(unix_time)
    
    #Select the start and end time for the interpolation
    
    #The starting minute value of the interpolation should be the next multiple of 10
    if time_datenum[0].minute%10 == 0:
        start_minute = str((time_datenum[0].minute//10)*10)
    else:
        start_minute = str((time_datenum[0].minute//10 + 1)*10)
        
    start_hour = str(time_datenum[0].hour)

    if int(start_minute) == 60:
        start_minute = '00'
        start_hour = str(time_datenum[0].hour + 1)
        
    
    end_hour = str(time_datenum[-1].hour)    
    
    #The ending minute value of the interpolation should end with a 9
    if (time_datenum[-1].minute-9)%10 == 0:
        end_minute = str((time_datenum[-1].minute//10*10) + 9)
    else:
        end_minute = str((time_datenum[-1].minute//10)*10 - 1)
    
    
    if int(end_minute) < 0:
        end_minute = '59'
        end_hour = str(time_datenum[-1].hour - 1)  
    
    #Convert start and end times into unix time and get interpolation times in unix time 
    timestamp_start = str(time_datenum[0].year) + "/" + str(time_datenum[0].month) + "/" + str(time_datenum[0].day) + \
    " " + start_hour + ":" + start_minute + ":00"

    time_datenum_start = datetime.strptime(timestamp_start,"%Y/%m/%d %H:%M:%S")
    unix_time_start = cal.timegm(datetime.timetuple(time_datenum_start)) 

    timestamp_end = str(time_datenum[-1].year) + "/" + str(time_datenum[-1].month) + "/" + str(time_datenum[-1].day) + \
    " " + end_hour + ":" + end_minute + ":59"

    time_datenum_end = datetime.strptime(timestamp_end,"%Y/%m/%d %H:%M:%S")
    unix_time_end = cal.timegm(datetime.timetuple(time_datenum_end)) 

    time_interp_unix = np.arange(unix_time_start,unix_time_end+1,interval)
    
    #Interpolate time series    
    
    ts_interp = [] 
    
    #If more than 75% of the data are valid, perform interpolation using only non-NaN data. (Every fifth point of the 
    #u and v data will be NaNs because of the vertically pointing beam.) 
    if float(len(ts[~np.isnan(ts)])/float(len(ts))) > 0.75:
        ts_temp = ts[~np.isnan(ts)]
        time_temp = unix_time[~np.isnan(ts)]
    else:
        ts_temp = ts
        time_temp = unix_time
    ts_interp = np.interp(time_interp_unix,time_temp,ts_temp)
    
    #If several points in a row have the same value, set these points to NaN. This can occur when the interpolation is 
    #performed on a dataset with one valid value surrounded by several NaNs.
    for i in range(2,len(ts_interp)-2):
        if ts_interp[i-2] == ts_interp[i] and ts_interp[i+2] == ts_interp[i]:
            ts_interp[i-2:i+2] = np.nan
        
    time_interp = [datetime.utcfromtimestamp(int(i) + round(i-int(i),10)) for i in time_interp_unix]

    return np.transpose(ts_interp),time_interp   

def min_diff(array_orig,array_to_find,tol):
    #Finds indices in array_orig that correspond to values closest to numbers in array_to_find with tolerance tol 

    #Inputs
    #array_orig: Original array where you want to find matching values
    #array_to_find: Array of numbers to find in array_orig
    #tol: Tolerance to find matching value

    #Outputs
    #found_indices: Indices corresponding to matching values. If no values matched with desired tolerance, index will be filled by NaN.     
    
    import numpy as np
    found_indices = []
    if not np.shape(array_to_find):
        array_to_find = [array_to_find]
    for i in array_to_find:
        min_difference = tol
        found_index_temp = np.nan
        for j in range(0,len(array_orig)):
            diff_temp = abs(i-array_orig[j])
            if diff_temp < min_difference:
                min_difference = diff_temp
                found_index_temp = j
        found_indices.append(found_index_temp)
    return np.array(found_indices)  

def get_10min_mean_ws_wd(u,v,time,frequency):
    #Calculates the 10-min. scalar average wind speed and wind direction at all measurement heights

    #Inputs
    #u: East-west velocity time series
    #v: North-south velocity time series
    #time: Timestamps in datetime format
    #frequency: Sampling frequency of velocity data

    #Outputs
    #U: 10-min. mean horizontal wind speeds
    #wd: 10-min. mean wind direction
    #time_datenum_10min: Timestamp corresponding to the start of each 10-min. averaging period 

    import numpy as np
    ten_min_count = int(frequency*60*10)
    U = []
    wd = []
    time_datenum_10min = []
    
    for i in np.arange(0,len(u)-ten_min_count+1,ten_min_count):
        U_height = []
        wd_height = []
        #10-min. window of data
        if len(np.shape(u)) > 1:
            u_temp = u[i:i+ten_min_count,:]
            v_temp = v[i:i+ten_min_count,:]
        else:
            u_temp = u[i:i+ten_min_count]
            v_temp = v[i:i+ten_min_count]
        for j in range(np.shape(u_temp)[1]):
            U_height.append(np.nanmean((u_temp[:,j]**2 + v_temp[:,j]**2)**0.5,axis=0));
            u_bar = np.nanmean(u_temp[:,j])
            v_bar = np.nanmean(v_temp[:,j])
            wd_height.append((180./np.pi)*(np.arctan2(u_bar,v_bar) + np.pi))
        U.append(U_height)
        wd.append(wd_height)
        time_datenum_10min.append(time[i])
    return np.array(U),np.array(wd),time_datenum_10min

def get_10min_shear_parameter(U,heights,height_needed):
    #Calculates the shear parameter for every 10-min. period of data by fitting power law equation to 
    #10-min. mean wind speeds 

    #Inputs
    #U: 10-min. mean horizontal wind speed at all measurement heights
    #heights: Measurement heights 
    #height_needed: Height where TI is being extracted - values used to calculate shear parameter 
    #should be centered around this height

    #Outputs
    #p: 10-min. values of shear parameter


    import numpy as np
    from functools import reduce
    from lidar_preprocessing_functions import min_diff
    import warnings
    p = []
    
    #Set heights for calculation of shear parameter and find corresponding indices
    zprofile = np.arange(0.5*height_needed,1.5*height_needed + 10,10)
    height_indices = np.unique(min_diff(heights,zprofile,5))
    height_indices = height_indices[~np.isnan(height_indices)]
    
    #Arrays of height and mean wind speed to use for calculation    
    heights_temp = np.array([heights[int(i)] for i in height_indices])
    U_temp = np.array([U[:,int(i)] for i in height_indices])
    
    mask = [~np.isnan(U_temp)]
    mask = reduce(np.logical_and, mask)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
    
    #For each set of 10-min. U values, use linear fit to determine value of shear parameter     
    for i in range(0,len(U)):
        try:
            try:
                p_temp = np.polyfit(np.log(heights_temp[mask[:,i]]),np.log(U_temp[mask[:,i],i]),1)
                p.append(p_temp[0])
            except np.RankWarning: 
                p.append(np.nan) 
        except:
            p.append(np.nan) 
    return np.array(p)  

    
def rotate_ws(u,v,w,frequency):
    #Performs coordinate rotation according to Eqs. 22-29 in Wilczak et al. (2001)
    #Reference: Wilczak, J. M., S. P. Oncley, and S. A. Stage, 2001: Sonic anemometer tilt correction algorithms. 
    #Bound.-Layer Meteor., 99, 127–150.

    #Inputs
    #u, v, w: Time series of east-west, north-south, and vertical wind speed components, respectively
    #frequency: Sampling frequency of velocity

    #Outputs
    #u_rot, v_rot, w_rot: Rotated u, v, and w wind speed, with u rotated into the 10-min. mean wind direction and 
    #the 10-min. mean of v and w forced to 0

    import numpy as np
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)
    
    u_rot = []
    v_rot = []
    w_rot = []
    
    #Perform coordinate rotation. First rotation rotates u into the mean wind direction and forces the mean v to 0. 
    #Second rotation forces the mean w to 0. 
    for i in np.arange(0,len(u)-ten_min_count+1,ten_min_count):
        u_temp = u[i:i+ten_min_count]
        v_temp = v[i:i+ten_min_count]
        w_temp = w[i:i+ten_min_count]
        phi_temp = np.arctan2(np.nanmean(v_temp),np.nanmean(u_temp))
        u1_temp = u_temp*np.cos(phi_temp) + v_temp*np.sin(phi_temp)
        v1_temp = -u_temp*np.sin(phi_temp) + v_temp*np.cos(phi_temp)
        w1_temp = w_temp;
        phi_temp2 = np.arctan2(np.nanmean(w1_temp),np.nanmean(u1_temp))
        u_rot.append(u1_temp*np.cos(phi_temp2) + w1_temp*np.sin(phi_temp2))
        v_rot.append(v1_temp)
        w_rot.append(-u1_temp*np.sin(phi_temp2) + w1_temp*np.cos(phi_temp2))
        
    return np.array(u_rot).ravel(),np.array(v_rot).ravel(),np.array(w_rot).ravel()   

def get_10min_var(ts,frequency):
    #Calculates variance for each 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #ts_var: 10-min. variance values from time series

    import numpy as np
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)
    
    ts_var = []

    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        ts_temp = ts[i:i+ten_min_count]
        ts_var.append(np.nanmean((ts_temp-np.nanmean(ts_temp))**2))
        
    return np.array(ts_var)     

def get_10min_spectrum(ts,frequency):
    #Calculate power spectrum for 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #S_A_fast: Spectral power 
    #frequency_fft: Frequencies correspond to spectral power values 

    import numpy as np
    N = len(ts)
    delta_f = float(frequency)/N
    frequency_fft = np.linspace(0,float(frequency)/2,float(N/2)) 
    F_A_fast = np.fft.fft(ts)/N
    E_A_fast = 2*abs(F_A_fast[0:int(N/2)]**2)
    S_A_fast = (E_A_fast)/delta_f 
    return S_A_fast,frequency_fft
    
def get_10min_spectrum_WC_raw(ts,frequency):
    #Calculate power spectrum for 10-min. period

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #S_A_fast: Spectral power 
    #frequency_fft: Frequencies correspond to spectral power values 

    import numpy as np
    N = len(ts)
    delta_f = float(frequency)/N
    frequency_fft = np.linspace(0,float(frequency)/2,float(N/2)) 
    F_A_fast = np.fft.fft(ts)/N
    E_A_fast = 2*abs(F_A_fast[0:int(N/2)]**2)
    S_A_fast = (E_A_fast)/delta_f
    #Data are only used for frequencies lower than 0.125 Hz. Above 0.125 Hz, the 
    #WINDCUBE spectrum calculated using raw data begins to show an artifact. This
    #artifact is due to the recording of the u, v, and w components for every beam 
    #position, which results in repeating components. 
    S_A_fast = S_A_fast[frequency_fft <= 0.125]
    frequency_fft = frequency_fft[frequency_fft <= 0.125]
    return S_A_fast,frequency_fft    
    
def get_10min_covar(ts1,ts2,frequency):
    #Calculate the covariance of two variables 

    #Inputs
    #ts1: Time series of variable 1
    #ts2: Time series of variable 2
    #frequency: Sampling frequency 

    #Outputs
    #ts_covar: 10-min. covariance of variables 1 and 2

    import numpy as np
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)
    
    ts_covar = []

    for i in np.arange(0,len(ts1)-ten_min_count+1,ten_min_count):
        ts_temp1 = ts1[i:i+ten_min_count]
        ts_temp2 = ts2[i:i+ten_min_count]
        mask = [~np.isnan(ts_temp1),~np.isnan(ts_temp2)]
        total_mask = reduce(np.logical_and, mask)
        ts_temp1 = ts_temp1[total_mask]
        ts_temp2 = ts_temp2[total_mask]
        ts_covar.append(np.nanmean((ts_temp1-np.nanmean(ts_temp1))*(ts_temp2-np.nanmean(ts_temp2))))
        
    return np.array(ts_covar)         
