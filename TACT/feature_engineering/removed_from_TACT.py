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


def var_adjustment(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode):
    #Uses Taylor's frozen turbulence hypothesis with data from the vertically
    #pointing beam to estimate new values of the u and v variance.

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

    import numpy as np

    w_N = np.zeros(len(vr_z))
    w_N[:] = np.nan
    w_E = np.zeros(len(vr_z))
    w_E[:] = np.nan
    w_S = np.zeros(len(vr_z))
    w_S[:] = np.nan
    w_W = np.zeros(len(vr_z))
    w_W[:] = np.nan

    u_bar = np.sin(np.radians(wd - 180))*U
    v_bar = np.cos(np.radians(wd - 180))*U

    delta_t_vert_beam = 1./frequency_vert_beam

    #Calculate the number of time steps needed for eddies to travel from one
    #side of the scanning circle to the other
    dist = height_needed/np.tan(np.radians(el_angle))
    delta_t_u = dist/u_bar
    interval_u = round(delta_t_u/delta_t_vert_beam)
    delta_t_v = dist/v_bar
    interval_v = round(delta_t_v/delta_t_vert_beam)

    #Estimate values of w at different sides of the scanning circle by using
    #Taylor's frozen turbulence hypothesis
    for i in range(len(vr_z)):
        try:
            w_N[i] = vr_z[i-interval_v]
            w_E[i] = vr_z[i-interval_u]
        except:
            w_N[i] = np.nan
            w_E[i] = np.nan
        try:
            w_S[i] = vr_z[i+interval_v]
            w_W[i] = vr_z[i+interval_u]
        except:
            w_S[i] = np.nan
            w_W[i] = np.nan

    if "taylor_ws" in mode:
        #Use the new values of w to estimate the u and v components using the DBS technique
        #and calculate the variance
        u_DBS_new = ((vr_e-vr_w) - (w_E-w_W)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))
        v_DBS_new = ((vr_n-vr_s) - (w_N-w_S)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))

        u_var_lidar_new = get_10min_var(u_DBS_new,frequency_vert_beam)
        v_var_lidar_new = get_10min_var(v_DBS_new,frequency_vert_beam)
    else:
        #Calculate change in w across the scanning circle in north-south and east-west directions
        dw_est1 = w_S - w_N
        dw_est2 = w_W - w_E

        vr1_var = get_10min_var(vr_n,1./4)
        vr2_var = get_10min_var(vr_e,1./4)
        vr3_var = get_10min_var(vr_s,1./4)
        vr4_var = get_10min_var(vr_w,1./4)
        dw_var1 = get_10min_var(dw_est1,1./4)
        dw_var2 = get_10min_var(dw_est2,1./4)

        vr1_vr3_var = get_10min_covar(vr_n,vr_s,1./4)
        vr2_vr4_var = get_10min_covar(vr_e,vr_w,1./4)

        vr1_dw_var = get_10min_covar(vr_n,dw_est1,1./4)
        vr3_dw_var = get_10min_covar(vr_s,dw_est1,1./4)
        vr2_dw_var = get_10min_covar(vr_e,dw_est2,1./4)
        vr4_dw_var = get_10min_covar(vr_w,dw_est2,1./4)

        #These equations are adapted from Newman et al. (2016), neglecting terms involving
        #du or dv, as these terms are expected to be small compared to dw
        #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
        #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
        #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
        u_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr2_var + vr4_var- 2*vr2_vr4_var + 2*vr2_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr4_dw_var*np.sin(np.radians(el_angle)) + dw_var2*np.sin(np.radians(el_angle))**2)

        v_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr1_var + vr3_var- 2*vr1_vr3_var + 2*vr1_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr3_dw_var*np.sin(np.radians(el_angle)) + dw_var1*np.sin(np.radians(el_angle))**2)

    #Rotate the variance into the mean wind direction
    #Note: The rotation should include a term with the uv covariance, but the
    #covariance terms are also affected by variance contamination. In Newman
    #et al. (2016), it was found that the uv covariance is usually close to 0 and
    #can safely be neglected.
    #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
    #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
    #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
    u_rot_var_new = u_var_lidar_new*(np.sin(np.radians(wd)))**2 + v_var_lidar_new*(np.cos(np.radians(wd)))**2

    #Calculate the wind speed and variance if w is assumed to be the same on all
    #sides of the scanning circle
    u_DBS = (vr_e-vr_w)/(2*np.cos(np.radians(el_angle)))
    v_DBS = (vr_n-vr_s)/(2*np.cos(np.radians(el_angle)))

    u_var_DBS = get_10min_var(u_DBS,frequency_vert_beam)
    v_var_DBS = get_10min_var(v_DBS,frequency_vert_beam)

    u_rot_var = u_var_DBS*(np.sin(np.radians(wd)))**2 + v_var_DBS*(np.cos(np.radians(wd)))**2

    return u_rot_var-u_rot_var_new

def acvf(ts):
    #Calculate autocovariance function for a time series

    #Inputs
    #ts: Time series of data

    #Outputs
    #ts_adj: Values of autovariance function starting from lag 0

    import numpy as np
    lags = range(0,len(ts))
    ts_adj = []
    for i in lags:
        ts_subset_temp = ts[i:len(ts)]
        ts_subset_temp2 = ts[0:len(ts)-i]
        ts_adj.append(np.nanmean((ts_subset_temp-np.nanmean(ts_subset_temp))*(ts_subset_temp2-np.nanmean(ts_subset_temp2))))
    return ts_adj

def inertial_subrange_func(t, b, C):
    #Inertial subrange fit for autocovariance function
    #t is lag time, b is the variance at lag 0, and C is a parameter corresponding to eddy dissipation

    return -C*t**(2./3) + b

def lenschow_technique(ts,frequency,mode_ws,option):
    #Apply different forms of the Lenschow et al. (2000) technique
    #Reference: Lenschow, D. H., V. Wulfmeyer, and C. Senff, 2000: Measuring second-through fourth-order moments in noisy data. J. Atmos. Oceanic Technol., 17, 1330–1347.

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #mode_noise: Type of Lenschow noise adjustment to be applied. Options are linear, subrange, and spectrum.

    #Outputs
    #new_ts_var: 10-min. variance after noise adjustment has been applied

    import numpy as np
    from scipy.optimize import curve_fit

    ts = fill_nan(ts)

    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)
    var_diff = []
    var_orig = []
    lags = np.arange(0,ten_min_count)/float(frequency)
    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        #10-min. window of data
        ts_window = ts[i:i+ten_min_count]
        ten_min_index = (i-1)/ten_min_count + 1
        var_orig.append(get_10min_var(ts_window,frequency))
        if 'linear' in option:
            #Use values of ACVF from first four non-zero lags to linearly extrpolate
            #ACVF to lag 0
            ts_adj = acvf(ts_window)
            x_vals = lags[1:4];
            y_vals = ts_adj[1:4]
            p = np.polyfit(x_vals,y_vals,1)
            var_diff.append(var_orig[ten_min_index]-p[1])
        if 'subrange' in option:
            #Use values of ACVF from first four non-zero lags to produce fit
            #to inertial subrange function. Value of function at lag 0 is assumed
            #to be the true variance.
            ts_adj = acvf(ts_window)
            x_vals = lags[1:4];
            y_vals = ts_adj[1:4]
            try:
                popt, pcov = curve_fit(inertial_subrange_func, x_vals, y_vals,\
                p0 = [np.mean((ts_window-np.mean(ts_window))**2),0.002])
                var_diff.append(var_orig[ten_min_index]-popt[0])
            except:
                 var_diff.append(np.nan)
        if 'spectrum' in option:
           #Assume spectral power at high frequencies is due only to noise. Average
           #the spectral power at the highest 20% of frequencies in the time series
           #and integrate the average power across all frequencies to estimate the
           #noise floor
           import numpy.ma as ma
           if "raw_WC" in mode_ws:
               [S,fr] = get_10min_spectrum_WC_raw(ts_window,frequency)
           else:
               [S,fr] = get_10min_spectrum(ts_window,frequency)
           x = ma.masked_inside(fr,0.8*fr[-1],fr[-1])
           func_temp = []
           for j in range(len(fr)):
               func_temp.append(np.mean(S[x.mask]))
           noise_floor = np.trapz(func_temp,fr)
           var_diff.append(noise_floor)

    var_diff = np.array(var_diff)
    #Only use var_diff values where the noise variance is positive
    var_diff[var_diff < 0] = 0
    new_ts_var = np.array(var_orig)-var_diff
    return new_ts_var

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
    E_A_fast = 2*abs(F_A_fast[0:N/2]**2)
    S_A_fast = (E_A_fast)/delta_f
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
    import functools
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    ts_covar = []

    for i in np.arange(0,len(ts1)-ten_min_count+1,ten_min_count):
        ts_temp1 = ts1[i:i+ten_min_count]
        ts_temp2 = ts2[i:i+ten_min_count]
        mask = [~np.isnan(ts_temp1),~np.isnan(ts_temp2)]
        total_mask = functools.reduce(np.logical_and, mask)
        ts_temp1 = ts_temp1[total_mask]
        ts_temp2 = ts_temp2[total_mask]
        ts_covar.append(np.nanmean((ts_temp1-np.nanmean(ts_temp1))*(ts_temp2-np.nanmean(ts_temp2))))

    return np.array(ts_covar)

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    #Adapted from code posted on Stack Overflow: http://stackoverflow.com/a/9815522
    #1-D linear interpolation to fill missing values

    #Inputs
    #A: Time series where NaNs need to be filled

    #Outputs
    #B: Time series with NaNs filled

    from scipy import interpolate
    import numpy as np
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    #Only perform interpolation if more than 75% of the data are valid
    if float(len(np.array(good).ravel()))/len(A) > 0.75:
        f = interpolate.interp1d(inds[good], A[good],bounds_error=False,fill_value='extrapolate')
        B = np.where(np.isfinite(A),A,f(inds))
    else:
        B = A
    return B

def spike_filter(ts,frequency):
    #Spike filter based on procedure used in Wang et al. (2015)
    #Reference: Wang, H., R. J. Barthelmie,  A. Clifton, and S. C. Pryor, 2015:
    #Wind measurements from arc scans with Doppler wind lidar, J. Atmos.
    #Ocean. Tech., 32, 2024–2040.


    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data

    #Outputs
    #ts_filtered_interp: Filtered time series with NaNs filled in

    import numpy as np


    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    ts_filtered = np.copy(ts)
    ts_filtered_interp = np.copy(ts)

    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        ts_window = ts_filtered[i:i+ten_min_count]

        #Calculate delta_v, difference between adjacent velocity values
        delta_v = np.zeros(len(ts_window)-1)
        for j in range(len(ts_window)-1):
            delta_v[j] = ts_window[j+1] - ts_window[j]
        q75, q25 = np.percentile(delta_v, [75 ,25])
        IQR= q75 - q25
        #If abs(delta_v) at times i and i-1 are larger than twice the interquartile
        #range (IQR) and the delta_v values at times i and i-1 are of opposite sign,
        #the velocity at time i is considered a spike and set to NaN.
        for j in range(1,len(ts_window)-1):
            if abs(delta_v[j]) > 2*IQR and abs(delta_v[j-1]) > 2*IQR:
                if np.sign(delta_v[j]) != np.sign(delta_v[j-1]):
                    ts_window[j] = np.nan
                    ts_filtered[i+j] = np.nan
        #Set entire 10-min. period to NaN if more than 40% of the velocity points
        #are already NaN.
        if (float(len(ts_window[np.isnan(ts_window)]))/len(ts_window)) > 0.4:
            ts_filtered[i:i+ten_min_count] = np.nan
        #Use a 1-D linear interpolation to fill in missing values
        ts_filtered_interp[i:i+ten_min_count] = fill_nan(ts_filtered[i:i+ten_min_count])
    return ts_filtered_interp


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

def Kaimal_spectrum_func(X, L):
    #Given values of frequency (fr), mean horizontal wind speed (U), streamwise
    #variance (u_var), and length scale (L), calculate idealized Kaimal spectrum
    #This uses the form given by Eq. 2.24 in Burton et al. (2001)
    #Reference: Burton, T., D. Sharpe, N. Jenkins, N., and E. Bossanyi, 2001:
    #Wind Energy Handbook, John Wiley & Sons, Ltd., 742 pp.
    fr,U,u_var = X
    return u_var*fr*((4*(L/U)/((1+6*(fr*L/U))**(5./3))))


def Kaimal_spectrum_func2(pars, x, data=None):
    #Kaimal spectrum function for fitting. Trying to minimize the difference
    #between the actual spectrum (data) and the modeled spectrum (model)
    vals = pars.valuesdict()
    L =  vals['L']
    U =  vals['U']
    u_var = vals['u_var']
    model = u_var*x*((4*(L/U)/((1+6*(x*L/U))**(5./3))))
    if data is None:
        return model
    return model-data

def spectral_adjustment(u_rot,frequency,mode_ws,option):
    #Estimates loss of variance due to volume averaging by extrapolating spectrum
    #out to higher frequencies and integrating spectrum over higher frequencies


    #Inputs
    #u_rot: Time series of streamwise wind speed
    #frequency: Sampling frequency of time series
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #option: Type of volume averaging adjustment to be applied. Options are spectral_adjustment_fit and acf.

    #Outputs
    #var_diff: Estimate of loss of streamwise variance due to volume averaging

    import numpy as np
    import scipy.signal
    from lmfit import minimize,Parameters
    ten_min_count = frequency*60*10

    var_diff = []

    for i in np.arange(0,len(u_rot)-ten_min_count+1,ten_min_count):
        u_temp = u_rot[i:i+ten_min_count]
        U = np.mean(u_temp)
        u_var = get_10min_var(u_temp,frequency)
        #Detrend time series before estimating parameters for modeled spectrum
        u_temp = scipy.signal.detrend(u_temp)
        if "raw_WC" in mode_ws:
            [S,fr] = get_10min_spectrum_WC_raw(u_temp,frequency)
        else:
            [S,fr] = get_10min_spectrum(u_temp,frequency)
        if "spectral_adjustment_fit" in option:
            #Find value of length scale that produces best fit to idealized
            #Kaimal spectrum
            fit_params = Parameters()
            fit_params.add('L', value=500,min=0,max=1500)
            fit_params.add('U', value=U,vary=False)
            fit_params.add('u_var', value=u_var,vary=False)
            out = minimize(Kaimal_spectrum_func2, fit_params, args=(fr,), kws={'data':fr*S})
            L = out.params['L'].value

        else:
            #Otherwise, use ACF to estimate integral length scale and use this
            #value for the length scale in the Kaimal modeled spectrum
            lags = np.arange(0,ten_min_count)/float(frequency)
            u_adj = acvf(u_temp)
            u_acf = u_adj/u_adj[0]
            indices = np.arange(0,len(u_acf))
            x = indices[np.array(u_acf)<=0]
            #ACF is integrated to the first zero crossing to esimate the integral
            #time scale and multipled by the mean horizontal wind speed to estimate
            #the integral length scale
            L = np.trapz(u_acf[:x[0]],lags[:x[0]])*U

        fr2 = np.linspace(0,float(10)/2,float(6000/2))
        #Calculate Kaimal spectrum from 0 to 5 Hz
        S_model = Kaimal_spectrum_func((fr2,U,u_var),L)
        #Integrate spectrum for frequency values higher than those in the original
        #spectrum from the lidar
        var_diff.append(np.trapz((S_model[fr2 > fr[-1]]/fr2[fr2 > fr[-1]]),fr2[fr2 > fr[-1]]))


    return np.array(var_diff)

def var_adjustment(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode):
    #Uses Taylor's frozen turbulence hypothesis with data from the vertically
    #pointing beam to estimate new values of the u and v variance.

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

    import numpy as np

    w_N = np.zeros(len(vr_z))
    w_N[:] = np.nan
    w_E = np.zeros(len(vr_z))
    w_E[:] = np.nan
    w_S = np.zeros(len(vr_z))
    w_S[:] = np.nan
    w_W = np.zeros(len(vr_z))
    w_W[:] = np.nan

    u_bar = np.sin(np.radians(wd - 180))*U
    v_bar = np.cos(np.radians(wd - 180))*U

    delta_t_vert_beam = 1./frequency_vert_beam

    #Calculate the number of time steps needed for eddies to travel from one
    #side of the scanning circle to the other
    dist = height_needed/np.tan(np.radians(el_angle))
    delta_t_u = dist/u_bar
    interval_u = np.round(delta_t_u/delta_t_vert_beam)
    delta_t_v = dist/v_bar
    interval_v = np.round(delta_t_v/delta_t_vert_beam)

    #Estimate values of w at different sides of the scanning circle by using
    #Taylor's frozen turbulence hypothesis
    for i in range(len(vr_z)):
        try:
            w_N[i] = vr_z[i-interval_v]
            w_E[i] = vr_z[i-interval_u]
        except:
            w_N[i] = np.nan
            w_E[i] = np.nan
        try:
            w_S[i] = vr_z[i+interval_v]
            w_W[i] = vr_z[i+interval_u]
        except:
            w_S[i] = np.nan
            w_W[i] = np.nan

    if "taylor_ws" in mode:
        #Use the new values of w to estimate the u and v components using the DBS technique
        #and calculate the variance
        u_DBS_new = ((vr_e-vr_w) - (w_E-w_W)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))
        v_DBS_new = ((vr_n-vr_s) - (w_N-w_S)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))

        u_var_lidar_new = get_10min_var(u_DBS_new,frequency_vert_beam)
        v_var_lidar_new = get_10min_var(v_DBS_new,frequency_vert_beam)
    else:
        #Calculate change in w across the scanning circle in north-south and east-west directions
        dw_est1 = w_S - w_N
        dw_est2 = w_W - w_E

        vr1_var = get_10min_var(vr_n,1./4)
        vr2_var = get_10min_var(vr_e,1./4)
        vr3_var = get_10min_var(vr_s,1./4)
        vr4_var = get_10min_var(vr_w,1./4)
        dw_var1 = get_10min_var(dw_est1,1./4)
        dw_var2 = get_10min_var(dw_est2,1./4)

        vr1_vr3_var = get_10min_covar(vr_n,vr_s,1./4)
        vr2_vr4_var = get_10min_covar(vr_e,vr_w,1./4)

        vr1_dw_var = get_10min_covar(vr_n,dw_est1,1./4)
        vr3_dw_var = get_10min_covar(vr_s,dw_est1,1./4)
        vr2_dw_var = get_10min_covar(vr_e,dw_est2,1./4)
        vr4_dw_var = get_10min_covar(vr_w,dw_est2,1./4)

        #These equations are adapted from Newman et al. (2016), neglecting terms involving
        #du or dv, as these terms are expected to be small compared to dw
        #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
        #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
        #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
        u_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr2_var + vr4_var- 2*vr2_vr4_var + 2*vr2_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr4_dw_var*np.sin(np.radians(el_angle)) + dw_var2*np.sin(np.radians(el_angle))**2)

        v_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr1_var + vr3_var- 2*vr1_vr3_var + 2*vr1_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr3_dw_var*np.sin(np.radians(el_angle)) + dw_var1*np.sin(np.radians(el_angle))**2)

    #Rotate the variance into the mean wind direction
    #Note: The rotation should include a term with the uv covariance, but the
    #covariance terms are also affected by variance contamination. In Newman
    #et al. (2016), it was found that the uv covariance is usually close to 0 and
    #can safely be neglected.
    #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
    #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning
    #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
    u_rot_var_new = u_var_lidar_new*(np.sin(np.radians(wd)))**2 + v_var_lidar_new*(np.cos(np.radians(wd)))**2

    #Calculate the wind speed and variance if w is assumed to be the same on all
    #sides of the scanning circle
    u_DBS = (vr_e-vr_w)/(2*np.cos(np.radians(el_angle)))
    v_DBS = (vr_n-vr_s)/(2*np.cos(np.radians(el_angle)))

    u_var_DBS = get_10min_var(u_DBS,frequency_vert_beam)
    v_var_DBS = get_10min_var(v_DBS,frequency_vert_beam)

    u_rot_var = u_var_DBS*(np.sin(np.radians(wd)))**2 + v_var_DBS*(np.cos(np.radians(wd)))**2

    return u_rot_var-u_rot_var_new


def lidar_processing_vol_averaging(u,frequency,mode_ws,mode_vol):
    #Function to estimate variance lost due to volume/temporal averaging

    #Inputs
    #u: Time series of streamwise wind speed
    #frequency: Sampling frequency of time series
    #mode_ws: raw_WC, VAD, or raw_ZephIR
    #mode_vol: Type of volume averaging adjustment to be applied. Options are spectral_adjustment_fit and acf.

    #Outputs
    #var_diff: Estimate of loss of streamwise variance due to volume averaging

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

    var_diff = var_adjustment(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode)

    #Set negative values of var_diff to 0 as they would increase the corrected variance
    #Note: This is not the best procedure and should probably be fixed at some point.
    #It's possible that at times, the change in w across the scanning circle could
    #decrease, rather than increase, the u and v variance.

    try:
        if var_diff < 0:
            var_diff = 0.
        return var_diff
    except:
        var_diff = 0.

def VAD_func(az, x1, x2, x3):
    import numpy as np
    return np.array(x3+x1*np.cos(np.radians(az)-x2))


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
    E_A_fast = 2*abs(F_A_fast[0:N/2]**2)
    S_A_fast = (E_A_fast)/delta_f
    return S_A_fast,frequency_fft


def rotate_ws(u,v,w,frequency):
    #Performs coordinate rotation according to Eqs. 22-29 in Wilczak et al. (2001)
    #Reference: Wilczak, J. M., S. P. Oncley, and S. A. Stage, 2001: Sonic anemometer tilt adjustment algorithms.
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
    import functools

    #Calculates the shear parameter for every 10-min. period of data by fitting power law equation to
    #10-min. mean wind speeds

    #Inputs
    #U: 10-min. mean horizontal wind speed at all measurement heights
    #heights: Measurement heights
    #height_needed: Height where TI is being extracted - values used to calculate shear parameter
    #should be centered around this height

    #Outputs
    #p: 10-min. values of shear parameter

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
    mask = functools.reduce(np.logical_and, mask)
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




