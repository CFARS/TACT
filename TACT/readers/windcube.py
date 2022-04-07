import numpy as np


def import_WC_file_VAD(filename, height_needed):
    """Outputs u, v, and w values from VAD technique

    Reads in WINDCUBE .rtd file and performs VAD technique at desired height,
    w values from vertical beam, measurement heights, and timestamp.

    Parameters
    ----------
    filename : str
        WINDCUBE v2 .rtd file to read
    height_needed : float
        Height where VAD analysis should be performed

    Returns
    -------
    tuple
        tuple of 6 numpy arrays
        u_VAD, v_VAD, w_VAD: u, v, and w values from VAD fit at height_needed
        vert_beam: Radial velocity from vertical beam at height_needed
        time_datenum: Timestamps corresponding to the start of each scan in datetime format
        time_datenum_vert_beam: Timestamps corresponding to vertical beam position in datetime format
    """
    from scipy.optimize import curve_fit
    from datetime import datetime

    inp = open(filename).readlines()
    height_array = str.split(inp[38])
    heights_temp = height_array[2:]

    if len(heights_temp) == 0:
        height_array = str.split(inp[39])
        heights_temp = height_array[2:]

    heights = [int(i) for i in heights_temp]

    height_needed_index = min_diff(heights, height_needed, 6.1)

    num_rows = 41
    timestamp = np.loadtxt(
        filename,
        delimiter="\t",
        usecols=(0,),
        dtype=str,
        unpack=True,
        skiprows=num_rows,
    )

    try:
        datetime.strptime(timestamp[0], "%Y/%m/%d %H:%M:%S.%f")
    except:
        num_rows = 42
        timestamp = np.loadtxt(
            filename,
            delimiter="\t",
            usecols=(0,),
            dtype=str,
            unpack=True,
            skiprows=num_rows,
        )

    time_datenum_temp = []
    bad_rows = []
    # Create list of rows where timestamp cannot be converted to datetime
    for i in range(0, len(timestamp)):
        try:
            time_datenum_temp.append(
                datetime.strptime(timestamp[i], "%Y/%m/%d %H:%M:%S.%f")
            )
        except:
            bad_rows.append(i)

    # Delete all timestamp and datetime values from first bad row to end of dataset
    if bad_rows:
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp, range(bad_rows[0], len(timestamp)), axis=0)
        time_datenum_temp = np.delete(
            time_datenum_temp, range(bad_rows[0], len(time_datenum_temp)), axis=0
        )
    else:
        footer_lines = 0

    # Skip lines that correspond to bad data
    az_angle = np.genfromtxt(
        filename,
        delimiter="\t",
        usecols=(1,),
        dtype=str,
        unpack=True,
        skip_header=num_rows,
        skip_footer=footer_lines,
    )
    vr_nan = np.empty(len(time_datenum_temp))
    vr_nan[:] = np.nan

    vr = []
    for i in range(1, len(heights) + 1):
        try:
            vr.append(
                -np.genfromtxt(
                    filename,
                    delimiter="\t",
                    usecols=(i * 9 - 4),
                    dtype=None,
                    skip_header=num_rows,
                    skip_footer=footer_lines,
                )
            )
        except:
            vr.append(vr_nan)

    vr = np.array(vr)
    vr = vr.transpose()

    bad_rows = []
    # Find rows where time decreases instead of increasing
    for i in range(1, len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)

    # Delete rows where time decreases instead of increasing
    if bad_rows:
        vr = np.delete(vr, bad_rows, axis=0)
        time_datenum_temp = np.delete(time_datenum_temp, bad_rows, axis=0)
        timestamp = np.delete(timestamp, bad_rows, axis=0)
        az_angle = np.delete(az_angle, bad_rows, axis=0)

    # Sort timestamp, vr, and az angle in order of ascending datetime value
    timestamp_sorted = [timestamp[i] for i in np.argsort(time_datenum_temp)]
    vr_sorted = np.array([vr[i, :] for i in np.argsort(time_datenum_temp)])
    az_angle_sorted = np.array([az_angle[i] for i in np.argsort(time_datenum_temp)])

    vert_beam = []
    vr_temp = []
    az_temp = []
    timestamp_az = []
    timestamp_vert_beam = []
    # Separate vertical beam values (where az angle = "V") from off-vertical beam values
    for i in range(0, len(az_angle_sorted)):
        if "V" in az_angle_sorted[i]:
            vert_beam.append(vr_sorted[i, height_needed_index])
            timestamp_vert_beam.append(timestamp_sorted[i])
        else:
            vr_temp.append(vr_sorted[i, height_needed_index])
            az_temp.append(float(az_angle_sorted[i]))
            timestamp_az.append(timestamp_sorted[i])

    vr_temp = np.array(vr_temp)
    elevation = 62
    u_VAD = []
    v_VAD = []
    w_VAD = []
    timestamp_VAD = []

    # Perform a VAD fit on each full scan
    for i in range(0, round(len(az_temp) / 4)):
        x_vals = np.array(az_temp[i * 4 + 1 : i * 4 + 5])
        y_vals = np.array(vr_temp[i * 4 + 1 : i * 4 + 5])
        if len(y_vals[np.isnan(y_vals)]) == 0:
            # Initial guesses for the VAD fit parameters
            p0 = np.array(
                [(np.max(y_vals) - np.min(y_vals)) / 2, 2 * np.pi, np.nanmean(y_vals)]
            )
            popt, pcov = curve_fit(VAD_func, x_vals.ravel(), y_vals.ravel(), p0.ravel())
            ws_temp = popt[0] / np.cos(np.radians(elevation))
            wd_temp = np.degrees(popt[1] - np.pi)
            if wd_temp > 360:
                wd_temp -= 360
            u_VAD.append(np.sin(np.radians(wd_temp) - np.pi) * ws_temp)
            v_VAD.append(np.cos(np.radians(wd_temp) - np.pi) * ws_temp)
            w_VAD.append(popt[2] / np.sin(np.radians(elevation)))
        else:
            u_VAD.append(np.nan)
            v_VAD.append(np.nan)
            w_VAD.append(np.nan)
        timestamp_VAD.append(timestamp_az[i * 4 + 1])

    # Convert VAD and vertical beam timestamps to datetime format

    time_datenum = []

    for i in range(0, len(timestamp_VAD)):
        time_datenum.append(datetime.strptime(timestamp_VAD[i], "%Y/%m/%d %H:%M:%S.%f"))

    time_datenum_vert_beam = []

    for i in range(0, len(timestamp_vert_beam)):
        time_datenum_vert_beam.append(
            datetime.strptime(timestamp_vert_beam[i], "%Y/%m/%d %H:%M:%S.%f")
        )

    return (
        np.array(u_VAD),
        np.array(v_VAD),
        np.array(w_VAD),
        np.array(vert_beam)[:, 0],
        np.array(time_datenum),
        np.array(time_datenum_vert_beam),
    )


def get_10min_spectrum_WC_raw(ts, frequency):
    """Calculate power spectrum for 10-min. period

    Parameters
    ----------
    ts : ?
        Time series of data
    frequency : float
        Sampling frequency of data

    Returns
    -------
    tuple
        S_A_fast : float
            Spectral power
        frequency_fft : float
            Frequencies correspond to spectral power values
    """

    import numpy as np

    N = len(ts)
    delta_f = float(frequency) / N
    frequency_fft = np.linspace(0, float(frequency) / 2, float(N / 2))
    F_A_fast = np.fft.fft(ts) / N
    E_A_fast = 2 * abs(F_A_fast[0 : N / 2] ** 2)
    S_A_fast = (E_A_fast) / delta_f
    # Data are only used for frequencies lower than 0.125 Hz. Above 0.125 Hz, the
    # WINDCUBE spectrum calculated using raw data begins to show an artifact. This
    # artifact is due to the recording of the u, v, and w components for every beam
    # position, which results in repeating components.
    S_A_fast = S_A_fast[frequency_fft <= 0.125]
    frequency_fft = frequency_fft[frequency_fft <= 0.125]
    return S_A_fast, frequency_fft


def import_WC_file(filename):
    """Reads in WINDCUBE .rtd file and outputs raw u, v, and w components, measurement heights, and timestamp.

    Parameters
    ----------
    filename : str
        WINDCUBE v2 .rtd file to read

    Returns
    -------
    tuple
        u_sorted, v_sorted, w_sorted: Raw u, v, and w values from all measurement heights
        heights: Measurement heights from file
        time_datenum_sorted: All timestamps in datetime format
    """

    import numpy as np
    from datetime import datetime
    import codecs

    # Read in row containing heights (either row 38 or 39) and convert heights to a set of integers.
    # inp = open(filename,encoding='ISO-8859-1').readlines()
    inp = open(filename).readlines()
    height_array = str.split(inp[38])
    heights_temp = height_array[2:]

    if len(heights_temp) == 0:
        height_array = str.split(inp[39])
        heights_temp = height_array[2:]

    heights = [int(i) for i in heights_temp]

    # Read in timestamps. There will be either 41 or 42 headerlines.
    num_rows = 41
    filecp = codecs.open(filename, encoding="ISO-8859-1")
    timestamp = np.loadtxt(
        filename,
        delimiter="\t",
        usecols=(0,),
        dtype=str,
        unpack=True,
        skiprows=num_rows,
    )

    try:
        datetime.strptime(timestamp[0], "%Y/%m/%d %H:%M:%S.%f")
    except:
        num_rows = 42
        timestamp = np.loadtxt(
            filecp,
            delimiter="\t",
            usecols=(0,),
            dtype=str,
            unpack=True,
            skiprows=num_rows,
        )

    # Convert timestamps to Python datetime format. Some timestamps may be blank and will raise an error during the
    # datetime conversion. The rows corresponding to these bad timestamps are recorded.
    time_datenum_temp = []
    bad_rows = []
    for i in range(0, len(timestamp)):
        try:
            time_datenum_temp.append(
                datetime.strptime(timestamp[i], "%Y/%m/%d %H:%M:%S.%f")
            )
        except:
            bad_rows.append(i)

    # If bad timestamps are detected, an error message is output to the screen and all timestamps including and following
    # the bad timestamp are deleted. The rows corresponding to these timestamps are categorized as footer lines and are not
    # used when reading in the velocity data.
    if bad_rows:
        print(filename, ": Issue reading timestamp")
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp, range(bad_rows[0], len(timestamp)), axis=0)
        time_datenum_temp = np.delete(
            time_datenum_temp, range(bad_rows[0], len(time_datenum_temp)), axis=0
        )
    else:
        footer_lines = 0

    # Create column of NaNs for measurement heights that raise an error.
    v_nan = np.empty(len(time_datenum_temp))
    v_nan[:] = np.nan

    u = []
    v = []
    w = []

    # Read in values of u, v, and w one measurement height at a time. Definitions of the wind components are as follows:
    # u is east-west wind (u > 0 means wind is coming from the west)
    # v is north-south wind (v > 0 means wind is coming from the south)
    # w is vertical wind (w > 0 means wind is upward)

    for i in range(1, len(heights) + 1):
        try:
            u.append(
                -np.genfromtxt(
                    filename,
                    delimiter="\t",
                    usecols=(i * 9 + 1),
                    dtype=None,
                    skip_header=num_rows,
                    skip_footer=footer_lines,
                )
            )
            v.append(
                -np.genfromtxt(
                    filename,
                    delimiter="\t",
                    usecols=(i * 9),
                    dtype=None,
                    skip_header=num_rows,
                    skip_footer=footer_lines,
                )
            )
            w.append(
                -np.genfromtxt(
                    filename,
                    delimiter="\t",
                    usecols=(i * 9 + 2),
                    dtype=None,
                    skip_header=num_rows,
                    skip_footer=footer_lines,
                )
            )
        except:
            u.append(v_nan)
            v.append(v_nan)
            w.append(v_nan)

    u = np.array(u).transpose()
    v = np.array(v).transpose()
    w = np.array(w).transpose()

    # Check to make sure all timestamps follow the initial timestamp. If a particular timestamp is earlier than the first
    # timestamp in the data file, this row is marked as a bad row and removed from the data.
    bad_rows = []
    for i in range(1, len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)

    if bad_rows:
        print(filename, ": Issue with timestamp order")
        u = np.delete(u, bad_rows, axis=0)
        v = np.delete(v, bad_rows, axis=0)
        w = np.delete(w, bad_rows, axis=0)
        time_datenum_temp = np.delete(time_datenum_temp, axis=0)
        timestamp = np.delete(timestamp, axis=0)

    # Sort data by timestamp to ensure that variables are in correct temporal order.
    time_datenum_sorted = np.array(
        [time_datenum_temp[i] for i in np.argsort(time_datenum_temp)]
    )
    u_sorted = np.array([u[i, :] for i in np.argsort(time_datenum_temp)])
    v_sorted = np.array([v[i, :] for i in np.argsort(time_datenum_temp)])
    w_sorted = np.array([w[i, :] for i in np.argsort(time_datenum_temp)])

    return u_sorted, v_sorted, w_sorted, heights, timestamp, time_datenum_sorted


def import_WC_file_vr(filename, height_needed):
    """Reads in WINDCUBE .rtd file and extracts off-vertical radial wind speed components at desired height.

    Parameters
    ----------
    filename : str
        WINDCUBE v2 .rtd file to read
    height_needed : float
        Height where off-vertical measurements should be extracted

    Returns
    -------
    tuple
        vr_n,vr_e,vr_s,vr_w: Time series from north-, east-, south-, and west-pointing beams,
            respectively, at height_needed
        time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w: Timestamps corresponding to
        north-, east-, south-, and west-pointing beams, respectively, in datetime format
    """

    inp = open(filename).readlines()
    height_array = str.split(inp[38])
    heights_temp = height_array[2:]

    if len(heights_temp) == 0:
        height_array = str.split(inp[39])
        heights_temp = height_array[2:]

    heights = [int(i) for i in heights_temp]

    height_needed_index = min_diff(heights, height_needed, 6.1)

    num_rows = 41
    timestamp = np.loadtxt(
        filename,
        delimiter="\t",
        usecols=(0,),
        dtype=str,
        unpack=True,
        skiprows=num_rows,
    )

    try:
        datetime.datetime.strptime(timestamp[0], "%Y/%m/%d %H:%M:%S.%f")
    except:
        num_rows = 42
        timestamp = np.loadtxt(
            filename,
            delimiter="\t",
            usecols=(0,),
            dtype=str,
            unpack=True,
            skiprows=num_rows,
        )

    time_datenum_temp = []
    bad_rows = []
    # Create list of rows where timestamp cannot be converted to datetime
    for i in range(0, len(timestamp)):
        try:
            time_datenum_temp.append(
                datetime.datetime.strptime(timestamp[i], "%Y/%m/%d %H:%M:%S.%f")
            )
        except:
            bad_rows.append(i)

    # Delete all timestamp and datetime values from first bad row to end of dataset
    if bad_rows:
        footer_lines = len(time_datenum_temp) - bad_rows[0] + 1
        timestamp = np.delete(timestamp, range(bad_rows[0], len(timestamp)), axis=0)
        time_datenum_temp = np.delete(
            time_datenum_temp, range(bad_rows[0], len(time_datenum_temp)), axis=0
        )
    else:
        footer_lines = 0

    # Skip lines that correspond to bad data
    az_angle = np.genfromtxt(
        filename,
        delimiter="\t",
        usecols=(1,),
        dtype=str,
        unpack=True,
        skip_header=num_rows,
        skip_footer=footer_lines,
    )

    vr_nan = np.empty(len(time_datenum_temp))
    vr_nan[:] = np.nan

    vr = []
    for i in range(1, len(heights) + 1):
        try:
            vr.append(
                -np.genfromtxt(
                    filename,
                    delimiter="\t",
                    usecols=(i * 9 - 4),
                    dtype=None,
                    skip_header=num_rows,
                    skip_footer=footer_lines,
                )
            )
        except:
            vr.append(vr_nan)

    vr = np.array(vr)
    vr = vr.transpose()

    bad_rows = []
    # Find rows where time decreases instead of increasing
    for i in range(1, len(time_datenum_temp)):
        if time_datenum_temp[i] < time_datenum_temp[0]:
            bad_rows.append(i)

    # Delete rows where time decreases instead of increasing
    if bad_rows:
        vr = np.delete(vr, bad_rows, axis=0)
        time_datenum_temp = np.delete(time_datenum_temp, bad_rows, axis=0)
        timestamp = np.delete(timestamp, bad_rows, axis=0)
        az_angle = np.delete(az_angle, bad_rows, axis=0)

    # Sort timestamp, vr, and az angle in order of ascending datetime value
    timestamp_sorted = [timestamp[i] for i in np.argsort(time_datenum_temp)]
    vr_sorted = np.array([vr[i, :] for i in np.argsort(time_datenum_temp)])
    az_angle_sorted = np.array([az_angle[i] for i in np.argsort(time_datenum_temp)])

    vr_sorted = np.array(vr_sorted)

    vr_temp = []
    az_temp = []
    timestamp_az = []
    vert_beam = []
    timestamp_vert_beam = []
    # Separate vertical beam values (where az angle = "V") from off-vertical beam values
    for i in range(0, len(az_angle_sorted)):
        if "V" in az_angle_sorted[i]:
            vert_beam.append(vr_sorted[i, :])
            timestamp_vert_beam.append(timestamp_sorted[i])
        else:
            vr_temp.append(vr_sorted[i, :])
            az_temp.append(float(az_angle_sorted[i]))
            timestamp_az.append(timestamp_sorted[i])

    vr_temp = np.array(vr_temp)
    az_temp = np.array(az_temp)

    # Extract data for north-, east-, south-, and west-pointing beams at height of interest
    vr_n = vr_temp[az_temp == 0, height_needed_index]
    vr_e = vr_temp[az_temp == 90, height_needed_index]
    vr_s = vr_temp[az_temp == 180, height_needed_index]
    vr_w = vr_temp[az_temp == 270, height_needed_index]

    # Convert timestamps to datetime format
    time_datenum = []
    time_datenum_vert_beam = []

    for i in range(0, len(timestamp_az)):
        time_datenum.append(
            datetime.datetime.strptime(timestamp_az[i], "%Y/%m/%d %H:%M:%S.%f")
        )

    for i in range(0, len(timestamp_vert_beam)):
        time_datenum_vert_beam.append(
            datetime.datetime.strptime(timestamp_vert_beam[i], "%Y/%m/%d %H:%M:%S.%f")
        )

    time_datenum = np.array(time_datenum)
    time_datenum_vert_beam = np.array(time_datenum_vert_beam)
    time_datenum_n = time_datenum[az_temp == 0]
    time_datenum_e = time_datenum[az_temp == 90]
    time_datenum_s = time_datenum[az_temp == 180]
    time_datenum_w = time_datenum[az_temp == 270]

    return (
        vr_n,
        vr_e,
        vr_s,
        vr_w,
        heights,
        time_datenum_n,
        time_datenum_e,
        time_datenum_s,
        time_datenum_w,
        vert_beam,
        time_datenum_vert_beam,
    )


def WC_processing_standard(filename, option, hub_height):
    """
    Parameters
    ----------
    filename : str
    option : str
    hub_height : float

    Returns
    -------
    tuple
    """

    if not "vr" in option:
        if "raw" in option:
            frequency = 1.0
            [u, v, w, heights, timestamp, time_datenum] = import_WC_file(filename)
            u_interp = []
            v_interp = []
            w_interp = []

            for i in range(len(heights)):
                [u_interp_temp, time_interp] = interp_ts(
                    u[:, i], time_datenum, 1.0 / frequency
                )
                [v_interp_temp, time_interp] = interp_ts(
                    v[:, i], time_datenum, 1.0 / frequency
                )
                [w_interp_temp, time_interp] = interp_ts(
                    w[:, i], time_datenum, 1.0 / frequency
                )
                u_interp.append(u_interp_temp)
                v_interp.append(v_interp_temp)
                w_interp.append(w_interp_temp)

            u_interp = np.transpose(np.array(u_interp))
            v_interp = np.transpose(np.array(v_interp))
            w_interp = np.transpose(np.array(w_interp))

            hub_height_index = min_diff(heights, [hub_height], 6.1)

            [U, wd, time_datenum_10min] = get_10min_mean_ws_wd(
                u_interp, v_interp, time_interp, frequency
            )

            if len(time_datenum_10min) != 0:

                p = get_10min_shear_parameter(U, heights, hub_height)
                U = U[:, hub_height_index]
                [u_rot, v_rot, w_rot] = rotate_ws(
                    u_interp[:, hub_height_index],
                    v_interp[:, hub_height_index],
                    w_interp[:, hub_height_index],
                    frequency,
                )
                return (
                    u_rot,
                    U[:, 0],
                    wd,
                    p,
                    time_datenum_10min,
                    time_interp,
                    hub_height_index,
                    u_interp[:, hub_height_index],
                    v_interp[:, hub_height_index],
                )
            else:
                p = []
                U = []
                [u_rot, v_rot, w_rot] = rotate_ws(
                    u_interp[:, hub_height_index],
                    v_interp[:, hub_height_index],
                    w_interp[:, hub_height_index],
                    frequency,
                )
                return (
                    u_rot,
                    U,
                    p,
                    time_datenum_10min,
                    time_interp,
                    hub_height_index,
                    u_interp[:, hub_height_index],
                    v_interp[:, hub_height_index],
                )
                # return u_rot,U[:,0],p,time_datenum_10min,time_interp,hub_height_index
        elif "VAD" in option:
            frequency = 1.0 / 4
            [
                u,
                v,
                w,
                vert_beam,
                time_datenum,
                time_datenum_vert_beam,
            ] = import_WC_file_VAD(filename, hub_height)

            [u_interp, time_interp] = interp_ts(u, time_datenum, 1.0 / frequency)
            [v_interp, time_interp] = interp_ts(v, time_datenum, 1.0 / frequency)
            [w_interp, time_interp] = interp_ts(w, time_datenum, 1.0 / frequency)
            [vert_beam_interp, time_interp] = interp_ts(
                vert_beam, time_datenum_vert_beam, 1.0 / frequency
            )

            u_interp = np.transpose(np.array(u_interp))
            v_interp = np.transpose(np.array(v_interp))
            w_interp = np.transpose(np.array(w_interp))
            vert_beam_interp = np.transpose(np.array(vert_beam_interp))

            u_interp = u_interp.reshape(len(u_interp), 1)
            v_interp = v_interp.reshape(len(v_interp), 1)
            w_interp = w_interp.reshape(len(w_interp), 1)
            vert_beam_interp = vert_beam_interp.reshape(len(vert_beam_interp), 1)

            [U, wd, time_datenum_10min] = get_10min_mean_ws_wd(
                u_interp, v_interp, time_interp, frequency
            )

            [u_rot, v_rot, w_rot] = rotate_ws(
                u_interp, v_interp, vert_beam_interp, frequency
            )

            return u_rot, U, w_rot, time_datenum_10min, time_interp

    else:
        frequency = 1.0 / 4
        #  [vr_n,vr_n_dispersion,vr_e,vr_e_dispersion,vr_s,vr_s_dispersion,vr_w,vr_w_dispersion,vert_beam,vert_beam_dispersion,
        #   u_VAD,v_VAD,w_VAD,heights,time_datenum_n,time_datenum_e,time_datenum_s,time_datenum_w,time_datenum_vert_beam,time_datenum_VAD,
        #   SNR_n,SNR_e,SNR_s,SNR_w,SNR_vert_beam] = import_WC_file_vr(filename,hub_height)
        [
            vr_n,
            vr_e,
            vr_s,
            vr_w,
            heights,
            time_datenum_n,
            time_datenum_e,
            time_datenum_s,
            time_datenum_w,
            vert_beam,
            time_datenum_vert_beam,
        ] = import_WC_file_vr(filename, hub_height)

        vr_n_interp = []
        vr_e_interp = []
        vr_s_interp = []
        vr_w_interp = []

        # Perform temporal interpolation
        [vr_n_interp, time_interp] = interp_ts(vr_n, time_datenum_n, 1.0 / frequency)
        [vr_e_interp, time_interp] = interp_ts(vr_e, time_datenum_e, 1.0 / frequency)
        [vr_s_interp, time_interp] = interp_ts(vr_s, time_datenum_s, 1.0 / frequency)
        [vr_w_interp, time_interp] = interp_ts(vr_w, time_datenum_w, 1.0 / frequency)

        return vr_n_interp, vr_e_interp, vr_s_interp, vr_w_interp
