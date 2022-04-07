import numpy as np


def log_of_ratio(x, xref):
    """Calculate natural logarithm of ratio between two values... useful for power law extrapolation

    Parameters
    ----------
    x : float
        numerator inside log
    xref : float
        denominator inside log

    Returns
    -------
    float
        log(x / xref)
    """
    x_new = np.log(x / xref)
    return x_new


def power_law(uref, h, href, shear):
    """
    Extrapolate wind speed (or other) according to power law.
    NOTE: see  https://en.wikipedia.org/wiki/Wind_profile_power_law
    :param uref: wind speed at reference height (same units as extrapolated wind speed, u)
    :param h: height of extrapolated wind speed (same units as href)
    :param href: reference height (same units as h)
    :param shear: shear exponent alpha (1/7 in neutral stability) (unitless)
    :return u: extrapolated wind speed (same units as uref)
    """
    u = np.array(uref) * np.array(h / href) ** np.array(shear)
    return u
