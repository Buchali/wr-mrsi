import numpy as np


def ppm_to_frequency(ppm_values, trn_freq, center_freq=4.7):
    """
    Convert ppm values to corresponding frequency values in an MRS signal.

    Parameters:
    ppm_values (float or array-like): PPM value(s) to convert
    trn_freq (float): Transmitter frequency in Hz
    center_freq (float): Center frequency in ppm (default is 4.7)

    Returns:
    float or np.array: Frequency value(s) in Hz corresponding to the input ppm value(s)
    """

    def convert(ppm):
        return trn_freq * (center_freq - ppm)

    if np.isscalar(ppm_values):
        return convert(ppm_values)
    else:
        return np.array([convert(ppm) for ppm in ppm_values])


def ppm_to_point_index(ppm_values, T, t_step, trn_freq, center_freq=4.7):
    """
    Convert ppm values to corresponding point indexes in an MRS signal.

    Parameters:
    ppm_values (float or array-like): PPM value(s) to convert
    T (int): Number of time points
    t_step (float): Time step between points
    trn_freq (float): Transmitter frequency
    center_freq (float): Center frequency in ppm (default is 4.7)

    Returns:
    int or np.array: Point index(es) corresponding to the input ppm value(s)
    """
    freqs = np.fft.fftshift(np.fft.fftfreq(T, d=t_step))
    ppm_axis = center_freq - (freqs / trn_freq)

    if np.isscalar(ppm_values):
        return np.argmin(np.abs(ppm_axis - ppm_values))
    else:
        return np.array([np.argmin(np.abs(ppm_axis - ppm)) for ppm in ppm_values])
