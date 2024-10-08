import cupy as cp
import numpy as np

def normalize_spectrum(spectrum):
    """
    Normalize the magnitude spectrum to range between 0 and 1.

    :param spectrum: CuPy array representing the magnitude spectrum.
    :return: Normalized magnitude spectrum as a CuPy array.
    """
    max_val = cp.max(spectrum)
    if max_val == 0:
        return spectrum
    return spectrum / max_val

def apply_gamma_correction(magnitude, gamma):
    """
    Apply gamma correction to the magnitude spectrum.

    :param magnitude: CuPy array representing the normalized magnitude spectrum.
    :param gamma: Gamma value for correction.
    :return: Gamma-corrected magnitude spectrum as a CuPy array.
    """
    return cp.power(magnitude, gamma)