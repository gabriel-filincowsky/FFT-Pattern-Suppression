import cupy as cp
import numpy as np
from typing import Any

def normalize_spectrum(spectrum: cp.ndarray) -> cp.ndarray:
    """
    Normalize the magnitude spectrum to range between 0 and 1.

    :param spectrum: CuPy array representing the magnitude spectrum.
    :return: Normalized magnitude spectrum as a CuPy array.
    """
    if not isinstance(spectrum, cp.ndarray):
        raise TypeError("Input spectrum must be a CuPy array.")
    
    max_val = cp.max(spectrum)
    if max_val == 0:
        return spectrum
    return spectrum / max_val

def apply_gamma_correction(magnitude: cp.ndarray, gamma: float) -> cp.ndarray:
    """
    Apply gamma correction to the magnitude spectrum.

    :param magnitude: CuPy array representing the normalized magnitude spectrum.
    :param gamma: Gamma value for correction.
    :return: Gamma-corrected magnitude spectrum as a CuPy array.
    """
    if not isinstance(magnitude, cp.ndarray):
        raise TypeError("Input magnitude must be a CuPy array.")
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than 0.")
    
    return cp.power(magnitude, gamma)