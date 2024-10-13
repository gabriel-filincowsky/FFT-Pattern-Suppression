import numpy as np
import cupy as cp

# Try to import CuPy; if unavailable, fallback to NumPy
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import cupy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter

    USE_CUPY = True
except ImportError:
    import numpy as cp  # Fallback to NumPy
    from scipy.ndimage import gaussian_filter as cp_gaussian_filter
    cp_fft = np.fft
    cp_ndimage = None
    USE_CUPY = False

def check_cupy():
    """Check if CuPy is available."""
    return USE_CUPY

def normalize_spectrum(spectrum: cp.ndarray) -> cp.ndarray:
    """Normalize the magnitude spectrum."""
    max_val = cp.max(spectrum)
    if max_val == 0:
        return cp.zeros_like(spectrum)
    return spectrum / max_val

def apply_gamma_correction(spectrum: cp.ndarray, gamma: float) -> cp.ndarray:
    """Apply gamma correction to the spectrum."""
    return spectrum ** gamma