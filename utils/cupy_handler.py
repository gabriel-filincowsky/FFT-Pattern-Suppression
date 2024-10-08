import numpy as np

# Try to import CuPy; if unavailable, fallback to NumPy
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter

    USE_CUPY = True
except ImportError:
    cp = np  # Use NumPy as a substitute
    cpx_ndimage = None  # Placeholder
    cp_fft = np.fft
    from scipy.ndimage import gaussian_filter as cp_gaussian_filter

    # Define cp.asnumpy to ensure compatibility when using NumPy
    cp.asnumpy = lambda x: x
    cp.asarray = np.asarray  # Ensure cp.asarray is defined
    USE_CUPY = False

def check_cupy():
    """Check if CuPy is available."""
    return USE_CUPY