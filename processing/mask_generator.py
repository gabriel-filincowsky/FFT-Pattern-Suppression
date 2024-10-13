import cupy as cp
import numpy as np
from utils.cupy_handler import cp_fft, cp_gaussian_filter

class MaskGenerator:
    """
    Generates various masks used in FFT processing.
    """

    def __init__(self):
        pass

    def create_circular_mask(self, shape: tuple, center: tuple, radius: float, falloff: float = 0.0) -> cp.ndarray:
        """
        Create a circular mask with a falloff.

        :param shape: Tuple of the mask shape (height, width)
        :param center: Tuple of the center coordinates (y, x)
        :param radius: Radius of the circular mask
        :param falloff: Smoothness of the mask edge
        :return: CuPy array representing the circular mask
        """
        Y, X = cp.ogrid[:shape[0], :shape[1]]
        dist_from_center = cp.sqrt((X - center[1])**2 + (Y - center[0])**2)
        mask = cp.ones(shape, dtype=cp.float32)
        if falloff > 0:
            mask[dist_from_center <= radius] = 0
            mask[(dist_from_center > radius) & (dist_from_center <= radius + falloff)] = (
                (radius + falloff - dist_from_center[(dist_from_center > radius) & (dist_from_center <= radius + falloff)]) / falloff
            )
        else:
            mask[dist_from_center <= radius] = 0
        return mask

    def create_exclusion_mask(self, shape: tuple, center: tuple, radius: float, aspect_ratio: float = 1.0,
                              orientation: float = 0.0, falloff: float = 0.0) -> cp.ndarray:
        """
        Create an exclusion mask based on aspect ratio and orientation.

        :param shape: Tuple of the mask shape (height, width)
        :param center: Tuple of the center coordinates (y, x)
        :param radius: Radius of the exclusion area
        :param aspect_ratio: Aspect ratio of the exclusion ellipse
        :param orientation: Rotation angle of the exclusion ellipse in degrees
        :param falloff: Smoothness of the mask edge
        :return: CuPy array representing the exclusion mask
        """
        Y, X = cp.ogrid[:shape[0], :shape[1]]
        y, x = center
        theta = cp.deg2rad(orientation)
        X_rot = (X - x) * cp.cos(theta) + (Y - y) * cp.sin(theta)
        Y_rot = -(X - x) * cp.sin(theta) + (Y - y) * cp.cos(theta)
        ellipse = (X_rot / (radius * aspect_ratio))**2 + (Y_rot / radius)**2
        mask = cp.ones(shape, dtype=cp.float32)
        if falloff > 0:
            mask[ellipse <= 1] = 0
            mask[(ellipse > 1) & (ellipse <= 1 + falloff / radius)] = (
                (1 + falloff / radius - ellipse[(ellipse > 1) & (ellipse <= 1 + falloff / radius)]) / (falloff / radius)
            )
        else:
            mask[ellipse <= 1] = 0
        return mask

    def create_antialiasing_mask(self, shape: tuple, intensity_pct: float) -> cp.ndarray:
        """
        Create an anti-aliasing mask to smooth high-frequency components.

        :param shape: Tuple of the mask shape (height, width)
        :param intensity_pct: Intensity percentage of the anti-aliasing filter
        :return: CuPy array representing the anti-aliasing mask
        """
        Y, X = cp.ogrid[:shape[0], :shape[1]]
        crow, ccol = shape[0] // 2, shape[1] // 2
        radius = (intensity_pct / 100) * min(crow, ccol)
        mask = 1 - cp.exp(-((Y - crow)**2 + (X - ccol)**2) / (2 * (radius**2)))
        return mask