import cupy as cp
import numpy as np
from utils.cupy_handler import cp_gaussian_filter

class MaskGenerator:
    """
    Generates various masks used in FFT processing.
    """

    def create_circular_mask(self, shape, center, radius, falloff=0):
        """
        Create a circular mask with a falloff.

        :param shape: Tuple of the mask shape (height, width)
        :param center: Tuple of the center coordinates (y, x)
        :param radius: Radius of the circular mask
        :param falloff: Smoothness of the mask edge
        :return: CuPy array representing the circular mask
        """
        Y, X = cp.ogrid[:shape[0], :shape[1]]
        dist_from_center = cp.sqrt((Y - center[0])**2 + (X - center[1])**2)
        mask = cp.where(dist_from_center <= radius, 1, 0)
        if falloff > 0:
            falloff_region = cp.logical_and(dist_from_center > radius, dist_from_center <= radius + falloff)
            mask = cp.where(falloff_region, 1 - (dist_from_center - radius) / falloff, mask)
        return mask

    def create_exclusion_mask(self, shape, center, radius, aspect_ratio=1.0, orientation=0, falloff=0):
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
        a = radius
        b = radius / aspect_ratio
        cos_theta = cp.cos(theta)
        sin_theta = cp.sin(theta)
        expr = (( (Y - y) * cos_theta + (X - x) * sin_theta )**2) / a**2 + \
               (( (Y - y) * sin_theta - (X - x) * cos_theta )**2) / b**2
        mask = cp.where(expr <= 1, 0, 1)
        if falloff > 0:
            falloff_region = cp.logical_and(expr > 1, expr <= 1 + falloff)
            mask = cp.where(falloff_region, 1 - (expr - 1) / falloff, mask)
        return mask

    def create_antialiasing_mask(self, shape, intensity_pct):
        """
        Create an anti-aliasing mask to smooth high-frequency components.

        :param shape: Tuple of the mask shape (height, width)
        :param intensity_pct: Intensity percentage of the anti-aliasing filter
        :return: CuPy array representing the anti-aliasing mask
        """
        Y, X = cp.ogrid[:shape[0], :shape[1]]
        center_y, center_x = shape[0] // 2, shape[1] // 2
        distance = cp.sqrt((Y - center_y)**2 + (X - center_x)**2)
        max_distance = cp.max(distance)
        intensity = intensity_pct / 100.0
        mask = 1 - cp.clip(distance / (max_distance * intensity), 0, 1)
        return mask