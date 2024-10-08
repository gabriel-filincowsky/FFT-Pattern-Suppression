import cupy as cp
import numpy as np
from utils.cupy_handler import cp_gaussian_filter

class MaskGenerator:
    """
    Generates various masks for frequency domain image processing.
    """

    def create_circular_mask(self, shape, center, radius, falloff=0):
        """
        Create a circular mask with an optional falloff region.

        :param shape: Tuple representing the shape of the mask (H, W).
        :param center: Tuple representing the center of the mask (row, col).
        :param radius: Radius of the mask in pixels.
        :param falloff: Falloff width for smooth transitions.
        :return: Circular mask as a CuPy array with values ranging from 0 to 1.
        """
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        dist_from_center = cp.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

        if falloff > 0:
            # Smooth transition using sigmoid function
            mask = 1 / (1 + cp.exp((dist_from_center - radius) / falloff))
        else:
            # Hard circular mask
            mask = cp.where(dist_from_center <= radius, 0, 1).astype(cp.float32)

        return mask

    def create_exclusion_mask(self, shape, center, radius, aspect_ratio=1.0, orientation=0, falloff=0):
        """
        Create an exclusion mask centered at a particular frequency region.

        :param shape: Tuple representing the shape of the mask (H, W).
        :param center: Tuple representing the center of exclusion (row, col).
        :param radius: Radius of the exclusion area in pixels.
        :param aspect_ratio: Aspect ratio for elliptical exclusion areas.
        :param orientation: Orientation angle in degrees for the ellipse.
        :param falloff: Falloff width for smooth transitions.
        :return: Exclusion mask as a CuPy array with values ranging from 0 to 1.
        """
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        Y = Y - center[0]
        X = X - center[1]

        # Rotate coordinates
        theta = cp.deg2rad(orientation)
        X_rot = X * cp.cos(theta) + Y * cp.sin(theta)
        Y_rot = -X * cp.sin(theta) + Y * cp.cos(theta)

        # Compute distance with aspect ratio
        dist = cp.sqrt((X_rot / (radius * aspect_ratio)) ** 2 + (Y_rot / radius) ** 2)

        if falloff > 0:
            mask = 1 / (1 + cp.exp((dist - 1) / (falloff / radius)))
        else:
            mask = cp.where(dist <= 1, 0, 1).astype(cp.float32)

        return mask

    def create_antialiasing_mask(self, shape, intensity_pct):
        """
        Create an anti-aliasing mask to smooth high-frequency components.

        :param shape: Tuple representing the shape of the mask (H, W).
        :param intensity_pct: Intensity percentage for the anti-aliasing effect.
        :return: Anti-aliasing mask as a CuPy array with values ranging from 0 to 1.
        """
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        center = (H // 2, W // 2)
        dist_from_center = cp.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

        # Define intensity based on percentage
        max_dist = cp.max(dist_from_center)
        radius = max_dist * (intensity_pct / 100)

        # Create Gaussian mask for anti-aliasing
        mask = cp_gaussian_filter(1 / (1 + cp.exp((dist_from_center - radius) / 10)), sigma=10)

        return mask