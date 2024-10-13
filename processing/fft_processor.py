import numpy as np
from utils.cupy_handler import cp, cp_fft, cp_gaussian_filter
from processing.mask_generator import MaskGenerator
from skimage.feature import peak_local_max
import cupyx.scipy.ndimage as cpx_ndimage
import cupy as cp
from skimage.filters import gaussian
from typing import Tuple
import logging
from utils.config_manager import ConfigManager
from utils.constants import (
    PADDING_SIZE,
    MAX_HIGH_PASS_RADIUS,
    MIN_HIGH_PASS_RADIUS,
    MAX_GAUSSIAN_BLUR_PCT,
    MIN_GAUSSIAN_BLUR_PCT,
    MAX_PEAK_THRESHOLD,
    MIN_PEAK_THRESHOLD,
    # Import other necessary constants as needed
)
import cupyx

class FFTProcessor:
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the FFTProcessor with default parameters.
        
        :param config_manager: Instance of ConfigManager to retrieve parameters.
        """
        self.config_manager = config_manager
        self.padding_size = self.config_manager.get_parameter("Padding Size", PADDING_SIZE)  # Ensure "Padding Size" is defined
        self.max_high_pass_radius = self.config_manager.get_parameter("Max High Pass Radius", MAX_HIGH_PASS_RADIUS)
        self.min_high_pass_radius = self.config_manager.get_parameter("Min High Pass Radius", MIN_HIGH_PASS_RADIUS)
        self.mask_generator = MaskGenerator(config_manager=self.config_manager)
        self.parameters = {}
        self.blurred_image = None
        self.highpass_image = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("FFTProcessor initialized.")

        # Initialize a CuPy memory pool for efficient GPU memory management
        self.pool = cupyx.cuda.MemoryPool()
        cp.cuda.set_allocator(self.pool.malloc)
        self.logger.debug("CuPy memory pool initialized for FFTProcessor.")

    def process_image(self, image, parameters):
        """
        Process the image using FFT and provided parameters.
        """
        try:
            self.logger.info("Starting image processing using FFT.")
            if image is None:
                self.logger.error("Input image is None. Please ensure an image is loaded before processing.")
                raise ValueError("Input image is None.")
    
            # Convert the image to a CuPy array (if not already) using pinned memory for faster transfers
            self.logger.debug("Converting image to CuPy array.")
            image_cp = cp.asarray(image, dtype=cp.float32)
            image_cp = cp.ascontiguousarray(image_cp)
    
            # Retrieve parameters
            self.logger.debug("Retrieving processing parameters.")
            highpass_radius = self.config_manager.get_parameter("High-Pass Filter Radius", 10.0)  # pixels
            gaussian_sigma_pct = self.config_manager.get_parameter("Gaussian Blur (%)", 0.9)  # percent
            peak_min_distance = self.config_manager.get_parameter("Peak Min Distance", 10)  # pixels
            peak_threshold = self.config_manager.get_parameter("Peak Threshold", 0.5)  # unitless
            radius_pct = self.config_manager.get_parameter("Radius (%)", 10.0)  # percent
            aspect_ratio = self.config_manager.get_parameter("Aspect Ratio", 1.0)  # unitless
            orientation = self.config_manager.get_parameter("Orientation", 0.0)  # degrees
            falloff_pct = self.config_manager.get_parameter("Falloff (%)", 10.0)  # percent
            mask_radius_pct = self.config_manager.get_parameter("Mask Radius (%)", 5.0)  # percent
            peak_mask_falloff_pct = self.config_manager.get_parameter("Peak Mask Falloff (%)", 10.0)  # percent
            gamma_correction = self.config_manager.get_parameter("Gamma Correction", self.config_manager.get_parameter("Gamma Correction", default=1.0))  # unitless
            antialiasing_intensity = self.config_manager.get_parameter("Anti-Aliasing Intensity (%)", 50.0)  # percent
            enable_peak_suppression = self.config_manager.get_parameter("Enable Frequency Peak Suppression", False)  # boolean
            enable_attenuation = self.config_manager.get_parameter("Enable Attenuation", False)  # boolean
            enable_antialiasing = self.config_manager.get_parameter("Enable Anti-Aliasing Filter", False)  # boolean
    
            # Calculate additional derived parameters
            highpass_radius_px = highpass_radius  # pixels
    
            # Apply Gaussian Blur with optimized CuPy operations
            self.logger.info("Applying Gaussian Blur.")
            blurred_color = cp.zeros_like(image_cp)
            for i in range(3):
                blurred_color[:, :, i] = cp_gaussian_filter(
                    image_cp[:, :, i], sigma=highpass_radius
                )
            self.logger.debug("Gaussian Blur applied successfully.")
    
            # ... additional processing steps with optimized CuPy methods ...
    
            self.logger.info(f"Image processing completed successfully. Final image shape: {final_image_cp.shape}")
            return final_image_cp
        
        except Exception as e:
            self.logger.exception(f"Error during image processing: {str(e)}")
            raise
        finally:
            # Free any temporary GPU memory if needed
            self.pool.free_all_blocks()
            self.logger.debug("Freed all unused GPU memory.")

    def pad_to_power_of_two(self, image: cp.ndarray) -> Tuple[cp.ndarray, Tuple[int, int]]:
        """Pad image dimensions to the nearest power of two for FFT optimization and ensure even dimensions."""
        new_height = int(2 ** cp.ceil(cp.log2(image.shape[0])))
        new_width = int(2 ** cp.ceil(cp.log2(image.shape[1])))

        # Initial padding to power of two
        padded_image = cp.pad(
            image,
            pad_width=((0, new_height - image.shape[0]), (0, new_width - image.shape[1]), (0, 0)),
            mode='constant',
            constant_values=128.0
        )

        extra_padding = (0, 0)  # (pad_h, pad_w)

        # Check for even dimensions and pad if necessary
        if padded_image.shape[0] % 2 != 0:
            padded_image = cp.pad(padded_image, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=128.0)
            extra_padding = (1, extra_padding[1])

        if padded_image.shape[1] % 2 != 0:
            padded_image = cp.pad(padded_image, pad_width=((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=128.0)
            extra_padding = (extra_padding[0], 1)

        # Add assertion to confirm the padded image has even dimensions
        assert padded_image.shape[0] % 2 == 0 and padded_image.shape[1] % 2 == 0, \
            f"Padded image dimensions are not even: {padded_image.shape}"

        self.logger.debug(f"Padded image shape: {padded_image.shape}")
        return padded_image, extra_padding

    def crop_padding(self, image: cp.ndarray, original_shape: tuple, extra_padding: Tuple[int, int]) -> cp.ndarray:
        """Crop the padded areas from the image, accounting for any extra padding added."""
        pad_h, pad_w = self.padding_size, self.padding_size
        crop_h = (image.shape[0] - original_shape[0]) // 2 - extra_padding[0]
        crop_w = (image.shape[1] - original_shape[1]) // 2 - extra_padding[1]
        cropped_image = image[crop_h : crop_h + original_shape[0], crop_w : crop_w + original_shape[1], :]

        # Add assertion to confirm the cropped image matches target shape
        assert cropped_image.shape[:2] == original_shape, f"Cropped image shape {cropped_image.shape} does not match target shape {original_shape}"

        self.logger.debug(f"Cropped image shape: {cropped_image.shape}")
        return cropped_image

    def crop_color_image(self, color_image: cp.ndarray, target_shape: tuple) -> cp.ndarray:
        """Crop the color image to match the target shape."""
        cropped_color = color_image[:target_shape[0], :target_shape[1], :]
        return cropped_color

    def compute_fft(self, image_cp: cp.ndarray) -> cp.ndarray:
        """Compute the FFT of the image and shift the zero frequency component to the center."""
        fft_result = cp_fft.fft2(image_cp, axes=(0,1))
        fft_shifted = cp_fft.fftshift(fft_result, axes=(0,1))
        return fft_shifted

    def compute_ifft(self, fft_shifted: cp.ndarray) -> cp.ndarray:
        """Compute the inverse FFT to reconstruct the image."""
        ifft_shifted = cp_fft.ifftshift(fft_shifted, axes=(0,1))
        im_ifft = cp_fft.ifft2(ifft_shifted, axes=(0,1))
        return cp.abs(im_ifft)