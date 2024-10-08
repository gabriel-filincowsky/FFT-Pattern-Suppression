from models.image_model import ImageModel
from models.parameters_model import ParametersModel
from processing.fft_processor import FFTProcessor
from processing.mask_generator import MaskGenerator
from utils.cupy_handler import cp, cp_fft
import numpy as np
from skimage.feature import peak_local_max
import logging

class ProcessingController:
    """
    Controller for performing image processing tasks.
    """
    def __init__(self, image_model: ImageModel, parameters_model: ParametersModel, 
                 fft_processor: FFTProcessor, mask_generator: MaskGenerator):
        self.image_model = image_model
        self.parameters_model = parameters_model
        self.fft_processor = fft_processor
        self.mask_generator = mask_generator
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_image(self) -> None:
        """
        Process the current image based on available parameters.
        
        Applies high-pass filtering, FFT processing, mask creation, and image reconstruction.
        """
        if self.image_model.original_image is None:
            self.logger.warning("No image loaded to process.")
            return
        
        # Ensure original_image is a CuPy array
        if not isinstance(self.image_model.original_image, cp.ndarray):
            self.image_model.original_image = cp.asarray(self.image_model.original_image)
        
        # Retrieve parameters
        params = self.parameters_model.parameters
        highpass_radius = params.get("High-Pass Filter Radius", 10.0)
        gaussian_blur_pct = params.get("Gaussian Blur (%)", 1.0)
        peak_min_distance = params.get("Peak Min Distance", 10)
        peak_threshold = params.get("Peak Threshold", 0.0010)
        mask_radius_pct = params.get("Mask Radius (%)", 10.0)
        aspect_ratio = params.get("Aspect Ratio", 1.0)
        orientation = params.get("Orientation", 0)
        peak_mask_falloff_pct = params.get("Peak Mask Falloff (%)", 0.0)
        antialiasing_intensity_pct = params.get("Anti-Aliasing Intensity (%)", 50.0)
        enable_peak_suppression = params.get("Enable Frequency Peak Suppression", True)
        enable_attenuation = params.get("Enable Attenuation", True)
        enable_antialiasing = params.get("Enable Anti-Aliasing Filter", True)
        
        # High-Pass Filtering
        highpass_color = self.apply_high_pass_filter(highpass_radius)
        
        # Convert high-pass color image to grayscale
        highpass_gray = cp.mean(highpass_color, axis=2)
        
        # Pad the high-pass grayscale image with neutral gray
        padded_highpass_gray = self.pad_image(highpass_gray)
        
        # Perform FFT processing on padded_highpass_gray
        im_fft_shifted = self.fft_processor.compute_fft(padded_highpass_gray)
        H, W = im_fft_shifted.shape
        crow, ccol = H // 2, W // 2
        
        # Convert relative measurements to absolute pixels
        diag_length = np.sqrt(H**2 + W**2)
        gaussian_sigma = (gaussian_blur_pct / 100) * diag_length
        exclude_radius = (params.get("Exclude Radius (%)", 10.0) / 100) * diag_length / 2
        exclude_falloff = (params.get("Exclude Falloff (%)", 10.0) / 100) * diag_length / 2
        mask_radius = (mask_radius_pct / 100) * diag_length / 2
        peak_mask_falloff = (peak_mask_falloff_pct / 100) * diag_length / 2
        
        # Preprocess the FFT magnitude spectrum
        magnitude_spectrum = cp.abs(im_fft_shifted)
        blurred_spectrum = self.fft_processor.apply_gaussian_blur(magnitude_spectrum, gaussian_sigma)
        
        # Normalize the magnitude spectrum
        normalized_magnitude = blurred_spectrum / blurred_spectrum.max()
        
        # Apply gamma correction
        gamma = params.get("Gamma Correction", 1.0)
        adjusted_magnitude = self.fft_processor.apply_gamma_correction(normalized_magnitude, gamma)
        
        # Create attenuation mask
        attenuation_mask = adjusted_magnitude
        
        # Optionally apply frequency peak suppression
        if enable_peak_suppression:
            spectrum_np = cp.asnumpy(blurred_spectrum)
            exclusion_mask_cp = self.mask_generator.create_exclusion_mask(
                shape=im_fft_shifted.shape,
                center=(crow, ccol),
                radius=exclude_radius,
                aspect_ratio=aspect_ratio,
                orientation=orientation,
                falloff=exclude_falloff
            )
            inclusion_mask_np = cp.asnumpy(exclusion_mask_cp).astype(np.uint8)
            coordinates = peak_local_max(
                spectrum_np,
                min_distance=int(peak_min_distance),
                threshold_abs=peak_threshold * spectrum_np.max(),
                exclude_border=False,
                labels=inclusion_mask_np
            )
            
            # Create masks for each detected peak
            peak_masks = [self.mask_generator.create_circular_mask(
                shape=im_fft_shifted.shape,  # Updated from 'self.fft_processor.fft_shape'
                center=tuple(coord),
                radius=mask_radius,
                falloff=peak_mask_falloff
            ) for coord in coordinates]
            
            # Combine all peak masks
            if peak_masks:
                combined_peak_mask = cp.ones(im_fft_shifted.shape, dtype=cp.float32)
                for mask in peak_masks:
                    combined_peak_mask *= mask
            else:
                combined_peak_mask = cp.ones(im_fft_shifted.shape, dtype=cp.float32)
        else:
            combined_peak_mask = cp.ones(im_fft_shifted.shape, dtype=cp.float32)
        
        # Combined mask is now only the peak mask
        combined_mask = combined_peak_mask
        
        # Step 1: Apply the frequency peak suppression mask
        im_fft_filtered = im_fft_shifted * combined_mask
        
        if enable_peak_suppression and enable_attenuation:
            # Step 2: Create suppression areas mask (inverted mask)
            suppression_areas_mask = 1 - combined_mask
            # Step 3: Apply attenuation mask to frequencies in suppression areas
            im_fft_suppression = im_fft_shifted * suppression_areas_mask * attenuation_mask
            # Step 4: Add the attenuated frequencies back to the filtered FFT image
            im_fft_filtered += im_fft_suppression
        
        # Apply Anti-Aliasing Filter if enabled
        if enable_antialiasing:
            antialiasing_mask = self.mask_generator.create_antialiasing_mask(
                shape=im_fft_shifted.shape, 
                intensity_pct=antialiasing_intensity_pct
            )
            im_fft_filtered *= antialiasing_mask
        
        # Inverse FFT to reconstruct the high-frequency grayscale image
        im_ifft = cp_fft.ifftshift(im_fft_filtered)
        im_new = cp.abs(cp_fft.ifft2(im_ifft))
        
        # Crop the image to remove the padding
        pad = self.fft_processor.padding_size
        im_new_cropped = im_new[pad:-pad, pad:-pad]
        
        # Ensure the cropped image has the original dimensions
        target_shape = highpass_gray.shape
        im_new_cropped = im_new_cropped[:target_shape[0], :target_shape[1]]
        
        # Expand dimensions to match color channels
        im_new_expanded = cp.repeat(im_new_cropped[:, :, cp.newaxis], 3, axis=2)
        
        # Crop the low-pass color image to match the processed image
        blurred_color = blurred_color[:im_new_cropped.shape[0], :im_new_cropped.shape[1], :]
        
        # Blend high-frequency details with low-frequency color image
        final_image_cp = cp.clip(
            blurred_color + im_new_expanded - 128.0, 0, 255
        ).astype(cp.float32)
        
        # Convert to NumPy array for saving
        final_image_np = cp.asnumpy(final_image_cp)
        self.image_model.set_processed_image(final_image_np)
        self.logger.info("Image processing completed and processed_image updated.")
    
    def apply_high_pass_filter(self, radius: float) -> cp.ndarray:
        """
        Apply a high-pass filter to the color image.
        
        :param radius: Radius for the high-pass filter.
        :return: High-pass filtered color image as a CuPy array.
        """
        color_image_cp = cp.asarray(self.image_model.original_image)
        blurred_color = self.fft_processor.apply_high_pass_filter(color_image_cp, radius)
        return blurred_color
    
    def pad_image(self, image: cp.ndarray) -> cp.ndarray:
        """
        Pad the image with a neutral gray value to minimize FFT artifacts.
        
        :param image: Grayscale image as a CuPy array.
        :return: Padded image as a CuPy array.
        """
        padding_value = 128.0  # Neutral gray value
        padded_image = cp.pad(
            image,
            pad_width=self.fft_processor.padding_size,  # Using CuPy's pad
            mode='constant',
            constant_values=padding_value
        )
        return padded_image
    
    # Similarly, ensure that all inputs to CuPy functions are CuPy arrays
    def create_circular_mask(self, shape, center, radius, falloff=0):
        # ... implementation ...
        pass
    
    def create_exclusion_mask(self, shape, center, radius, aspect_ratio=1.0, orientation=0, falloff=0):
        # ... implementation ...
        pass
    
    def create_antialiasing_mask(self, shape, intensity_pct):
        # ... implementation ...
        pass