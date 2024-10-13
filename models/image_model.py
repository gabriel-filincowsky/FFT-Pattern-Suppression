import cupy as cp
import numpy as np

class ImageModel:
    """
    Represents the image data structure, storing original, processed, and intermediate images.
    """
    def __init__(self):
        self.original_image = None  # Original color image as a CuPy array
        self.processed_image = None  # Final processed image as a NumPy array
        self.intermediate_images = {}  # Dictionary to store intermediate image states
        self.blurred_image = None  # Blurred image (pixels)
        self.highpass_image = None  # High-pass filtered image (pixels)

    def set_original_image(self, image: cp.ndarray):
        """
        Set the original image and reset intermediate states.
        """
        self.original_image = image
        self.reset_images()

    def set_processed_image(self, image: np.ndarray):
        """
        Set the processed image.
        """
        self.processed_image = image

    def get_processed_image(self) -> np.ndarray:
        """
        Retrieve the processed image.
        """
        return self.processed_image

    def set_blurred_image(self, image: cp.ndarray):
        """
        Set the blurred image.
        
        :param image: CuPy array representing the blurred image (pixels).
        """
        self.blurred_image = image

    def get_blurred_image(self):
        """
        Get the blurred image.
        
        :return: CuPy array representing the blurred image (pixels).
        """
        return self.blurred_image

    def set_highpass_image(self, image: cp.ndarray):
        """
        Set the highpass image.
        
        :param image: CuPy array representing the highpass image (pixels).
        """
        self.highpass_image = image

    def get_highpass_image(self):
        """
        Get the highpass image.
        
        :return: CuPy array representing the highpass image (pixels).
        """
        return self.highpass_image

    def add_intermediate_image(self, key: str, image: cp.ndarray):
        """
        Add an intermediate image state.
        """
        self.intermediate_images[key] = image

    def get_intermediate_image(self, key: str) -> cp.ndarray:
        """
        Retrieve an intermediate image state.
        """
        return self.intermediate_images.get(key, None)

    def reset_images(self):
        """
        Reset processed and intermediate images.
        """
        self.processed_image = None
        self.intermediate_images.clear()
        self.blurred_image = None
        self.highpass_image = None