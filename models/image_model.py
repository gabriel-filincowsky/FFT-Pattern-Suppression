import numpy as np

class ImageModel:
    """
    Represents the image data structure, storing original, processed, and intermediate images.
    """
    def __init__(self):
        self.original_image = None  # Original color image as a NumPy array
        self.processed_image = None  # Final processed image as a NumPy array
        self.intermediate_images = {}  # Dictionary to store intermediate image states

    def set_original_image(self, image: np.ndarray):
        """
        Set the original image.
        
        :param image: NumPy array representing the original image.
        """
        self.original_image = image
        self.reset_images()

    def set_processed_image(self, image: np.ndarray):
        """
        Set the processed image.
        
        :param image: NumPy array representing the processed image.
        """
        self.processed_image = image

    def add_intermediate_image(self, key: str, image: np.ndarray):
        """
        Add an intermediate image state.
        
        :param key: Identifier for the intermediate image.
        :param image: NumPy array representing the intermediate image.
        """
        self.intermediate_images[key] = image

    def reset_images(self):
        """
        Reset processed and intermediate images.
        """
        self.processed_image = None
        self.intermediate_images.clear()