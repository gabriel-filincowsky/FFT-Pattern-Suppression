from models.image_model import ImageModel
from models.parameters_model import ParametersModel
from views.main_window import MainWindow
from processing.fft_processor import FFTProcessor
from processing.mask_generator import MaskGenerator
from controllers.image_controller import ImageController
from controllers.processing_controller import ProcessingController
from PyQt5.QtWidgets import QMessageBox
import sys
import logging

class MainController:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Models
        self.image_model = ImageModel()
        self.parameters_model = ParametersModel()
        
        # Initialize Processing Modules
        self.fft_processor = FFTProcessor()
        self.mask_generator = MaskGenerator()
        
        # Initialize Sub-controllers
        self.image_controller = ImageController(self.image_model, self.parameters_model)
        self.processing_controller = ProcessingController(
            self.image_model, self.parameters_model, self.fft_processor, self.mask_generator
        )
        
        # Initialize Main Window
        self.view = MainWindow(self)
        
    def load_image(self, file_path):
        """
        Load an image and update the model.
        
        :param file_path: Path to the image file to load.
        """
        success = self.image_controller.load_image_from_file(file_path)
        if success:
            self.logger.info(f"Image loaded successfully from {file_path}")
            self.update_view()
        else:
            self.logger.error(f"Failed to load image from {file_path}")
            QMessageBox.warning(self.view, "Load Image", "Failed to load the selected image.")
    
    def save_image(self, file_path):
        """
        Save the processed image.
        
        :param file_path: Path to save the processed image.
        """
        success = self.image_controller.save_processed_image(file_path)
        if success:
            self.logger.info(f"Image saved successfully to {file_path}")
            QMessageBox.information(self.view, "Save Image", "Image saved successfully.")
        else:
            self.logger.error(f"Failed to save image to {file_path}")
            QMessageBox.warning(self.view, "Save Image", "Failed to save the processed image.")
    
    def update_parameters(self, parameter_key, value):
        """
        Update a processing parameter.
        
        :param parameter_key: The key/name of the parameter to update.
        :param value: The new value for the parameter.
        """
        self.parameters_model.set_parameter(parameter_key, value)
        self.parameters_model.save_parameters()
        self.logger.info(f"Parameter '{parameter_key}' updated to {value}")
        self.update_image()
    
    def update_image(self):
        """
        Update the processed image based on current parameters.
        """
        if self.image_model.original_image is not None:
            self.processing_controller.process_image()
            self.view.update_image_display()
        else:
            self.logger.warning("No image loaded to update.")
    
    def batch_process_images(self, input_dir, output_dir):
        """
        Batch process images from input directory and save to output directory.
        
        :param input_dir: Directory containing input images.
        :param output_dir: Directory to save processed images.
        """
        self.image_controller.batch_process_images(input_dir, output_dir, self.processing_controller.process_image)
    
    def run(self):
        """
        Run the application.
        """
        self.view.show()