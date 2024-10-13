from models import ImageModel
from processing.mask_generator import MaskGenerator
from views import MainWindow, Phase1View, Phase2View
from processing.fft_processor import FFTProcessor
from utils.file_handler import load_image, save_image
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

from PyQt5 import QtWidgets, QtCore
import os
import cv2
import cupy as cp
from controllers.image_controller import ImageController
from controllers.processing_controller import ProcessingController  # Ensure ProcessingController is imported
import logging

class MainController:
    def __init__(self):
        # Replace the existing logging initialization with the following
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize ConfigManager
        self.config_manager = ConfigManager()

        # Initialize models with ConfigManager
        self.image_model = ImageModel()

        # Initialize processing components
        self.mask_generator = MaskGenerator(config_manager=self.config_manager)
        self.fft_processor = FFTProcessor(config_manager=self.config_manager)

        # Initialize controllers
        self.processing_controller = ProcessingController(
            image_model=self.image_model,
            config_manager=self.config_manager,
            fft_processor=self.fft_processor,
            mask_generator=self.mask_generator,
            main_controller=self  # Pass MainController instance
        )  # Initialize ProcessingController
        self.image_controller = ImageController(
            image_model=self.image_model,
            config_manager=self.config_manager,
            processing_controller=self.processing_controller
        )  # Initialize ImageController

        # Initialize views
        self.phase1_view = Phase1View(self)
        self.phase2_view = Phase2View(self)

        # Initialize main window
        self.window = MainWindow(self)

        # Connect signals
        self.processing_controller.processing_completed.connect(self.window.update_image_display)
        self.image_controller.image_loaded.connect(self.on_image_loaded)
        self.window.show()

        # Set padding sizes using centralized configuration
        self.padding_size = self.config_manager.get_parameter("Padding Size", PADDING_SIZE)  # Ensure "Padding Size" exists in DEFAULT_CONFIG
        self.max_high_pass_radius = self.config_manager.get_parameter("Max High Pass Radius", MAX_HIGH_PASS_RADIUS)
        self.min_high_pass_radius = self.config_manager.get_parameter("Min High Pass Radius", MIN_HIGH_PASS_RADIUS)

    def on_image_loaded(self, success: bool):
        """Handle the image_loaded signal from ImageController."""
        if success:
            self.logger.info("Image loaded successfully. Processing image...")
            self.update_image()
        else:
            self.logger.error("Image loading failed. Cannot process image.")

    def run(self):
        self.logger.info("Running the application.")
        self.window.show()

    def load_image(self, file_path):
        """Handle loading of an image given the file path."""
        if file_path:
            # Initiate asynchronous image loading
            self.image_controller.load_image_from_file_async(file_path)
        else:
            self.logger.warning("No file selected to load.")
            print("No file selected.")

    def save_image(self, save_path):
        self.image_controller.save_processed_image(save_path)

    def batch_process_images(self):
        """Batch process images in a selected directory."""
        input_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.window, "Select Input Directory"
        )
        if input_dir:
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.window, "Select Output Directory"
            )
            if not output_dir:
                QtWidgets.QMessageBox.warning(
                    self.window, "Error", "No output directory selected."
                )
                return

            self.image_controller.batch_process_images(input_dir, output_dir)

    def update_image(self):
        """Process and update the image based on current parameters."""
        if self.image_model.original_image is None:
            raise ValueError("No image loaded. Please load an image before processing.")

        self.processing_controller.process_image(current_tab="Phase 1")

    def get_processed_image(self):
        return self.image_model.processed_image

    def get_phase1_view(self):
        return self.phase1_view

    def get_phase2_view(self):
        return self.phase2_view

    def set_hovered_slider(self, slider_name):
        self.hovered_slider = slider_name

    def update_parameters(self, param_name, param_value):
        """Update a parameter in the ConfigManager and process the image."""
        # Add the following line at the beginning of the method
        self.logger.debug(f"Updating parameter: {param_name} = {param_value}")
        self.config_manager.set_parameter(param_name, param_value)
        self.update_image()

    def get_images_for_display(self):
        """Retrieve images for display."""
        return (
            self.image_model.original_image,
            self.image_model.blurred_image,
            self.image_model.highpass_image
        )

    def initiate_phase2_processing(self):
        """Start processing when entering Phase 2."""
        current_tab = self.window.tabs.tabText(self.window.tabs.currentIndex())
        if current_tab == "2. Detailed Filtering":
            # Replace the existing logging line with the following
            self.logger.info("Initiating FFT processing for Phase 2.")
            self.processing_controller.process_image(current_tab="Phase 2")
        else:
            # Replace the existing logging line with the following
            self.logger.info("FFT processing not initiated as current tab is not Phase 2.")