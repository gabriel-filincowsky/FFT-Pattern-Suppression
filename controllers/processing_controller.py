from models.image_model import ImageModel
from utils.config_manager import ConfigManager
from processing.fft_processor import FFTProcessor
from processing.mask_generator import MaskGenerator
from utils.cupy_handler import normalize_spectrum, apply_gamma_correction
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
import numpy as np
import cupy as cp
from skimage.feature import peak_local_max
import logging
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject, QThread

class ProcessingThread(QThread):
    processing_done = pyqtSignal()
    
    def __init__(self, processor, image, parameters):
        super().__init__()
        self.processor = processor
        self.image = image
        self.parameters = parameters

    def run(self):
        self.processor.process_image(self.image, self.parameters)
        self.processing_done.emit()

class ProcessingController(QObject):
    """
    Controller for performing image processing tasks.
    """
    processing_completed = pyqtSignal()

    def __init__(self, image_model: ImageModel, config_manager: ConfigManager,
                 fft_processor: FFTProcessor, mask_generator: MaskGenerator, main_controller):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.image_model = image_model
        self.fft_processor = fft_processor
        self.mask_generator = mask_generator
        self.main_controller = main_controller
        self.padding_size = PADDING_SIZE  # pixels
        self.max_high_pass_radius = MAX_HIGH_PASS_RADIUS  # pixels
        self.min_high_pass_radius = MIN_HIGH_PASS_RADIUS  # pixels
        self.logger.info("ProcessingController initialized.")

    def process_image(self, current_tab="Phase 1"):
        """
        Process the image based on the current phase and parameters.
        
        :param current_tab: String indicating the current phase.
        """
        self.logger.info(f"Processing image for {current_tab}.")
        if self.image_model.original_image is None:
            self.logger.error("No image loaded. Cannot process.")
            return

        # Start processing in a separate thread to keep UI responsive
        self.logger.debug("Starting processing thread.")
        # Pre-fetch all parameters to minimize per-call config access
        parameters = self.config_manager.get_all_parameters()
        self.thread = ProcessingThread(self.fft_processor, self.image_model.original_image, parameters)
        self.thread.processing_done.connect(self.on_processing_done)
        self.thread.start()
        self.logger.info("Processing thread started.")

    def on_processing_done(self):
        """
        Callback when image processing is complete.
        """
        self.logger.debug("Processing thread completed.")
        self.image_model.processed_image = self.fft_processor.final_image
        self.image_model.highpass_image = self.fft_processor.highpass_image
        self.processing_completed.emit()
        self.logger.info("Processing completed and images updated.")