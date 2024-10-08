from utils.file_handler import load_image, save_image, select_input_directory, select_output_directory
from models.image_model import ImageModel
from models.parameters_model import ParametersModel
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
import os
import logging
from controllers.processing_controller import ProcessingController

class ImageController:
    """
    Controller for managing image loading, saving, and batch processing.
    """
    def __init__(self, image_model: ImageModel, parameters_model: ParametersModel, processing_controller: ProcessingController):
        self.image_model = image_model
        self.parameters_model = parameters_model
        self.processing_controller = processing_controller
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_image_from_file(self, file_path: str) -> bool:
        """
        Load a single image and update the model.

        :param file_path: Path to the image file.
        :return: True if successful, False otherwise.
        """
        image = load_image(file_path)
        if image is not None:
            self.image_model.set_original_image(image)
            self.logger.info(f"Loaded image from {file_path}")
            return True
        self.logger.error(f"Failed to load image from {file_path}")
        return False

    def save_processed_image(self, save_path: str) -> bool:
        """
        Save the processed image to disk.

        :param save_path: Path to save the image.
        :return: True if successful, False otherwise.
        """
        if self.image_model.processed_image is not None:
            success = save_image(self.image_model.processed_image, save_path)
            if success:
                self.logger.info(f"Saved processed image to {save_path}")
                return True
        self.logger.error("No processed image to save.")
        return False

    def select_input_directory_for_batch(self) -> str:
        """
        Open a dialog to select the input directory for batch processing.

        :return: Path to the selected input directory, or an empty string if canceled.
        """
        return select_input_directory()

    def select_output_directory_for_batch(self) -> str:
        """
        Open a dialog to select the output directory for batch processing.

        :return: Path to the selected output directory, or an empty string if canceled.
        """
        return select_output_directory()

    def batch_process_images(self, input_dir: str, output_dir: str) -> None:
        """
        Batch process images from input directory and save to output directory.

        :param input_dir: Directory containing input images.
        :param output_dir: Directory to save processed images.
        """
        if not os.path.exists(input_dir):
            QMessageBox.warning(None, "Batch Processing", f"Input directory {input_dir} does not exist.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory at {output_dir}")

        image_files = [f for f in os.listdir(input_dir) if self.is_image_file(f)]
        total_files = len(image_files)
        if total_files == 0:
            QMessageBox.warning(None, "Batch Processing", "No image files found in the input directory.")
            return

        progress = QProgressDialog("Processing images...", "Cancel", 0, total_files)
        progress.setWindowTitle("Batch Processing")
        progress.setWindowModality(Qt.WindowModal)

        for i, file_name in enumerate(image_files, start=1):
            if progress.wasCanceled():
                self.logger.info("Batch processing was canceled by the user.")
                break
            file_path = os.path.join(input_dir, file_name)
            self.logger.info(f"Processing image: {file_path}")
            success = self.load_image_from_file(file_path)
            if success:
                self.processing_controller.process_image()
                save_path = os.path.join(output_dir, file_name)
                self.save_processed_image(save_path)
            progress.setValue(i)

        progress.setValue(total_files)
        QMessageBox.information(None, "Batch Processing", "Batch processing completed successfully.")

    def is_image_file(self, filename: str) -> bool:
        """
        Check if a file is an image based on its extension.

        :param filename: Name of the file.
        :return: True if it's an image file, False otherwise.
        """
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        return filename.lower().endswith(valid_extensions)