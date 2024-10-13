from utils.file_handler import load_image, save_image, select_input_directory, select_output_directory
from models.image_model import ImageModel
from utils.config_manager import ConfigManager
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import os
import logging
from controllers.processing_controller import ProcessingController
from concurrent.futures import ThreadPoolExecutor

class ImageController(QObject):
    """
    Controller for managing image loading, saving, and batch processing.
    """
    # Define a signal to notify when an image is loaded
    image_loaded = pyqtSignal(bool)  # Emits True on success, False on failure

    def __init__(self, image_model: ImageModel, config_manager: ConfigManager, processing_controller: ProcessingController):
        super().__init__()
        self.config_manager = config_manager
        self.image_model = image_model
        self.processing_controller = processing_controller
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on GPU capabilities

    def load_image_from_file_async(self, file_path: str) -> None:
        """Asynchronously load an image and update the model."""
        def task():
            try:
                image = load_image(file_path)
                self.image_model.set_original_image(image)
                self.logger.info(f"Loaded image from {file_path}")
                self.image_loaded.emit(True)  # Emit success
            except Exception as e:
                self.logger.error(f"Failed to load image from {file_path}: {e}")
                QMessageBox.warning(None, "Load Error", f"Failed to load image from {file_path}\nError: {e}")
                self.image_loaded.emit(False)  # Emit failure

        self.executor.submit(task)

    def load_image_from_file(self, file_path: str) -> bool:
        """
        Initiate asynchronous loading of an image.
        
        :param file_path: Path to the image file.
        :return: True if loading is initiated, False otherwise.
        """
        if file_path:
            self.load_image_from_file_async(file_path)
            return True
        else:
            self.logger.warning("No file selected to load.")
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

    def batch_process_images_async(self, input_dir: str, output_dir: str, progress_dialog: QProgressDialog) -> None:
        """Asynchronously batch process images."""
        def task(file_name):
            try:
                file_path = os.path.join(input_dir, file_name)
                image = load_image(file_path)
                self.image_model.set_original_image(image)
                self.processing_controller.process_image(current_tab="Phase 2")
                save_path = os.path.join(output_dir, file_name)
                save_image(self.image_model.processed_image, save_path)
                self.logger.info(f"Processed and saved image: {file_name}")
            except Exception as e:
                self.logger.error(f"Error processing {file_name}: {e}")

        image_files = [f for f in os.listdir(input_dir) if self.is_image_file(f)]
        total_files = len(image_files)
        if total_files == 0:
            QMessageBox.warning(None, "Batch Processing", "No image files found in the input directory.")
            return

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, file_name in enumerate(image_files, start=1):
                if progress_dialog.wasCanceled():
                    self.logger.info("Batch processing was canceled by the user.")
                    break
                futures.append(executor.submit(task, file_name))
                progress_dialog.setValue(i)
            for future in futures:
                future.result()  # To catch exceptions if any

        progress_dialog.setValue(len(image_files))
        QMessageBox.information(None, "Batch Processing", "Batch processing completed successfully.")

    def batch_process_images(self, input_dir: str, output_dir: str) -> None:
        """
        Batch process images from input directory and save to output directory asynchronously.
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
        progress.show()

        self.batch_process_images_async(input_dir, output_dir, progress)

    def is_image_file(self, filename: str) -> bool:
        """Check if a file is a supported image format."""
        return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))