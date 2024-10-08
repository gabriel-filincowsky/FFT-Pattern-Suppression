import os
import cv2
import numpy as np
import cupy as cp
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def load_image(file_path: str) -> cp.ndarray:
    """
    Load an image from the specified file path.

    :param file_path: Path to the image file.
    :return: Image as a CuPy array in BGR format, or None if loading fails.
    """
    if not os.path.exists(file_path):
        QMessageBox.warning(None, "Load Image", f"The file {file_path} does not exist.")
        return None

    image_np = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image_np is None:
        QMessageBox.warning(None, "Load Image", f"Failed to load the image from {file_path}.")
        return None

    # Convert to CuPy array
    image_cp = cp.asarray(image_np)
    return image_cp

def save_image(image: cp.ndarray, save_path: str) -> bool:
    """
    Save an image to the specified file path.

    :param image: Image as a CuPy array.
    :param save_path: Path where the image will be saved.
    :return: True if saving is successful, False otherwise.
    """
    try:
        # Convert to NumPy array before saving
        image_np = cp.asnumpy(image)
        success = cv2.imwrite(save_path, image_np)
        if not success:
            QMessageBox.warning(None, "Save Image", f"Failed to save the image to {save_path}.")
        return success
    except Exception as e:
        QMessageBox.warning(None, "Save Image", f"Error saving image: {e}")
        return False

def select_input_directory() -> str:
    """
    Open a dialog to select the input directory for batch processing.

    :return: Path to the selected input directory, or an empty string if canceled.
    """
    dialog = QFileDialog()
    directory = dialog.getExistingDirectory(None, "Select Input Directory")
    return directory if directory else ""

def select_output_directory() -> str:
    """
    Open a dialog to select the output directory for batch processing.

    :return: Path to the selected output directory, or an empty string if canceled.
    """
    dialog = QFileDialog()
    directory = dialog.getExistingDirectory(None, "Select Output Directory")
    return directory if directory else ""