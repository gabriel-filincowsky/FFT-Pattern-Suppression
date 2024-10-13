import os
import cv2
import numpy as np
import cupy as cp
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def load_image(file_path: str) -> cp.ndarray:
    """
    Load an image from the specified file path efficiently.
    
    :param file_path: Path to the image file.
    :return: Image as a CuPy array in RGB format, or None if loading fails.
    """
    if isinstance(file_path, tuple):
        file_path = file_path[0]
    if not isinstance(file_path, (str, os.PathLike)):
        raise TypeError(f"Expected 'file_path' to be a string or path-like object, got {type(file_path)} instead.")

    # Load image in BGR format
    image_np = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image_np is None:
        raise ValueError(f"Failed to load image from {file_path}. Please check the file path and try again.")

    # Convert to RGB format and transfer to GPU memory
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_cp = cp.asarray(image_rgb, dtype=cp.float32)
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