from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QTabWidget, QWidget, QLabel, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cupy as cp  # Added for CuPy compatibility
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
from utils.config_manager import ConfigManager  # Ensure ConfigManager is imported
from utils.constants import PADDING_SIZE, DEFAULT_COLOR  # Import all necessary constants

class MainWindow(QtWidgets.QMainWindow):
    """
    Main Window of the FFT-Based Image Processing Application.

    Integrates different phase views and provides a central canvas for image display.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.config_manager = controller.config_manager  # Access ConfigManager
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
        self.setWindowTitle("FFT-Based Image Processing Application")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Tab widget for different phases
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Phase 1 and Phase 2 Views
        self.phase1_view = self.controller.get_phase1_view()
        self.phase2_view = self.controller.get_phase2_view()
        self.tabs.addTab(self.phase1_view, "1. High-Pass Filter")
        self.tabs.addTab(self.phase2_view, "2. Detailed Filtering")

        # Image Display Canvas using QLabel and QPixmap
        self.processed_image_label = QLabel(self)
        self.processed_image_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.processed_image_label)

        # FFT Visualization Canvas using Matplotlib
        self.fft_figure = Figure(figsize=(6, 4))
        self.fft_canvas = FigureCanvas(self.fft_figure)
        self.fft_toolbar = NavigationToolbar(self.fft_canvas, self)
        main_layout.addWidget(self.fft_toolbar)
        main_layout.addWidget(self.fft_canvas)

        # Menu Bar and Actions
        self.setup_menu()

        # Connect 'Next' button signal to switch tab
        self.phase1_view.next_phase.connect(self.switch_to_phase2)

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Load Image Action
        load_action = QtWidgets.QAction('Load Image', self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        # Save Image Action
        save_action = QtWidgets.QAction('Save Image', self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        # Batch Process Action
        batch_action = QtWidgets.QAction('Batch Process', self)
        batch_action.triggered.connect(self.batch_process)
        file_menu.addAction(batch_action)

    def load_image(self):
        """Handle loading of an image."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            options=options,
        )
        if file_path:
            self.controller.load_image(file_path)

    def save_image(self):
        """Handle saving of the processed image."""
        if self.controller.image_model.processed_image is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "No image to save.")
            return

        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif *.tiff)",
            options=options,
        )
        if save_path:
            # Access processed image and save via ConfigManager if needed
            success = self.controller.save_image(save_path)
            if success:
                QtWidgets.QMessageBox.information(
                    self, "Image Saved", f"Image saved to {save_path}"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Save Error", f"Failed to save the image to {save_path}."
                )

    def batch_process(self):
        """Handle batch processing of images."""
        self.controller.batch_process_images()

    def update_image_display(self):
        """Update the processed image displayed in the UI."""
        processed_image = self.controller.image_model.get_processed_image()
        if processed_image is not None:
            # Convert CuPy array to NumPy
            if isinstance(processed_image, cp.ndarray):
                processed_image = cp.asnumpy(processed_image)
            
            # Ensure image is in uint8
            processed_image = processed_image.astype(np.uint8)
            
            # Convert RGB to QImage
            height, width, channel = processed_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_img)
            
            # Maintain even dimensions in display
            if height % 2 != 0 or width % 2 != 0:
                pixmap = self.ensure_even_dimensions(pixmap)
            
            # Set QPixmap to QLabel
            self.processed_image_label.setPixmap(pixmap.scaled(
                self.processed_image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            ))

    def ensure_even_dimensions(self, pixmap: QPixmap) -> QPixmap:
        """Ensure the QPixmap has even dimensions by adding a 1-pixel border if necessary."""
        width = pixmap.width()
        height = pixmap.height()
        new_width = width + 1 if width % 2 != 0 else width
        new_height = height + 1 if height % 2 != 0 else height

        if new_width != width or new_height != height:
            new_pixmap = QPixmap(new_width, new_height)
            new_pixmap.fill(QtCore.Qt.gray)  # Fill with neutral gray
            painter = QtGui.QPainter(new_pixmap)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()
            return new_pixmap
        return pixmap

    def switch_to_phase2(self):
        """Enable and switch to Phase 2 tab."""
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        self.controller.initiate_phase2_processing()