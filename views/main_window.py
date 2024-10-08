from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    """
    Main Window of the FFT-Based Image Processing Application.
    
    Integrates different phase views and provides a central canvas for image display.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
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
        
        # Phase 1 Tab
        self.phase1_tab = QWidget()
        self.tabs.addTab(self.phase1_tab, "High-Pass Filter")
        self.init_phase1_ui()
        
        # Phase 2 Tab
        self.phase2_tab = QWidget()
        self.tabs.addTab(self.phase2_tab, "Detailed Filtering")
        self.init_phase2_ui()
        
        # Image Display Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        
        # Toolbar for matplotlib
        self.toolbar = QtWidgets.QToolBar()
        self.addToolBar(self.toolbar)
        self.toolbar.addWidget(QtWidgets.QLabel("Image Display:"))
        
        # Menu Bar
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
        
    def init_phase1_ui(self):
        """Initialize the UI components for Phase 1."""
        phase1_layout = QtWidgets.QVBoxLayout(self.phase1_tab)
        
        # High-Pass Filter Radius Slider
        self.high_pass_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.high_pass_slider.setMinimum(0)
        self.high_pass_slider.setMaximum(100)
        self.high_pass_slider.setValue(10)  # Default value
        self.high_pass_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.high_pass_slider.setTickInterval(10)
        self.high_pass_slider.valueChanged.connect(self.on_high_pass_slider_changed)
        
        # Label to show current slider value
        self.high_pass_label = QLabel("High-Pass Filter Radius: 10")
        
        # Add widgets to layout
        phase1_layout.addWidget(self.high_pass_label)
        phase1_layout.addWidget(self.high_pass_slider)
    
    def init_phase2_ui(self):
        """Initialize the UI components for Phase 2."""
        phase2_layout = QtWidgets.QVBoxLayout(self.phase2_tab)
        
        # Detailed Filtering Controls (e.g., additional sliders, buttons)
        # Example: Gaussian Blur Slider
        self.gaussian_blur_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gaussian_blur_slider.setMinimum(0)
        self.gaussian_blur_slider.setMaximum(100)
        self.gaussian_blur_slider.setValue(10)  # Default value
        self.gaussian_blur_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.gaussian_blur_slider.setTickInterval(10)
        self.gaussian_blur_slider.valueChanged.connect(self.on_gaussian_blur_slider_changed)
        
        # Label to show current slider value
        self.gaussian_blur_label = QLabel("Gaussian Blur (%): 10")
        
        # Add widgets to layout
        phase2_layout.addWidget(self.gaussian_blur_label)
        phase2_layout.addWidget(self.gaussian_blur_slider)
        
        # Additional controls can be added similarly
        
    def load_image(self):
        """Handle loading of an image."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", 
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)", 
                                                   options=options)
        if file_path:
            self.controller.load_image(file_path)
    
    def save_image(self):
        """Handle saving of the processed image."""
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                                                  "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;TIFF Files (*.tiff *.tif)", 
                                                  options=options)
        if save_path:
            self.controller.save_image(save_path)
    
    def batch_process(self):
        """Handle batch processing of images."""
        self.controller.batch_process_images()
    
    def on_high_pass_slider_changed(self, value):
        """Handle changes to the High-Pass Filter Radius slider."""
        self.high_pass_label.setText(f"High-Pass Filter Radius: {value}")
        self.controller.update_parameters("High-Pass Filter Radius", value)
        self.update_image_display()
    
    def on_gaussian_blur_slider_changed(self, value):
        """Handle changes to the Gaussian Blur slider."""
        self.gaussian_blur_label.setText(f"Gaussian Blur (%): {value}")
        self.controller.update_parameters("Gaussian Blur (%)", value)
        self.update_image_display()
    
    def update_image_display(self):
        """Update the image displayed on the canvas."""
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        
        # Display Original Image
        if self.controller.image_model.original_image is not None:
            ax.imshow(self.controller.image_model.original_image.astype(np.uint8))
            ax.set_title("Original Image")
            ax.axis("off")
        
        # Display Processed Image
        if self.controller.image_model.processed_image is not None:
            ax.imshow(self.controller.image_model.processed_image.astype(np.uint8))
            ax.set_title("Processed Image")
            ax.axis("off")
        
        self.canvas.draw()