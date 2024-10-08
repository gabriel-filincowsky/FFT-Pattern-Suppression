from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider

class Phase2View(QWidget):
    """
    View for Phase 2: Detailed Filtering.
    
    Provides controls for adjusting detailed filtering parameters like Gaussian blur.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_phase2_ui()

    def init_phase2_ui(self):
        """Initialize the UI components for Phase 2."""
        layout = QVBoxLayout(self)

        # Gaussian Blur Slider
        self.gaussian_blur_slider = QSlider(QtCore.Qt.Horizontal)
        self.gaussian_blur_slider.setMinimum(0)
        self.gaussian_blur_slider.setMaximum(100)
        self.gaussian_blur_slider.setValue(10)  # Default value
        self.gaussian_blur_slider.setTickPosition(QSlider.TicksBelow)
        self.gaussian_blur_slider.setTickInterval(10)
        self.gaussian_blur_slider.valueChanged.connect(self.on_slider_changed)

        # Label to show current slider value
        self.gaussian_blur_label = QLabel("Gaussian Blur (%): 10")

        # Add widgets to layout
        layout.addWidget(self.gaussian_blur_label)
        layout.addWidget(self.gaussian_blur_slider)

    def on_slider_changed(self, value):
        """Handle slider value changes."""
        self.gaussian_blur_label.setText(f"Gaussian Blur (%): {value}")
        self.controller.update_parameters("Gaussian Blur (%)", value)
        self.controller.update_image()