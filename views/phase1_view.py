from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider

class Phase1View(QWidget):
    """
    View for Phase 1: High-Pass Filtering.
    
    Provides controls for adjusting the high-pass filter radius.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_phase1_ui()

    def init_phase1_ui(self):
        """Initialize the UI components for Phase 1."""
        layout = QVBoxLayout(self)

        # High-Pass Filter Radius Slider
        self.high_pass_slider = QSlider(QtCore.Qt.Horizontal)
        self.high_pass_slider.setMinimum(0)
        self.high_pass_slider.setMaximum(100)
        self.high_pass_slider.setValue(10)  # Default value
        self.high_pass_slider.setTickPosition(QSlider.TicksBelow)
        self.high_pass_slider.setTickInterval(10)
        self.high_pass_slider.valueChanged.connect(self.on_slider_changed)

        # Label to show current slider value
        self.high_pass_label = QLabel("High-Pass Filter Radius: 10")

        # Add widgets to layout
        layout.addWidget(self.high_pass_label)
        layout.addWidget(self.high_pass_slider)

    def on_slider_changed(self, value):
        """Handle slider value changes."""
        self.high_pass_label.setText(f"High-Pass Filter Radius: {value}")
        self.controller.update_parameters("High-Pass Filter Radius", value)
        self.controller.update_image()