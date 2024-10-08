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

        # Padding Size Slider and SpinBox
        padding_layout = QtWidgets.QHBoxLayout()
        info_button = self.create_info_icon("Adjusts the padding size applied during FFT processing.")
        lbl = QtWidgets.QLabel("Padding Size")

        # Slider
        sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setValue(16)  # Default padding size
        sld.valueChanged.connect(lambda value: self.controller.update_parameters("padding_size", value))
        sld.setObjectName("Padding Size")
        sld.installEventFilter(self)

        # SpinBox
        val_input = QtWidgets.QSpinBox()
        val_input.setMinimum(0)
        val_input.setMaximum(100)
        val_input.setValue(16)
        val_input.valueChanged.connect(lambda value: self.controller.update_parameters("padding_size", value))

        # Store widgets if necessary
        self.sliders_widgets["Padding Size"] = sld
        self.slider_values["Padding Size"] = val_input
        self.sliders["Padding Size"] = (sld, 0, 100)

        # Add to layout
        padding_layout.addWidget(info_button)
        padding_layout.addWidget(lbl)
        padding_layout.addWidget(sld)
        padding_layout.addWidget(val_input)
        layout.addLayout(padding_layout)

        # Add widgets to layout
        layout.addWidget(self.high_pass_label)
        layout.addWidget(self.high_pass_slider)

    def on_slider_changed(self, value):
        """Handle slider value changes."""
        self.high_pass_label.setText(f"High-Pass Filter Radius: {value}")
        self.controller.update_parameters("High-Pass Filter Radius", value)
        self.controller.update_image()