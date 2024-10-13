from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider
import numpy as np  # If NumPy is used

class Phase2View(QWidget):
    """
    View for Phase 2: Detailed Filtering.
    
    Provides controls for adjusting detailed filtering parameters like Gaussian blur.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        # Initialize dictionaries for sliders and widgets
        self.sliders_widgets = {}
        self.slider_values = {}
        self.sliders = {}

        # Initialize checkboxes
        self.checkboxes = {}

        self.init_phase2_ui()

    def init_phase2_ui(self):
        """Initialize the UI components for Phase 2."""
        layout = QVBoxLayout(self)

        # Parameters for sliders
        # **Added Explicit Parameter Type Flag**
        slider_params = [
            ("Gaussian Blur (%)", 0.0, 5.0, 1.0, "Adjusts the amount of Gaussian blur applied to the FFT magnitude spectrum.", True),
            ("Peak Min Distance", 1, 50, 10, "Minimum distance between detected peaks in the frequency domain.", False),
            ("Peak Threshold", 0.0000, 0.0150, 0.0010, "Threshold for peak detection in the frequency domain.", True),
            ("Mask Radius (%)", 0.1, 10.0, 1.0, "Radius of the circular mask applied to suppress detected peaks.", True),
            ("Peak Mask Falloff (%)", 0.0, 5.0, 0.0, "Smoothness of the transition at the edge of the peak masks.", True),
            ("Radius (%)", 0.0, 50.0, 10.0, "Radius of the central area to preserve in the frequency domain.", True),
            ("Aspect Ratio", 0.1, 5.0, 1.0, "Aspect ratio of the preserved central area.", True),
            ("Orientation", 0, 180, 0, "Rotation angle of the preserved central area.", False),
            ("Falloff (%)", 0.0, 5.0, 0.0, "Smoothness of the transition at the edge of the preserved area.", True),
            ("Gamma Correction", 0.1, 1.0, 1.0, "Adjusts the gamma value used for attenuation of frequencies.", True),
            ("Anti-Aliasing Intensity (%)", 0.0, 100.0, 50.0, "Controls the intensity of the anti-aliasing filter applied.", True),
        ]

        # Create sliders
        # **Updated for Loop to Unpack the Explicit Flag**
        for label, min_val, max_val, init_val, tooltip, is_float in slider_params:
            slider_layout = QtWidgets.QHBoxLayout()
            info_button = self.create_info_icon(tooltip)
            lbl = QtWidgets.QLabel(label)

            # Slider
            sld = QSlider(QtCore.Qt.Horizontal)
            sld.setMinimum(0)
            sld.setMaximum(1000)
            sld.setValue(int((init_val - min_val) / (max_val - min_val) * sld.maximum()))
            sld.valueChanged.connect(self.on_slider_changed)
            sld.setObjectName(label)
            sld.installEventFilter(self)

            # **Updated Condition for SpinBox Type**
            if is_float:
                val_input = QtWidgets.QDoubleSpinBox()
                val_input.setDecimals(4 if "Threshold" in label else 3)  # Adjust decimal places as needed
                val_input.setSingleStep(0.1)
            else:
                val_input = QtWidgets.QSpinBox()
                # Removed the following line to prevent AttributeError
                # val_input.setDecimals(0)

            # **Updated setRange with Error Handling**
            try:
                val_input.setRange(min_val, max_val)  # Now passes correct types
            except TypeError as e:
                print(f"Error setting range for {label}: {e}")

            val_input.setValue(init_val)
            val_input.valueChanged.connect(self.on_spinbox_changed)
            val_input.setObjectName(f"{label}_spinbox")

            # Store widgets
            self.sliders_widgets[label] = sld
            self.slider_values[label] = val_input
            self.sliders[label] = (sld, min_val, max_val)

            # Add to layout
            slider_layout.addWidget(info_button)
            slider_layout.addWidget(lbl)
            slider_layout.addWidget(sld)
            slider_layout.addWidget(val_input)
            layout.addLayout(slider_layout)

        # Initialize checkboxes
        checkbox_params = [
            ("Enable Frequency Peak Suppression", False, "Toggle frequency peak suppression to reduce periodic patterns."),
            ("Enable Attenuation", False, "Enable attenuation using gamma correction to smoothly reduce the intensity of detected peaks."),
            ("Enable Anti-Aliasing Filter", False, "Apply an anti-aliasing filter to smooth out high-frequency components."),
        ]

        for label, checked, tooltip in checkbox_params:
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setObjectName(label)
            checkbox.setChecked(checked)
            checkbox.stateChanged.connect(self.on_checkbox_changed)
            checkbox.setToolTip(tooltip)
            self.checkboxes[label] = checkbox
            layout.addWidget(checkbox)

    def create_info_icon(self, tooltip_text):
        """Create an info icon with tooltip."""
        info_button = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        painter.drawEllipse(1, 1, 14, 14)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "i")
        painter.end()
        info_button.setPixmap(pixmap)
        info_button.setToolTip(tooltip_text)
        info_button.setAlignment(QtCore.Qt.AlignCenter)
        info_button.setFixedWidth(20)
        return info_button

    def on_slider_changed(self, value):
        """Handle slider value changes."""
        sender = self.sender()
        label = sender.objectName()
        min_val, max_val = self.sliders[label][1:]
        val = value / sender.maximum() * (max_val - min_val) + min_val

        val_input = self.slider_values[label]
        val_input.blockSignals(True)
        if isinstance(val_input, QtWidgets.QSpinBox):
            val = int(round(val))
        else:
            val = round(val, 4 if label == "Peak Threshold" else 3)
        val_input.setValue(val)
        val_input.blockSignals(False)
        self.controller.update_parameters(label, val)
        self.controller.update_image()

    def on_spinbox_changed(self, value):
        """Handle spinbox value changes."""
        sender = self.sender()
        label = sender.objectName().replace("_spinbox", "")
        sld, min_val, max_val = self.sliders[label]
        sld.blockSignals(True)
        sld.setValue(int((value - min_val) / (max_val - min_val) * sld.maximum()))
        sld.blockSignals(False)
        self.controller.update_parameters(label, value)
        self.controller.update_image()

    def on_checkbox_changed(self, state):
        """Handle checkbox state changes."""
        sender = self.sender()
        label = sender.objectName()
        value = sender.isChecked()
        self.controller.update_parameters(label, value)
        self.controller.update_image()

    def eventFilter(self, source, event):
        """Event filter to detect when hovering over the sliders."""
        if event.type() == QtCore.QEvent.Enter:
            label = source.objectName()
            self.controller.set_hovered_slider(label)
            self.controller.update_image()
        elif event.type() == QtCore.QEvent.Leave:
            label = source.objectName()
            self.controller.set_hovered_slider(None)
            self.controller.update_image()
        return super().eventFilter(source, event)