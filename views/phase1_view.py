from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import cupy as cp

class Phase1View(QWidget):
    """
    View for Phase 1: High-Pass Filtering.
    
    Provides controls for adjusting the high-pass filter radius.
    """
    next_phase = QtCore.pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        # Initialize dictionaries for sliders and widgets
        self.sliders_widgets = {}
        self.slider_values = {}
        self.sliders = {}

        self.init_phase1_ui()

    def init_phase1_ui(self):
        """Initialize the UI components for Phase 1."""
        layout = QVBoxLayout()

        # High-Pass Filter Radius Slider and SpinBox
        slider_layout = QtWidgets.QHBoxLayout()
        info_button = self.create_info_icon(
            "Adjusts the radius of the Gaussian blur applied to the image. This helps determine the cutoff frequency for the high-pass filter."
        )
        lbl = QtWidgets.QLabel("High-Pass Filter Radius")

        # Slider
        sld = QSlider(QtCore.Qt.Horizontal)
        sld.setMinimum(0)
        sld.setMaximum(1000)
        sld.setValue(100)  # Default value corresponding to radius 1.0
        sld.valueChanged.connect(self.on_slider_changed)
        sld.setObjectName("High-Pass Filter Radius")
        sld.installEventFilter(self)

        # SpinBox
        val_input = QtWidgets.QDoubleSpinBox()
        val_input.setDecimals(1)
        val_input.setSingleStep(0.1)
        val_input.setRange(0.1, 10.0)
        val_input.setValue(1.0)
        val_input.valueChanged.connect(self.on_spinbox_changed)

        # Store widgets
        self.sliders_widgets["High-Pass Filter Radius"] = sld
        self.slider_values["High-Pass Filter Radius"] = val_input
        self.sliders["High-Pass Filter Radius"] = (sld, 0.1, 10.0)

        # Add to layout
        slider_layout.addWidget(info_button)
        slider_layout.addWidget(lbl)
        slider_layout.addWidget(sld)
        slider_layout.addWidget(val_input)
        layout.addLayout(slider_layout)

        # Add QLabel for displaying images
        self.original_image_label = QLabel(self)
        self.blurred_image_label = QLabel(self)
        self.highpass_image_label = QLabel(self)

        # Add image labels to the layout
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.blurred_image_label)
        image_layout.addWidget(self.highpass_image_label)
        layout.addLayout(image_layout)

        # **Modify the connection of the "Load Image" button**
        load_button = QtWidgets.QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)  # Connect to the new load_image method
        slider_layout.addWidget(load_button)

        # 'Next' Button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.on_next_clicked)
        layout.addWidget(self.next_button)

        self.setLayout(layout)

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
        min_val, max_val = self.sliders["High-Pass Filter Radius"][1:]
        val = value / 1000 * (max_val - min_val) + min_val
        self.slider_values["High-Pass Filter Radius"].blockSignals(True)
        self.slider_values["High-Pass Filter Radius"].setValue(val)
        self.slider_values["High-Pass Filter Radius"].blockSignals(False)
        self.controller.update_parameters("High-Pass Filter Radius", val)
        self.controller.update_image()

    def on_spinbox_changed(self, value):
        """Handle spinbox value changes."""
        min_val, max_val = self.sliders["High-Pass Filter Radius"][1:]
        sld = self.sliders_widgets["High-Pass Filter Radius"]
        sld.blockSignals(True)
        sld.setValue(int((value - min_val) / (max_val - min_val) * 1000))
        sld.blockSignals(False)
        self.controller.update_parameters("High-Pass Filter Radius", value)
        self.controller.update_image()

    def eventFilter(self, source, event):
        """Event filter to detect when hovering over the slider."""
        if event.type() == QtCore.QEvent.Enter or event.type() == QtCore.QEvent.Leave:
            if source.objectName() == "High-Pass Filter Radius":
                self.controller.set_hovered_slider("High-Pass Filter Radius" if event.type() == QtCore.QEvent.Enter else None)
                # Only update the image if one is loaded
                if self.controller.image is not None:
                    self.controller.update_image()
        return super().eventFilter(source, event)

    def display_image(self, original_image, blurred_image, highpass_image):
        """Display the processed images in the UI."""
        # Convert CuPy arrays to NumPy arrays
        original_np = cp.asnumpy(original_image)
        blurred_np = cp.asnumpy(blurred_image)
        highpass_np = cp.asnumpy(highpass_image)

        # Convert images to QImage format
        def to_qimage(image_np):
            if image_np.size == 0:
                raise ValueError("Empty image array. Cannot convert to QImage.")
            
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            return QtGui.QImage(image_np.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Update QLabel with images
        self.original_image_label.setPixmap(QtGui.QPixmap.fromImage(to_qimage(original_np)))
        self.blurred_image_label.setPixmap(QtGui.QPixmap.fromImage(to_qimage(blurred_np)))
        self.highpass_image_label.setPixmap(QtGui.QPixmap.fromImage(to_qimage(highpass_np)))

    def load_image(self):
        """Handle loading of an image via the Load Image button."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            options=options,
        )
        if file_path:
            # Pass the selected file path to the controller's load_image method
            self.controller.load_image(file_path)
        else:
            print("No file selected to load.")

    def on_next_clicked(self):
        """Handle the 'Next' button click to transition to Phase 2."""
        self.next_phase.emit()