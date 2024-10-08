# Previous monolithic version of the application for reference.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Required Libraries
import sys
import os
import json
import numpy as np
import psutil  # For memory checks

# Try to import CuPy; if unavailable, use NumPy as a substitute
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter

    USE_CUPY = True
except ImportError:
    import numpy as cp  # Use NumPy as a substitute
    import scipy.ndimage as cpx_ndimage
    import numpy.fft as cp_fft
    from scipy.ndimage import gaussian_filter as cp_gaussian_filter

    # Define cp.asnumpy to ensure compatibility when using NumPy
    cp.asnumpy = lambda x: x
    cp.asarray = np.asarray  # Ensure cp.asarray is defined
    USE_CUPY = False

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2  # For image I/O
from skimage.feature import peak_local_max  # For peak detection
import threading  # For thread safety
from PyQt5.QtWidgets import QTabWidget, QWidget

# Main Application Class
class FFTImageProcessingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.im = None  # Original grayscale image
        self.im_color = None  # Original color image
        self.im_color_resized = None  # Resized color image for display
        self.im_fft_shifted = None  # Shifted FFT of the image
        self.highpass_image_full = None  # High-pass filtered image at full resolution
        self.final_image_full = None  # Final processed image at full resolution
        self.highpass_image = None  # High-pass image for display
        self.final_image = None  # Final image for display
        self.blurred_image = None  # Blurred image for Phase 1 preview
        self.unsaved_changes = False  # Track unsaved changes
        self.show_original = False  # Toggle for before/after comparison
        self.processing_lock = threading.Lock()  # For thread safety

        self.sliders = {}
        self.slider_values = {}
        self.sliders_widgets = {}
        self.checkboxes = {}

        # Zoom and pan variables
        self.zoom_pan_data = {}
        self.is_updating_axes = False  # Flag to prevent recursive axis updates

        # GUI components
        self.init_ui()

        # Load default parameters if available
        self.load_default_parameters()

        # Define Padding Size
        self.PADDING_SIZE = 16  # Number of pixels to pad on each side

        # Hovered slider
        self.hovered_slider = None

    def init_ui(self):
        """Initialize the user interface components."""
        self.setWindowTitle("FFT-Based Image Processing Application")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.phase1_tab = QWidget()
        self.phase2_tab = QWidget()
        self.tabs.addTab(self.phase1_tab, "1. High-Pass Filter")
        self.tabs.addTab(self.phase2_tab, "2. Detailed Filtering")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Initialize phase tabs
        self.init_phase1_ui()
        self.init_phase2_ui()

        # Matplotlib Figure and Canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add canvas and toolbar to main layout
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # Connect canvas events for hover functionality
        self.motion_cid = self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def init_phase1_ui(self):
        """Initialize the UI components for Phase 1."""
        phase1_layout = QtWidgets.QVBoxLayout(self.phase1_tab)

        # Load Image Button
        load_button = QtWidgets.QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        load_button.setToolTip("Load an image for processing.")
        phase1_layout.addWidget(load_button)

        # High-Pass Filter Radius Slider and SpinBox with Info Icon
        slider_layout = QtWidgets.QHBoxLayout()
        info_button = self.create_info_icon(
            "Adjusts the radius of the Gaussian blur applied to the image. This helps determine the cutoff frequency for the high-pass filter."
        )
        lbl = QtWidgets.QLabel("High-Pass Filter Radius")

        # Slider
        sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sld.setMinimum(0)
        sld.setMaximum(1000)
        sld.setValue(100)  # Assuming initial value 1.0
        sld.valueChanged.connect(
            lambda value: self.on_slider_changed("High-Pass Filter Radius", value)
        )
        sld.setObjectName("High-Pass Filter Radius")
        sld.installEventFilter(self)

        # SpinBox
        val_input = QtWidgets.QDoubleSpinBox()
        val_input.setDecimals(1)
        val_input.setSingleStep(0.1)
        val_input.setRange(0.1, 10.0)
        val_input.setValue(1.0)
        val_input.valueChanged.connect(
            lambda value: self.on_spinbox_changed("High-Pass Filter Radius", value)
        )

        # Store in dictionaries
        self.sliders_widgets["High-Pass Filter Radius"] = sld
        self.slider_values["High-Pass Filter Radius"] = val_input
        self.sliders["High-Pass Filter Radius"] = (sld, 0.1, 10.0)

        # Add to layout
        slider_layout.addWidget(info_button)
        slider_layout.addWidget(lbl)
        slider_layout.addWidget(sld)
        slider_layout.addWidget(val_input)
        phase1_layout.addLayout(slider_layout)

        # Reset Zoom Button
        reset_zoom_button1 = QtWidgets.QPushButton("Reset Zoom")
        reset_zoom_button1.clicked.connect(self.reset_zoom)
        reset_zoom_button1.setToolTip("Reset zoom and pan to default.")
        phase1_layout.addWidget(reset_zoom_button1, alignment=QtCore.Qt.AlignRight)

        # Next Button
        next_button = QtWidgets.QPushButton("Next")
        next_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        phase1_layout.addWidget(next_button, alignment=QtCore.Qt.AlignRight)

    def init_phase2_ui(self):
        """Initialize the UI components for Phase 2."""
        phase2_layout = QtWidgets.QVBoxLayout(self.phase2_tab)

        # Controls Layout
        controls_layout = QtWidgets.QHBoxLayout()
        phase2_layout.addLayout(controls_layout)

        # Sliders Layout
        sliders_layout = QtWidgets.QVBoxLayout()
        sliders_widget = QtWidgets.QWidget()
        sliders_widget.setLayout(sliders_layout)
        sliders_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        controls_layout.addWidget(sliders_widget)

        # Right Layout for Checkboxes and Buttons
        right_layout = QtWidgets.QVBoxLayout()
        controls_layout.addLayout(right_layout)

        # Checkboxes
        checkboxes_layout = QtWidgets.QVBoxLayout()
        checkbox_params = [
            (
                "Enable Frequency Peak Suppression",
                self.update_image,
                "Toggle frequency peak suppression to reduce periodic patterns.",
            ),
            (
                "Enable Attenuation",
                self.update_image,
                "Enable attenuation using gamma correction to smoothly reduce the intensity of detected peaks.",
            ),
            (
                "Enable Anti-Aliasing Filter",
                self.update_image,
                "Apply an anti-aliasing filter to smooth out high-frequency components.",
            ),
        ]

        for label, slot, tooltip in checkbox_params:
            cb_layout = QtWidgets.QHBoxLayout()
            info_button = self.create_info_icon(tooltip)
            cb = QtWidgets.QCheckBox(label)
            cb.stateChanged.connect(slot)
            cb.setToolTip(tooltip)
            cb_layout.addWidget(info_button)
            cb_layout.addWidget(cb)
            checkboxes_layout.addLayout(cb_layout)
            self.checkboxes[label] = cb

        checkboxes_widget = QtWidgets.QWidget()
        checkboxes_widget.setLayout(checkboxes_layout)
        checkboxes_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        right_layout.addWidget(checkboxes_widget)

        # Buttons Layout
        buttons_layout = QtWidgets.QVBoxLayout()
        buttons_widget = QtWidgets.QWidget()
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        right_layout.addWidget(buttons_widget)
        right_layout.addStretch()

        # Sliders and Labels with Groupings and Info Icons
        slider_params = [
            ("Gaussian Blur (%)", 0.0, 5.0, 1.0, self.update_image, "Adjusts the amount of Gaussian blur applied to the FFT magnitude spectrum."),
            # Frequency Peak Suppression Group
            ("---", None, None, None, None, None),  # Separator
            ("Frequency Peak Suppression", None, None, None, None, None),
            ("Peak Min Distance", 1, 50, 10, self.update_image, "Minimum distance between detected peaks in the frequency domain."),
            ("Peak Threshold", 0.0000, 0.0150, 0.0010, self.update_image, "Threshold for peak detection in the frequency domain."),
            ("Mask Radius (%)", 0.1, 10.0, 1.0, self.update_image, "Radius of the circular mask applied to suppress detected peaks."),
            ("Peak Mask Falloff (%)", 0.0, 5.0, 0.0, self.update_image, "Smoothness of the transition at the edge of the peak masks."),
            # Preserve Center Group
            ("---", None, None, None, None, None),  # Separator
            ("Preserve Center", None, None, None, None, None),
            ("Radius (%)", 0.0, 50.0, 10.0, self.update_image, "Radius of the central area to preserve in the frequency domain."),
            ("Aspect Ratio", 0.1, 5.0, 1.0, self.update_image, "Aspect ratio of the preserved central area."),
            ("Orientation", 0, 180, 0, self.update_image, "Rotation angle of the preserved central area."),
            ("Falloff (%)", 0.0, 5.0, 0.0, self.update_image, "Smoothness of the transition at the edge of the preserved area."),
            # Attenuation Group
            ("---", None, None, None, None, None),  # Separator
            ("Attenuation", None, None, None, None, None),
            ("Gamma Correction", 0.1, 1.0, 1.0, self.update_image, "Adjusts the gamma value used for attenuation of frequencies."),
            # Anti-Aliasing Group
            ("---", None, None, None, None, None),  # Separator
            ("Anti-Aliasing Intensity (%)", 0.0, 100.0, 50.0, self.update_image, "Controls the intensity of the anti-aliasing filter applied."),
        ]

        group_labels = set()

        for param in slider_params:
            label = param[0]
            if label == "---":
                # Separator
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.HLine)
                separator.setFrameShadow(QtWidgets.QFrame.Sunken)
                sliders_layout.addWidget(separator)
            elif param[1] is None:
                # Group Label
                group_label = QtWidgets.QLabel(f"<b>{label}</b>")
                sliders_layout.addWidget(group_label)
                group_labels.add(label)
            else:
                min_val, max_val, init_val, slot, tooltip = param[1:]
                slider_layout = QtWidgets.QHBoxLayout()

                # Info Button
                info_button = self.create_info_icon(tooltip)

                lbl = QtWidgets.QLabel(label)

                # Slider
                sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                sld.setMinimum(0)
                sld.setMaximum(1000)
                if label == "Peak Threshold":
                    sld.setMaximum(150000)
                sld.setValue(
                    int((init_val - min_val) / (max_val - min_val) * sld.maximum())
                )
                sld.valueChanged.connect(
                    lambda value, lbl=label: self.on_slider_changed(lbl, value)
                )
                sld.setObjectName(label)
                sld.installEventFilter(self)
                self.sliders_widgets[label] = sld

                # Value Display and Input
                if isinstance(init_val, float):
                    val_input = QtWidgets.QDoubleSpinBox()
                    if label == "Peak Threshold":
                        val_input.setDecimals(4)
                        val_input.setSingleStep(0.0001)
                    else:
                        val_input.setDecimals(3)
                        val_input.setSingleStep(0.1)
                else:
                    val_input = QtWidgets.QSpinBox()
                    val_input.setSingleStep(1)
                val_input.setRange(min_val, max_val)
                val_input.setValue(init_val)
                val_input.valueChanged.connect(
                    lambda value, lbl=label: self.on_spinbox_changed(lbl, value)
                )
                self.slider_values[label] = val_input

                self.sliders[label] = (sld, min_val, max_val)

                # Add to layout
                slider_layout.addWidget(info_button)
                slider_layout.addWidget(lbl)
                slider_layout.addWidget(sld)
                slider_layout.addWidget(val_input)
                sliders_layout.addLayout(slider_layout)

        # Reset Zoom Button
        reset_zoom_button2 = QtWidgets.QPushButton("Reset Zoom")
        reset_zoom_button2.clicked.connect(self.reset_zoom)
        reset_zoom_button2.setToolTip("Reset zoom and pan to default.")
        buttons_layout.addWidget(reset_zoom_button2)

        # Buttons
        # Save Image Button
        save_button = QtWidgets.QPushButton("Save Processed Image")
        save_button.clicked.connect(self.save_image)
        save_button.setToolTip("Save the processed image to disk.")
        buttons_layout.addWidget(save_button)

        # Batch Processing Button
        batch_button = QtWidgets.QPushButton("Batch Process Images")
        batch_button.clicked.connect(self.batch_process)
        batch_button.setToolTip("Process multiple images in a folder.")
        buttons_layout.addWidget(batch_button)

        # Save Parameters Button
        save_params_button = QtWidgets.QPushButton("Save Parameters")
        save_params_button.clicked.connect(self.save_parameters)
        save_params_button.setToolTip("Save current parameters to a file.")
        buttons_layout.addWidget(save_params_button)

        # Load Parameters Button
        load_params_button = QtWidgets.QPushButton("Load Parameters")
        load_params_button.clicked.connect(self.load_parameters)
        load_params_button.setToolTip("Load parameters from a file.")
        buttons_layout.addWidget(load_params_button)

        # Set Defaults Button
        set_defaults_button = QtWidgets.QPushButton("Set as Default Parameters")
        set_defaults_button.clicked.connect(self.save_default_parameters)
        set_defaults_button.setToolTip("Set current parameters as default.")
        buttons_layout.addWidget(set_defaults_button)

        # Back Button to return to Phase 1
        back_button = QtWidgets.QPushButton("Back")
        back_button.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        buttons_layout.addWidget(back_button)

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

    def on_slider_changed(self, label, value):
        """Update the spinbox when the slider changes."""
        sld, min_val, max_val = self.sliders[label]
        val_input = self.slider_values[label]
        if max_val - min_val == 0:
            val = min_val
        else:
            val = value / sld.maximum() * (max_val - min_val) + min_val

        # Check if the spinbox is a QSpinBox or QDoubleSpinBox
        if isinstance(val_input, QtWidgets.QSpinBox):
            val = int(round(val))  # Ensure integer value for QSpinBox
        else:
            val = float(val)  # Ensure float value for QDoubleSpinBox
            val = round(val, 4 if label == "Peak Threshold" else 3)  # Adjust decimal places

        val_input.blockSignals(True)
        val_input.setValue(val)
        val_input.blockSignals(False)
        self.update_image()

    def on_spinbox_changed(self, label, value):
        """Update the slider when the spinbox changes."""
        sld, min_val, max_val = self.sliders[label]
        val_input = self.slider_values[label]

        # Convert value to appropriate type
        if isinstance(val_input, QtWidgets.QSpinBox):
            value = int(value)
        else:
            value = float(value)
            value = round(value, 4 if label == "Peak Threshold" else 3)

        sld.blockSignals(True)
        sld.setValue(int((value - min_val) / (max_val - min_val) * sld.maximum()))
        sld.blockSignals(False)
        self.update_image()

    def on_tab_changed(self, index):
        """Handle tab change events."""
        self.processing_phase = index + 1
        self.update_image()

    def get_slider_value(self, label):
        """Retrieve the current value of a slider."""
        sld, min_val, max_val = self.sliders[label]
        if max_val - min_val == 0:
            return min_val
        value = sld.value() / max(sld.maximum(), 1) * (max_val - min_val) + min_val
        if isinstance(self.slider_values[label], QtWidgets.QSpinBox):
            return int(round(value))
        else:
            return float(round(value, 4 if label == "Peak Threshold" else 3))

    def update_image(self):
        """Update the processed image based on the current settings."""
        with self.processing_lock:
            if self.im_color is None:
                return

            # Determine display area dimensions
            display_width = self.canvas.get_width_height()[0]
            display_height = self.canvas.get_width_height()[1]

            # Process a downscaled version of the image for display
            scaling_factor = min(
                display_width / self.im_color.shape[1],
                display_height / self.im_color.shape[0],
                1.0,
            )
            new_size = (
                int(self.im_color.shape[1] * scaling_factor),
                int(self.im_color.shape[0] * scaling_factor),
            )
            im_color_resized = cv2.resize(
                self.im_color, new_size, interpolation=cv2.INTER_AREA
            )
            im_color_cp = cp.asarray(im_color_resized)

            # Store the resized color image for plotting
            self.im_color_resized = im_color_resized

            if self.processing_phase == 1:
                # Phase 1: Blurred Image Adjustment
                highpass_radius = self.get_slider_value("High-Pass Filter Radius")

                # Apply Gaussian Blur to the color image
                blurred_color = cp.zeros_like(im_color_cp)
                for i in range(3):
                    blurred_color[:, :, i] = cp_gaussian_filter(
                        im_color_cp[:, :, i], sigma=highpass_radius
                    )

                self.blurred_image = cp.asnumpy(blurred_color)
                # No further processing in Phase 1
            else:
                # Phase 2: Detailed Filtering
                # Get all parameter values
                highpass_radius = self.get_slider_value("High-Pass Filter Radius")

                gaussian_sigma_pct = self.get_slider_value("Gaussian Blur (%)")
                peak_min_distance = self.get_slider_value("Peak Min Distance")
                peak_threshold = self.get_slider_value("Peak Threshold")
                exclude_radius_pct = self.get_slider_value("Radius (%)")
                exclude_aspect_ratio = self.get_slider_value("Aspect Ratio")
                exclude_orientation = self.get_slider_value("Orientation")
                exclude_falloff_pct = self.get_slider_value("Falloff (%)")
                mask_radius_pct = self.get_slider_value("Mask Radius (%)")
                peak_mask_falloff_pct = self.get_slider_value("Peak Mask Falloff (%)")
                gamma_correction = self.get_slider_value("Gamma Correction")
                antialiasing_intensity = self.get_slider_value(
                    "Anti-Aliasing Intensity (%)"
                ) / 100.0

                enable_peak_suppression = (
                    self.checkboxes["Enable Frequency Peak Suppression"].isChecked()
                )
                enable_attenuation = (
                    self.checkboxes["Enable Attenuation"].isChecked()
                )
                enable_antialiasing = (
                    self.checkboxes["Enable Anti-Aliasing Filter"].isChecked()
                )

                # Apply High-Pass Filter to the color image
                blurred_color = cp.zeros_like(im_color_cp)
                for i in range(3):
                    blurred_color[:, :, i] = cp_gaussian_filter(
                        im_color_cp[:, :, i], sigma=highpass_radius
                    )

                highpass_color = im_color_cp - blurred_color + 128.0
                highpass_color = cp.clip(highpass_color, 0, 255).astype(cp.float32)

                # Convert high-pass color image to grayscale
                highpass_gray = cp.mean(highpass_color, axis=2)

                # Pad the high-pass grayscale image with neutral gray
                padding_value = 128.0  # Neutral gray value
                padded_highpass_gray = cp.pad(
                    highpass_gray,
                    pad_width=self.PADDING_SIZE,
                    mode="constant",
                    constant_values=padding_value,
                )

                # Perform FFT processing on padded_highpass_gray
                im_fft = cp_fft.fft2(padded_highpass_gray)
                self.im_fft_shifted = cp_fft.fftshift(im_fft)

                # Retrieve image dimensions
                H, W = self.im_fft_shifted.shape

                # Calculate center coordinates
                crow, ccol = H // 2, W // 2

                # Convert relative measurements to absolute pixels
                diag_length = np.sqrt(H**2 + W**2)
                gaussian_sigma = (gaussian_sigma_pct / 100) * diag_length
                exclude_radius = (exclude_radius_pct / 100) * diag_length / 2
                exclude_falloff = (exclude_falloff_pct / 100) * diag_length / 2
                mask_radius = (mask_radius_pct / 100) * diag_length / 2
                peak_mask_falloff = (peak_mask_falloff_pct / 100) * diag_length / 2

                # Preprocess the FFT magnitude spectrum
                magnitude_spectrum = cp.abs(self.im_fft_shifted)
                if gaussian_sigma > 0:
                    blurred_spectrum = cp_gaussian_filter(
                        magnitude_spectrum, sigma=gaussian_sigma
                    )
                else:
                    blurred_spectrum = magnitude_spectrum
                self.blurred_spectrum = blurred_spectrum  # Store for visualization

                # Normalize the magnitude spectrum
                normalized_magnitude = blurred_spectrum / blurred_spectrum.max()

                # Apply gamma correction
                gamma = gamma_correction  # Gamma value from the slider
                adjusted_magnitude = normalized_magnitude ** gamma

                # Create attenuation mask
                attenuation_mask = adjusted_magnitude

                # Optionally apply frequency peak suppression
                if enable_peak_suppression:
                    # Convert to NumPy array for peak detection
                    spectrum_np = cp.asnumpy(blurred_spectrum)

                    # Create a central exclusion mask to prevent detecting the central peak
                    exclusion_mask = self.create_exclusion_mask(
                        shape=self.im_fft_shifted.shape,
                        center=(crow, ccol),
                        radius=exclude_radius,
                        aspect_ratio=exclude_aspect_ratio,
                        orientation=exclude_orientation,
                        falloff=exclude_falloff,
                    )

                    # Convert exclusion mask to NumPy array (True where peaks are allowed)
                    inclusion_mask = cp.asnumpy(exclusion_mask)

                    # Use inclusion_mask directly in peak_local_max
                    coordinates = peak_local_max(
                        spectrum_np,
                        min_distance=int(peak_min_distance),
                        threshold_abs=peak_threshold * spectrum_np.max(),
                        exclude_border=False,
                        labels=inclusion_mask.astype(np.uint8),
                    )

                    # Create masks for each detected peak
                    peak_masks = []
                    for coord in coordinates:
                        mask = self.create_circular_mask(
                            self.im_fft_shifted.shape,
                            center=coord,
                            radius=mask_radius,
                            falloff=peak_mask_falloff,
                        )
                        peak_masks.append(mask)

                    # Combine all peak masks
                    if peak_masks:
                        combined_peak_mask = cp.ones(
                            self.im_fft_shifted.shape, dtype=cp.float32
                        )
                        for mask in peak_masks:
                            combined_peak_mask *= mask
                    else:
                        combined_peak_mask = cp.ones(
                            self.im_fft_shifted.shape, dtype=cp.float32
                        )
                else:
                    combined_peak_mask = cp.ones(
                        self.im_fft_shifted.shape, dtype=cp.float32
                    )

                # Combined mask is now only the peak mask
                combined_mask = combined_peak_mask

                # Step 1: Apply the frequency peak suppression mask
                im_fft_filtered = self.im_fft_shifted * combined_mask

                if enable_peak_suppression and enable_attenuation:
                    # Step 2: Create suppression areas mask (inverted mask)
                    suppression_areas_mask = 1 - combined_mask
                    # Step 3: Apply gamma correction to frequencies in suppression areas
                    im_fft_suppression = (
                        self.im_fft_shifted * suppression_areas_mask * attenuation_mask
                    )
                    # Step 4: Add the attenuated frequencies back to the filtered FFT image
                    im_fft_filtered = im_fft_filtered + im_fft_suppression

                # Apply Anti-Aliasing Filter if enabled
                if enable_antialiasing:
                    antialiasing_mask = self.create_antialiasing_mask(
                        self.im_fft_shifted.shape, antialiasing_intensity
                    )
                    im_fft_filtered *= antialiasing_mask

                # Store the filtered FFT for plotting
                self.im_fft_filtered = im_fft_filtered

                # Inverse FFT to reconstruct the high-frequency grayscale image
                im_ifft = cp_fft.ifftshift(im_fft_filtered)
                im_new = cp.abs(cp_fft.ifft2(im_ifft))

                # Crop the image to remove the padding
                pad = self.PADDING_SIZE
                im_new_cropped = im_new[pad:-pad, pad:-pad]

                # Ensure the cropped image has the original dimensions
                # Adjust size if necessary due to rounding during padding
                target_shape = highpass_gray.shape
                im_new_cropped = im_new_cropped[: target_shape[0], : target_shape[1]]

                # Expand dimensions to match color channels
                im_new_expanded = cp.repeat(im_new_cropped[:, :, cp.newaxis], 3, axis=2)

                # Crop the low-pass color image to match the processed image
                lowpass_color_cropped = blurred_color[
                    : im_new_cropped.shape[0], : im_new_cropped.shape[1], :
                ]

                # Blend high-frequency details with low-frequency color image
                final_image_cp = cp.clip(
                    lowpass_color_cropped + im_new_expanded - 128.0, 0, 255
                ).astype(cp.float32)

                # Convert to NumPy array for plotting
                self.final_image = cp.asnumpy(final_image_cp)
                self.highpass_image = cp.asnumpy(highpass_color)

                # Set unsaved changes flag
                self.unsaved_changes = True

            # Update the plot
            self.plot_images()

    def create_circular_mask(self, shape, center, radius, falloff=0):
        """Create a circular mask centered at 'center' with given 'radius' and 'falloff'."""
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        dist_from_center = cp.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = cp.ones(shape, dtype=cp.float32)
        mask[dist_from_center <= radius] = 0  # Suppress frequencies within the radius

        if falloff > 0:
            mask = cp_gaussian_filter(mask, sigma=falloff)

        return mask

    def create_antialiasing_mask(self, shape, intensity):
        """Create an anti-aliasing mask that suppresses high frequencies."""
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        radius = cp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_radius = cp.sqrt((cx) ** 2 + (cy) ** 2)
        # Use a smooth transition function
        mask = 1 - intensity * (1 - cp.cos(cp.pi * radius / max_radius)) / 2
        return mask.astype(cp.float32)  # Ensure float32 precision

    def create_exclusion_mask(
        self, shape, center, radius, aspect_ratio=1.0, orientation=0, falloff=0
    ):
        """Create an elliptical exclusion mask centered at 'center' with given 'radius', 'aspect_ratio', 'orientation', and 'falloff'."""
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        cy, cx = center

        # Adjust coordinates
        Xc = X - cx
        Yc = Y - cy

        # Apply rotation
        theta = cp.deg2rad(orientation)
        cos_theta = cp.cos(theta)
        sin_theta = cp.sin(theta)
        Xr = Xc * cos_theta + Yc * sin_theta
        Yr = -Xc * sin_theta + Yc * cos_theta

        # Apply aspect ratio
        distance = (Xr / radius) ** 2 + (Yr / (radius * aspect_ratio)) ** 2

        # Create the mask
        mask = cp.ones(shape, dtype=cp.bool_)
        mask[distance <= 1] = False  # Exclude frequencies within the ellipse

        if falloff > 0:
            mask = cp_gaussian_filter(mask.astype(cp.float32), sigma=falloff)
            mask = mask > 0.5  # Convert back to boolean

        return mask

    def plot_images(self):
        """Plot the images based on the current processing phase."""
        # Save current xlim and ylim before clearing
        if hasattr(self, "axes"):
            current_lims = [(ax.get_xlim(), ax.get_ylim()) for ax in self.axes]
        else:
            current_lims = None

        self.figure.clear()
        if self.processing_phase == 1:
            # Phase 1: Display Original Image, Blurred Image, and High-Pass Image
            ax1 = self.figure.add_subplot(1, 3, 1)
            ax2 = self.figure.add_subplot(1, 3, 2)
            ax3 = self.figure.add_subplot(1, 3, 3)
            self.axes = [ax1, ax2, ax3]

            # Original Image (Resized)
            ax1.imshow(self.im_color_resized.astype(np.uint8))
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Blurred Image
            if self.blurred_image is not None:
                ax2.imshow(self.blurred_image.astype(np.uint8))
                ax2.set_title("Blurred Image")
                ax2.axis("off")

            # High-Pass Image (Difference)
            if self.blurred_image is not None:
                highpass_image = self.im_color_resized - self.blurred_image + 128.0
                highpass_image = np.clip(highpass_image, 0, 255)
                ax3.imshow(highpass_image.astype(np.uint8))
                ax3.set_title("High-Pass Image")
                ax3.axis("off")
            else:
                ax3.axis("off")

        else:
            # Phase 2: Display Original Image, Filtered FFT Spectrum, and Processed Image
            ax1 = self.figure.add_subplot(1, 3, 1)
            ax2 = self.figure.add_subplot(1, 3, 2)
            ax3 = self.figure.add_subplot(1, 3, 3)
            self.axes = [ax1, ax2, ax3]

            # Original Image (Resized)
            ax1.imshow(self.im_color_resized.astype(np.uint8))
            ax1.set_title("Original Image")
            ax1.axis("off")

            # FFT Spectrum
            if self.hovered_slider in [
                "Gaussian Blur (%)",
                "Peak Min Distance",
                "Peak Threshold",
                "Mask Radius (%)",
                "Peak Mask Falloff (%)",
                "Radius (%)",
                "Aspect Ratio",
                "Orientation",
                "Falloff (%)",
            ]:
                magnitude_spectrum = cp.log(cp.abs(self.im_fft_filtered) + 1)
                ax2.set_title("Filtered FFT Spectrum")

                # Overlay exclusion shape visualization
                H, W = magnitude_spectrum.shape
                diag_length = np.sqrt(H**2 + W**2)
                exclude_radius_pct = self.get_slider_value("Radius (%)")
                exclude_aspect_ratio = self.get_slider_value("Aspect Ratio")
                exclude_orientation = self.get_slider_value("Orientation")

                exclude_radius = (exclude_radius_pct / 100) * diag_length / 2

                # Generate Ellipse Path for visualization
                from matplotlib.patches import Ellipse

                ellipse = Ellipse(
                    xy=(W / 2, H / 2),
                    width=2 * exclude_radius * 1 / exclude_aspect_ratio,
                    height=2 * exclude_radius,
                    angle=exclude_orientation,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=1,
                )
                ax2.add_patch(ellipse)
            else:
                magnitude_spectrum = cp.log(cp.abs(self.im_fft_filtered) + 1)
                ax2.set_title("Filtered FFT Spectrum")

            ax2.imshow(cp.asnumpy(magnitude_spectrum), cmap="gray", origin="upper")
            ax2.axis("off")

            # Processed Image or Original Image
            if self.show_original:
                im_display = self.im_color_resized / 255.0
                ax3.set_title("Original Image")
            else:
                if self.final_image is not None:
                    im_display = self.final_image / 255.0
                    ax3.set_title("Processed Image")
                else:
                    im_display = np.zeros_like(self.im_color_resized) / 255.0
                    ax3.set_title("Processed Image (Unavailable)")

            ax3.imshow(im_display.astype(np.float32))
            ax3.axis("off")

        # Synchronize zoom and pan
        self.synchronize_axes()

        # Restore zoom and pan if available
        if current_lims:
            for lim, ax in zip(current_lims, self.axes):
                ax.set_xlim(lim[0])
                ax.set_ylim(lim[1])

        self.canvas.draw()

    def synchronize_axes(self):
        """Synchronize zoom and pan across all axes in the current tab."""
        # Disconnect previous callbacks if any
        if hasattr(self, '_callback_cids'):
            for cid in self._callback_cids:
                cid['axis'].callbacks.disconnect(cid['xlim_cid'])
                cid['axis'].callbacks.disconnect(cid['ylim_cid'])
        else:
            self._callback_cids = []

        # Clear the list for new callbacks
        self._callback_cids = []

        # Connect callbacks and store cids
        for ax in self.figure.axes:
            xlim_cid = ax.callbacks.connect("xlim_changed", self.on_xlim_changed)
            ylim_cid = ax.callbacks.connect("ylim_changed", self.on_ylim_changed)
            self._callback_cids.append({'axis': ax, 'xlim_cid': xlim_cid, 'ylim_cid': ylim_cid})

    def on_xlim_changed(self, ax):
        """Handle x-axis limit changes."""
        if sys.is_finalizing():
            return
        if self.is_updating_axes:
            return
        self.is_updating_axes = True
        xlim = ax.get_xlim()
        for other_ax in self.figure.axes:
            if other_ax is not ax:
                other_ax.set_xlim(xlim)
        self.is_updating_axes = False
        self.canvas.draw_idle()

    def on_ylim_changed(self, ax):
        """Handle y-axis limit changes."""
        if sys.is_finalizing():
            return
        if self.is_updating_axes:
            return
        self.is_updating_axes = True
        ylim = ax.get_ylim()
        for other_ax in self.figure.axes:
            if other_ax is not ax:
                other_ax.set_ylim(ylim)
        self.is_updating_axes = False
        self.canvas.draw_idle()

    def reset_zoom(self):
        """Reset zoom and pan to default for all axes."""
        for ax in self.figure.axes:
            ax.autoscale()
        self.canvas.draw_idle()

    def eventFilter(self, source, event):
        """Event filter to detect when the user hovers over specific sliders."""
        if event.type() == QtCore.QEvent.Enter:
            if source.objectName() in [
                "Radius (%)",
                "Gaussian Blur (%)",
                "High-Pass Filter Radius",
                "Exclude Aspect Ratio",
                "Orientation",
                "Falloff (%)",
            ]:
                self.hovered_slider = source.objectName()
                self.update_image()
            else:
                self.hovered_slider = None
        elif event.type() == QtCore.QEvent.Leave:
            self.hovered_slider = None
            self.update_image()
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        """Handle the window close event."""
        if self.unsaved_changes and self.final_image is not None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the processed image before exiting?",
                QtWidgets.QMessageBox.Save
                | QtWidgets.QMessageBox.Discard
                | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Save,
            )
            if reply == QtWidgets.QMessageBox.Save:
                self.save_image()
                event.accept()
            elif reply == QtWidgets.QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
                return  # Early return if the event is ignored
        else:
            event.accept()

        # Disconnect callbacks to prevent AttributeError during shutdown
        if hasattr(self, '_callback_cids'):
            for cid in self._callback_cids:
                cid['axis'].callbacks.disconnect(cid['xlim_cid'])
                cid['axis'].callbacks.disconnect(cid['ylim_cid'])

        # Disconnect the canvas event callback
        if hasattr(self, 'motion_cid'):
            self.canvas.mpl_disconnect(self.motion_cid)

        # Clean up CuPy memory
        if USE_CUPY:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        # Release resources
        self.figure.clear()
        plt.close(self.figure)

    # Hover Method for Original Image
    def on_motion(self, event):
        """Handle mouse motion events to show original image on hover."""
        if hasattr(self, "axes") and event.inaxes == self.axes[-1]:
            if not self.show_original:
                self.show_original = True
                self.plot_images()
        else:
            if self.show_original:
                self.show_original = False
                self.plot_images()

    # Image Processing Methods
    def load_image(self):
        """Load an image using a file dialog."""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            options=options,
        )
        if file_path:
            # Load the image in color mode (BGR)
            im_color = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if im_color is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load the image.")
                return
            # Convert BGR to RGB
            im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
            # Store the color image as a NumPy array
            self.im_color = im_color.astype(np.float32)
            # Create a grayscale version for initial display
            self.im = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Reset FFT variables
            self.im_fft_shifted = None
            self.final_image_full = None
            self.highpass_image_full = None

            # Set processing phase to 1
            self.processing_phase = 1
            self.tabs.setCurrentIndex(0)

            # Update the image
            self.update_image()

    def load_default_parameters(self):
        """Load default parameters if available."""
        try:
            with open("default_parameters.json", "r") as f:
                params = json.load(f)
            for label, value in params.items():
                if label in self.sliders:
                    val_input = self.slider_values[label]
                    val_input.blockSignals(True)
                    # Ensure value is of correct type
                    if isinstance(val_input, QtWidgets.QSpinBox):
                        value = int(round(value))
                    else:
                        value = float(value)
                        value = round(value, 4 if label == "Peak Threshold" else 3)
                    val_input.setValue(value)
                    val_input.blockSignals(False)
                    self.on_spinbox_changed(label, value)
                elif label in self.checkboxes:
                    cb = self.checkboxes[label]
                    cb.blockSignals(True)
                    cb.setChecked(value)
                    cb.blockSignals(False)
            self.update_image()
        except FileNotFoundError:
            pass  # Use initial default values

    def save_default_parameters(self):
        """Save current parameters as the default settings."""
        params = {label: self.slider_values[label].value() for label in self.sliders}
        params.update(
            {label: self.checkboxes[label].isChecked() for label in self.checkboxes}
        )
        with open("default_parameters.json", "w") as f:
            json.dump(params, f)
        QtWidgets.QMessageBox.information(
            self,
            "Default Parameters Set",
            "Current parameters have been set as default.",
        )

    def save_parameters(self):
        """Save current parameters to a JSON file."""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Parameters",
            "",
            "JSON Files (*.json)",
            options=options,
        )
        if file_path:
            params = {label: self.slider_values[label].value() for label in self.sliders}
            params.update(
                {label: self.checkboxes[label].isChecked() for label in self.checkboxes}
            )
            with open(file_path, "w") as f:
                json.dump(params, f)
            QtWidgets.QMessageBox.information(
                self, "Parameters Saved", f"Parameters saved to {file_path}"
            )

    def load_parameters(self):
        """Load parameters from a JSON file."""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Parameters",
            "",
            "JSON Files (*.json)",
            options=options,
        )
        if file_path:
            with open(file_path, "r") as f:
                params = json.load(f)
            for label, value in params.items():
                if label in self.sliders:
                    val_input = self.slider_values[label]
                    val_input.blockSignals(True)
                    # Ensure value is of correct type
                    if isinstance(val_input, QtWidgets.QSpinBox):
                        value = int(round(value))
                    else:
                        value = float(value)
                        value = round(value, 4 if label == "Peak Threshold" else 3)
                    val_input.setValue(value)
                    val_input.blockSignals(False)
                    self.on_spinbox_changed(label, value)
                elif label in self.checkboxes:
                    cb = self.checkboxes[label]
                    cb.blockSignals(True)
                    cb.setChecked(value)
                    cb.blockSignals(False)
            self.update_image()
            QtWidgets.QMessageBox.information(
                self, "Parameters Loaded", f"Parameters loaded from {file_path}"
            )

    def save_image(self):
        """Save the processed image using a file dialog."""
        if self.im_color is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "No image loaded.")
            return

        # Process the full-resolution image before saving
        self.process_full_resolution()

        if self.final_image_full is not None:
            options = QtWidgets.QFileDialog.Options()
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Processed Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif *.tiff)",
                options=options,
            )
            if save_path:
                # Convert to NumPy array and save using OpenCV
                im_to_save = self.final_image_full.astype(np.uint8)
                im_to_save = cv2.cvtColor(im_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, im_to_save)
                self.unsaved_changes = False  # Reset unsaved changes flag
                QtWidgets.QMessageBox.information(
                    self, "Image Saved", f"Image saved to {save_path}"
                )
        else:
            QtWidgets.QMessageBox.warning(self, "No Image", "No image to save.")

    def batch_process(self):
        """Batch process images in a selected directory."""
        input_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Input Directory"
        )
        if input_dir:
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Output Directory"
            )
            if not output_dir:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "No output directory selected."
                )
                return

            image_files = [
                f
                for f in os.listdir(input_dir)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                )
            ]

            if not image_files:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Images",
                    "No supported image files found in the input directory.",
                )
                return

            progress = QtWidgets.QProgressDialog(
                "Processing images...", "Cancel", 0, len(image_files), self
            )
            progress.setWindowModality(QtCore.Qt.WindowModal)

            for i, file_name in enumerate(image_files):
                if progress.wasCanceled():
                    break

                file_path = os.path.join(input_dir, file_name)
                # Load the image
                im_color = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if im_color is None:
                    continue  # Skip files that cannot be loaded
                # Convert BGR to RGB
                im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
                self.im_color = im_color.astype(np.float32)

                # Reset FFT variables
                self.im_fft_shifted = None
                self.final_image_full = None
                self.highpass_image_full = None

                # Update the image and process full resolution
                self.process_full_resolution()

                # Save the processed image
                if self.final_image_full is not None:
                    im_to_save = self.final_image_full.astype(np.uint8)
                    save_path = os.path.join(output_dir, file_name)
                    im_to_save = cv2.cvtColor(im_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, im_to_save)
                else:
                    continue  # Skip if processing failed

                progress.setValue(i + 1)

            progress.setValue(len(image_files))
            QtWidgets.QMessageBox.information(
                self, "Batch Processing", "Batch processing completed."
            )

    def process_full_resolution(self):
        """Process the image at full resolution for saving."""
        with self.processing_lock:
            if self.im_color is None:
                return

            im_color_cp = cp.asarray(self.im_color)

            # Get all parameter values
            highpass_radius = self.get_slider_value("High-Pass Filter Radius")

            gaussian_sigma_pct = self.get_slider_value("Gaussian Blur (%)")
            peak_min_distance = self.get_slider_value("Peak Min Distance")
            peak_threshold = self.get_slider_value("Peak Threshold")
            exclude_radius_pct = self.get_slider_value("Radius (%)")
            exclude_aspect_ratio = self.get_slider_value("Aspect Ratio")
            exclude_orientation = self.get_slider_value("Orientation")
            exclude_falloff_pct = self.get_slider_value("Falloff (%)")
            mask_radius_pct = self.get_slider_value("Mask Radius (%)")
            peak_mask_falloff_pct = self.get_slider_value("Peak Mask Falloff (%)")
            gamma_correction = self.get_slider_value("Gamma Correction")
            antialiasing_intensity = (
                self.get_slider_value("Anti-Aliasing Intensity (%)") / 100.0
            )

            enable_peak_suppression = (
                self.checkboxes["Enable Frequency Peak Suppression"].isChecked()
            )
            enable_attenuation = (
                self.checkboxes["Enable Attenuation"].isChecked()
            )
            enable_antialiasing = (
                self.checkboxes["Enable Anti-Aliasing Filter"].isChecked()
            )

            # Apply High-Pass Filter to the color image
            blurred_color = cp.zeros_like(im_color_cp)
            for i in range(3):
                blurred_color[:, :, i] = cp_gaussian_filter(
                    im_color_cp[:, :, i], sigma=highpass_radius
                )

            highpass_color = im_color_cp - blurred_color + 128.0
            highpass_color = cp.clip(highpass_color, 0, 255).astype(cp.float32)

            # Convert high-pass color image to grayscale
            highpass_gray = cp.mean(highpass_color, axis=2)

            # Pad the high-pass grayscale image with neutral gray
            padding_value = 128.0  # Neutral gray value
            padded_highpass_gray = cp.pad(
                highpass_gray,
                pad_width=self.PADDING_SIZE,
                mode="constant",
                constant_values=padding_value,
            )

            # Perform FFT processing on padded_highpass_gray
            im_fft = cp_fft.fft2(padded_highpass_gray)
            self.im_fft_shifted = cp_fft.fftshift(im_fft)

            # Retrieve image dimensions
            H, W = self.im_fft_shifted.shape

            # Calculate center coordinates
            crow, ccol = H // 2, W // 2

            # Convert relative measurements to absolute pixels
            diag_length = np.sqrt(H**2 + W**2)
            gaussian_sigma = (gaussian_sigma_pct / 100) * diag_length
            exclude_radius = (exclude_radius_pct / 100) * diag_length / 2
            exclude_falloff = (exclude_falloff_pct / 100) * diag_length / 2
            mask_radius = (mask_radius_pct / 100) * diag_length / 2
            peak_mask_falloff = (peak_mask_falloff_pct / 100) * diag_length / 2

            # Preprocess the FFT magnitude spectrum
            magnitude_spectrum = cp.abs(self.im_fft_shifted)
            if gaussian_sigma > 0:
                blurred_spectrum = cp_gaussian_filter(
                    magnitude_spectrum, sigma=gaussian_sigma
                )
            else:
                blurred_spectrum = magnitude_spectrum
            self.blurred_spectrum = blurred_spectrum  # Store for visualization

            # Normalize the magnitude spectrum
            normalized_magnitude = blurred_spectrum / blurred_spectrum.max()

            # Apply gamma correction
            gamma = gamma_correction  # Gamma value from the slider
            adjusted_magnitude = normalized_magnitude ** gamma

            # Create attenuation mask
            attenuation_mask = adjusted_magnitude

            # Optionally apply frequency peak suppression
            if enable_peak_suppression:
                # Convert to NumPy array for peak detection
                spectrum_np = cp.asnumpy(blurred_spectrum)

                # Create a central exclusion mask to prevent detecting the central peak
                exclusion_mask = self.create_exclusion_mask(
                    shape=self.im_fft_shifted.shape,
                    center=(crow, ccol),
                    radius=exclude_radius,
                    aspect_ratio=exclude_aspect_ratio,
                    orientation=exclude_orientation,
                    falloff=exclude_falloff,
                )

                # Convert exclusion mask to NumPy array
                inclusion_mask = cp.asnumpy(exclusion_mask)

                # Use inclusion_mask directly in peak_local_max
                coordinates = peak_local_max(
                    spectrum_np,
                    min_distance=int(peak_min_distance),
                    threshold_abs=peak_threshold * spectrum_np.max(),
                    exclude_border=False,
                    labels=inclusion_mask.astype(np.uint8),
                )

                # Create masks for each detected peak
                peak_masks = []
                for coord in coordinates:
                    mask = self.create_circular_mask(
                        self.im_fft_shifted.shape,
                        center=coord,
                        radius=mask_radius,
                        falloff=peak_mask_falloff,
                    )
                    peak_masks.append(mask)

                # Combine all peak masks
                if peak_masks:
                    combined_peak_mask = cp.ones(
                        self.im_fft_shifted.shape, dtype=cp.float32
                    )
                    for mask in peak_masks:
                        combined_peak_mask *= mask
                else:
                    combined_peak_mask = cp.ones(
                        self.im_fft_shifted.shape, dtype=cp.float32
                    )
            else:
                combined_peak_mask = cp.ones(self.im_fft_shifted.shape, dtype=cp.float32)

            # Combined mask is now only the peak mask
            combined_mask = combined_peak_mask

            # Step 1: Apply the frequency peak suppression mask
            im_fft_filtered = self.im_fft_shifted * combined_mask

            if enable_peak_suppression and enable_attenuation:
                # Step 2: Create suppression areas mask (inverted mask)
                suppression_areas_mask = 1 - combined_mask
                # Step 3: Apply gamma correction to frequencies in suppression areas
                im_fft_suppression = (
                    self.im_fft_shifted * suppression_areas_mask * attenuation_mask
                )
                # Step 4: Add the attenuated frequencies back to the filtered FFT image
                im_fft_filtered = im_fft_filtered + im_fft_suppression

            # Apply Anti-Aliasing Filter if enabled
            if enable_antialiasing:
                antialiasing_mask = self.create_antialiasing_mask(
                    self.im_fft_shifted.shape, antialiasing_intensity
                )
                im_fft_filtered *= antialiasing_mask

            # Inverse FFT to reconstruct the high-frequency grayscale image
            im_ifft = cp_fft.ifftshift(im_fft_filtered)
            im_new = cp.abs(cp_fft.ifft2(im_ifft))

            # Crop the image to remove the padding
            pad = self.PADDING_SIZE
            im_new_cropped = im_new[pad:-pad, pad:-pad]

            # Ensure the cropped image has the original dimensions
            # Adjust size if necessary due to rounding during padding
            target_shape = highpass_gray.shape
            im_new_cropped = im_new_cropped[: target_shape[0], : target_shape[1]]

            # Expand dimensions to match color channels
            im_new_expanded = cp.repeat(im_new_cropped[:, :, cp.newaxis], 3, axis=2)

            # Crop the low-pass color image to match the processed image
            blurred_color = blurred_color[
                : im_new_cropped.shape[0], : im_new_cropped.shape[1], :
            ]

            # Blend high-frequency details with low-frequency color image
            final_image_cp = cp.clip(
                blurred_color + im_new_expanded - 128.0, 0, 255
            ).astype(cp.float32)

            # Convert to NumPy array for saving
            self.final_image_full = cp.asnumpy(final_image_cp)


# Run the Application
def main():
    """Main function to run the application."""
    app = QtWidgets.QApplication(sys.argv)
    ex = FFTImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
