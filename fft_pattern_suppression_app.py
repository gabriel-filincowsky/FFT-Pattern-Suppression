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

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2  # For image I/O
from skimage.feature import peak_local_max  # For peak detection

# Main Application Class
class FFTImageProcessingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.im = None  # Original image
        self.im_fft_shifted = None  # Shifted FFT of the image
        self.im_new = None  # Processed image
        self.im_fft_filtered = None  # Filtered FFT
        self.blurred_spectrum = None  # Blurred spectrum for visualization
        self.hovered_slider = None  # For slider hover events
        self.unsaved_changes = False  # Track unsaved changes
        self.show_original = False  # Toggle for before/after

        # GUI components
        self.init_ui()

        # Load default parameters if available
        self.load_default_parameters()

    def init_ui(self):
        """Initialize the user interface components."""
        self.setWindowTitle("FFT-Based Image Processing Application")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        preview_layout = QtWidgets.QVBoxLayout()
        controls_layout = QtWidgets.QHBoxLayout()
        sliders_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()  # For checkboxes and buttons
        checkboxes_layout = QtWidgets.QVBoxLayout()
        buttons_layout = QtWidgets.QVBoxLayout()

        # Add layouts to main layout with stretch factors
        main_layout.addLayout(preview_layout, stretch=3)
        main_layout.addLayout(controls_layout, stretch=1)

        # Previews
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Add Matplotlib Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        preview_layout.addWidget(self.toolbar)

        # Create grid layout for images and reset zoom button
        image_layout = QtWidgets.QGridLayout()
        preview_layout.addLayout(image_layout)

        # Add canvas to the grid layout, spanning 1 row and 3 columns
        image_layout.addWidget(self.canvas, 0, 0, 1, 3)

        # Reset Zoom Button
        reset_zoom_button = QtWidgets.QPushButton("Reset Zoom")
        reset_zoom_button.clicked.connect(self.reset_zoom)
        reset_zoom_button.setFixedSize(80, 25)  # Smaller size

        # Add reset zoom button under the third image
        image_layout.addWidget(reset_zoom_button, 1, 2, alignment=QtCore.Qt.AlignCenter)

        # Controls
        sliders_layout_widget = QtWidgets.QWidget()
        sliders_layout_widget.setLayout(sliders_layout)
        sliders_layout_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        checkboxes_layout_widget = QtWidgets.QWidget()
        checkboxes_layout_widget.setLayout(checkboxes_layout)
        checkboxes_layout_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)
        buttons_layout_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        # Add checkboxes and buttons to right_layout
        right_layout.addWidget(checkboxes_layout_widget)
        right_layout.addWidget(buttons_layout_widget)
        right_layout.addStretch()  # Add stretch to push the buttons up

        controls_layout.addWidget(sliders_layout_widget)
        controls_layout.addLayout(right_layout)

        # Sliders and labels
        self.sliders = {}
        self.slider_values = {}  # For spinboxes
        slider_params = [
            ("Gaussian Blur Sigma (%)", 0.0, 5.0, 1.0, self.update_image),  # Relative measurement
            ("Peak Min Distance", 1, 50, 10, self.update_image),
            ("Peak Threshold", 0.0000, 0.0150, 0.0010, self.update_image),  # Adjusted range and precision
            ("Exclude Radius (%)", 0.0, 50.0, 10.0, self.update_image),  # Relative measurement
            ("Mask Radius (%)", 0.1, 10.0, 1.0, self.update_image),  # Relative measurement
            ("Peak Mask Falloff (%)", 0.0, 5.0, 0.0, self.update_image),  # Relative measurement
            ("Gamma Correction", 0.1, 1.0, 1.0, self.update_image),  # Gamma correction slider
            ("Central Mask Radius (%)", 0.1, 50.0, 5.0, self.update_image),  # Relative measurement
            ("Central Mask Aspect Ratio", 0.1, 5.0, 1.0, self.update_image),
            ("Central Mask Rotation", 0, 180, 0, self.update_image),
            ("Central Mask Falloff (%)", 0.0, 5.0, 0.0, self.update_image),  # Relative measurement
            ("Anti-Aliasing Intensity (%)", 0.0, 100.0, 50.0, self.update_image),  # For anti-aliasing filter
        ]

        self.sliders_widgets = {}  # Store slider widgets for event filtering
        for label, min_val, max_val, init_val, slot in slider_params:
            slider_layout = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setToolTip(f"Adjust {label.lower()}.")

            # Slider
            sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld.setMinimum(0)
            if label == "Peak Threshold":
                sld.setMaximum(150000)  # For finer control
            else:
                sld.setMaximum(1000)
            sld.setValue(int((init_val - min_val) / (max_val - min_val) * sld.maximum()))
            sld.valueChanged.connect(lambda value, lbl=label: self.on_slider_changed(lbl, value))
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
            val_input.valueChanged.connect(lambda value, lbl=label: self.on_spinbox_changed(lbl, value))
            self.slider_values[label] = val_input

            slider_layout.addWidget(lbl)
            slider_layout.addWidget(sld)
            slider_layout.addWidget(val_input)
            sliders_layout.addLayout(slider_layout)
            self.sliders[label] = (sld, min_val, max_val)

        # Checkboxes
        self.checkboxes = {}
        checkbox_params = [
            ("Enable Frequency Peak Suppression", self.update_image),
            ("Enable Attenuation (Gamma Correction)", self.update_image),
            ("Include Central Preservation Mask", self.update_image),
            ("Invert Overall Mask", self.update_image),
            ("Enable Anti-Aliasing Filter", self.update_image),  # New checkbox
        ]

        for label, slot in checkbox_params:
            cb = QtWidgets.QCheckBox(label)
            cb.stateChanged.connect(slot)
            cb.setToolTip(f"Toggle {label.lower()}.")
            checkboxes_layout.addWidget(cb)
            self.checkboxes[label] = cb

        # Buttons
        # Load Image Button
        load_button = QtWidgets.QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        load_button.setToolTip("Load an image for processing.")
        buttons_layout.addWidget(load_button)

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

        # Connect canvas events for hover functionality
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    # Event Handling Methods
    def eventFilter(self, source, event):
        """Event filter to detect when the user hovers over specific sliders."""
        if event.type() == QtCore.QEvent.Enter:
            if source.objectName() == "Exclude Radius (%)":
                self.hovered_slider = "Exclude Radius (%)"
                self.update_image()
            elif source.objectName() == "Gaussian Blur Sigma (%)":
                self.hovered_slider = "Gaussian Blur Sigma (%)"
                self.update_image()
            else:
                self.hovered_slider = None
        elif event.type() == QtCore.QEvent.Leave:
            self.hovered_slider = None
            self.update_image()
        return super().eventFilter(source, event)

    def get_slider_value(self, label):
        """Retrieve the current value of a slider."""
        sld, min_val, max_val = self.sliders[label]
        if max_val - min_val == 0:
            return min_val
        value = sld.value() / sld.maximum() * (max_val - min_val) + min_val
        return value

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

    def closeEvent(self, event):
        """Handle the window close event."""
        if self.unsaved_changes and self.im_new is not None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the processed image before exiting?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Save,
            )
            if reply == QtWidgets.QMessageBox.Save:
                self.save_image()
                event.accept()
            elif reply == QtWidgets.QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # Hover Method for Original Image
    def on_motion(self, event):
        """Handle mouse motion events to show original image on hover."""
        if hasattr(self, 'ax3') and event.inaxes == self.ax3:
            if not self.show_original:
                self.show_original = True
                self.plot_images()
        else:
            if self.show_original:
                self.show_original = False
                self.plot_images()

    def reset_zoom(self):
        """Reset the zoom on the processed image."""
        if hasattr(self, 'ax3') and self.ax3 is not None:
            # Reset the view using the original image extents
            self.ax3.set_xlim(0, self.im_new.shape[1])
            self.ax3.set_ylim(self.im_new.shape[0], 0)  # Reverse y-axis limits
            self.canvas.draw()

    # Parameter Management Methods
    def load_default_parameters(self):
        """Load default parameters if available."""
        try:
            with open('default_parameters.json', 'r') as f:
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
        params.update({label: self.checkboxes[label].isChecked() for label in self.checkboxes})
        with open('default_parameters.json', 'w') as f:
            json.dump(params, f)
        QtWidgets.QMessageBox.information(self, "Default Parameters Set", "Current parameters have been set as default.")

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
            params.update({label: self.checkboxes[label].isChecked() for label in self.checkboxes})
            with open(file_path, 'w') as f:
                json.dump(params, f)
            QtWidgets.QMessageBox.information(self, "Parameters Saved", f"Parameters saved to {file_path}")

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
            with open(file_path, 'r') as f:
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
            QtWidgets.QMessageBox.information(self, "Parameters Loaded", f"Parameters loaded from {file_path}")

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
            # Load the image using OpenCV (supports various formats)
            im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load the image.")
                return
            im = im.astype(np.float32)
            self.im = cp.asarray(im)

            # Compute FFT
            im_fft = cp_fft.fft2(self.im)
            self.im_fft_shifted = cp_fft.fftshift(im_fft)

            # Update the image
            self.update_image()

    def save_image(self):
        """Save the processed image using a file dialog."""
        if self.im_new is not None:
            options = QtWidgets.QFileDialog.Options()
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Processed Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif *.tiff)",
                options=options,
            )
            if save_path:
                # Convert CuPy array to NumPy array and save using OpenCV
                im_to_save = cp.asnumpy(self.im_new)
                im_to_save = np.clip(im_to_save, 0, 255).astype(np.uint8)
                cv2.imwrite(save_path, im_to_save)
                self.unsaved_changes = False  # Reset unsaved changes flag
                QtWidgets.QMessageBox.information(self, "Image Saved", f"Image saved to {save_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "No Image", "No processed image to save.")

    def batch_process(self):
        """Batch process images in a selected directory."""
        input_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if input_dir:
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not output_dir:
                QtWidgets.QMessageBox.warning(self, "Error", "No output directory selected.")
                return
            # Process all images in the input directory
            image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            for file_name in image_files:
                file_path = os.path.join(input_dir, file_name)
                # Load the image
                im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue  # Skip files that cannot be loaded
                im = im.astype(np.float32)
                self.im = cp.asarray(im)
                # Compute FFT
                im_fft = cp_fft.fft2(self.im)
                self.im_fft_shifted = cp_fft.fftshift(im_fft)
                # Update the image
                self.update_image()
                # Save the processed image
                im_to_save = cp.asnumpy(self.im_new)
                im_to_save = np.clip(im_to_save, 0, 255).astype(np.uint8)
                save_path = os.path.join(output_dir, file_name)
                cv2.imwrite(save_path, im_to_save)
            QtWidgets.QMessageBox.information(self, "Batch Processing", "Batch processing completed.")

    def update_image(self):
        """Update the processed image based on the current settings."""
        if self.im_fft_shifted is None:
            return

        # Retrieve image dimensions
        H, W = self.im_fft_shifted.shape

        # Calculate center coordinates (always define crow and ccol)
        r, c = H, W
        crow, ccol = r // 2, c // 2

        # Retrieve slider and checkbox values
        gaussian_sigma_pct = self.slider_values["Gaussian Blur Sigma (%)"].value()
        peak_min_distance = self.slider_values["Peak Min Distance"].value()
        peak_threshold = self.slider_values["Peak Threshold"].value()
        exclude_radius_pct = self.slider_values["Exclude Radius (%)"].value()
        mask_radius_pct = self.slider_values["Mask Radius (%)"].value()
        peak_mask_falloff_pct = self.slider_values["Peak Mask Falloff (%)"].value()
        gamma_correction = self.slider_values["Gamma Correction"].value()
        central_mask_radius_pct = self.slider_values["Central Mask Radius (%)"].value()
        central_mask_aspect_ratio = self.slider_values["Central Mask Aspect Ratio"].value()
        central_mask_rotation = self.slider_values["Central Mask Rotation"].value()
        central_mask_falloff_pct = self.slider_values["Central Mask Falloff (%)"].value()
        antialiasing_intensity = self.slider_values["Anti-Aliasing Intensity (%)"].value() / 100.0  # Adjusted

        # Convert relative measurements to absolute pixels
        diag_length = np.sqrt(H**2 + W**2)
        gaussian_sigma = (gaussian_sigma_pct / 100) * diag_length
        exclude_radius = (exclude_radius_pct / 100) * diag_length / 2
        mask_radius = (mask_radius_pct / 100) * diag_length / 2
        peak_mask_falloff = (peak_mask_falloff_pct / 100) * diag_length / 2
        central_mask_radius = (central_mask_radius_pct / 100) * diag_length / 2
        central_mask_falloff = (central_mask_falloff_pct / 100) * diag_length / 2

        # Retrieve checkbox values
        enable_peak_suppression = self.checkboxes["Enable Frequency Peak Suppression"].isChecked()
        enable_attenuation = self.checkboxes["Enable Attenuation (Gamma Correction)"].isChecked()
        include_central_mask = self.checkboxes["Include Central Preservation Mask"].isChecked()
        invert_overall_mask = self.checkboxes["Invert Overall Mask"].isChecked()
        enable_antialiasing = self.checkboxes["Enable Anti-Aliasing Filter"].isChecked()

        # Preprocess the FFT magnitude spectrum
        magnitude_spectrum = cp.abs(self.im_fft_shifted)
        if gaussian_sigma > 0:
            blurred_spectrum = cp_gaussian_filter(magnitude_spectrum, sigma=gaussian_sigma)
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
            y, x = np.ogrid[:r, :c]
            distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
            exclusion_mask = distance > exclude_radius

            # Detect peaks
            coordinates = peak_local_max(
                spectrum_np,
                min_distance=int(peak_min_distance),
                threshold_abs=peak_threshold * spectrum_np.max(),
                exclude_border=False,
                labels=exclusion_mask.astype(np.uint8)
            )

            # Create masks for each detected peak
            peak_masks = []
            for coord in coordinates:
                mask = self.create_circular_mask(
                    self.im_fft_shifted.shape,
                    center=coord,
                    radius=mask_radius,
                    falloff=peak_mask_falloff
                )
                peak_masks.append(mask)

            # Combine all peak masks
            if peak_masks:
                combined_peak_mask = peak_masks[0]
                for mask in peak_masks[1:]:
                    combined_peak_mask *= mask
            else:
                combined_peak_mask = cp.ones(self.im_fft_shifted.shape, dtype=cp.float32)
        else:
            combined_peak_mask = cp.ones(self.im_fft_shifted.shape, dtype=cp.float32)

        # Create central preservation mask
        if include_central_mask:
            central_mask = self.create_elliptical_mask(
                self.im_fft_shifted.shape,
                center=(crow, ccol),
                radius=central_mask_radius,
                aspect_ratio=central_mask_aspect_ratio,
                rotation=central_mask_rotation,
                falloff=central_mask_falloff
            )
        else:
            central_mask = cp.ones(self.im_fft_shifted.shape, dtype=cp.float32)

        # Apply central mask to combined peak mask to protect central peak
        combined_mask = combined_peak_mask * central_mask

        # Optionally invert the overall mask
        if invert_overall_mask:
            combined_mask = 1 - combined_mask

        # Step 1: Apply the frequency peak suppression mask
        im_fft_filtered = self.im_fft_shifted * combined_mask

        if enable_peak_suppression and enable_attenuation:
            # Step 2: Create suppression areas mask (inverted mask)
            suppression_areas_mask = 1 - combined_mask
            # Step 3: Apply gamma correction to frequencies in suppression areas
            im_fft_suppression = self.im_fft_shifted * suppression_areas_mask * attenuation_mask
            # Step 4: Add the attenuated frequencies back to the filtered FFT image
            im_fft_filtered = im_fft_filtered + im_fft_suppression

        # Apply Anti-Aliasing Filter if enabled
        if enable_antialiasing:
            antialiasing_mask = self.create_antialiasing_mask(self.im_fft_shifted.shape, antialiasing_intensity)
            im_fft_filtered *= antialiasing_mask

        # Store the filtered FFT for plotting
        self.im_fft_filtered = im_fft_filtered

        # Inverse FFT to reconstruct the image
        im_ifft = cp_fft.ifftshift(im_fft_filtered)
        self.im_new = cp.abs(cp_fft.ifft2(im_ifft))

        # Set unsaved changes flag
        self.unsaved_changes = True

        # Update the plot
        self.plot_images()

    # Mask Creation Functions
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

    def create_elliptical_mask(self, shape, center, radius, aspect_ratio, rotation, falloff=0):
        """Create an elliptical mask centered at 'center' with given 'radius', 'aspect_ratio', 'rotation', and 'falloff'."""
        H, W = shape
        Y, X = cp.ogrid[:H, :W]
        cy, cx = center
        # Adjust coordinates
        Xc = X - cx
        Yc = Y - cy
        # Apply rotation
        theta = np.deg2rad(rotation)
        Xr = Xc * cp.cos(theta) + Yc * cp.sin(theta)
        Yr = -Xc * cp.sin(theta) + Yc * cp.cos(theta)
        # Apply aspect ratio
        distance = (Xr ** 2) / (radius ** 2) + (Yr ** 2) / ((radius * aspect_ratio) ** 2)
        mask = cp.ones(shape, dtype=cp.float32)
        mask[distance <= 1] = 1  # Preserve frequencies within the ellipse
        mask[distance > 1] = 0   # Suppress frequencies outside the ellipse

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
        # Adjust intensity to invert scale
        adjusted_intensity = intensity  # From 0 to 1
        mask = 1 - adjusted_intensity * (radius / max_radius)
        mask = cp.clip(mask, 0, 1)
        return mask

    # Visualization Method
    def plot_images(self):
        """Plot the original image, FFT spectrum, and processed image."""
        # Save current xlim and ylim before clearing
        if hasattr(self, 'ax3') and self.ax3 is not None:
            xlim = self.ax3.get_xlim()
            ylim = self.ax3.get_ylim()
        else:
            xlim, ylim = None, None

        self.figure.clear()
        ax1 = self.figure.add_subplot(1, 3, 1)
        ax2 = self.figure.add_subplot(1, 3, 2)
        ax3 = self.figure.add_subplot(1, 3, 3)
        self.ax3 = ax3  # Store ax3 for future reference

        # Original Image
        ax1.imshow(cp.asnumpy(self.im), cmap='gray', origin='upper')
        ax1.set_title('Original Image')
        ax1.axis('off')

        # FFT Spectrum
        if self.hovered_slider == "Gaussian Blur Sigma (%)" and self.blurred_spectrum is not None:
            # Display blurred spectrum
            magnitude_spectrum = cp.log(cp.abs(self.blurred_spectrum) + 1)
            ax2.set_title('Blurred FFT Spectrum')
        else:
            # Display filtered FFT spectrum
            magnitude_spectrum = cp.log(cp.abs(self.im_fft_filtered) + 1)
            ax2.set_title('Filtered FFT Spectrum')

        ax2.imshow(cp.asnumpy(magnitude_spectrum), cmap='gray', origin='upper')

        # Overlay exclusion radius visualization
        if self.hovered_slider == "Exclude Radius (%)":
            exclude_radius_pct = self.get_slider_value("Exclude Radius (%)")
            H, W = magnitude_spectrum.shape
            diag_length = np.sqrt(H**2 + W**2)
            exclude_radius = (exclude_radius_pct / 100) * diag_length / 2
            circle = plt.Circle((W / 2, H / 2), exclude_radius, color='red', fill=False, linewidth=1)
            ax2.add_artist(circle)

        ax2.axis('off')

        # Processed Image or Original Image
        if self.show_original:
            im_display = cp.asnumpy(self.im)
            ax3.set_title('Original Image')
        else:
            im_display = cp.asnumpy(self.im_new)
            ax3.set_title('Processed Image')

        ax3.imshow(im_display, cmap='gray', origin='upper')
        ax3.axis('off')

        # Restore xlim and ylim if they exist
        if xlim and ylim:
            ax3.set_xlim(xlim)
            ax3.set_ylim(ylim)

        self.canvas.draw()

# Run the Application
def main():
    """Main function to run the application."""
    app = QtWidgets.QApplication(sys.argv)
    ex = FFTImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
