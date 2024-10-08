# FFT Pattern Suppression for Image Restoration

**Revision:** 2024-10-08

**Changes:**
+ **Rev. 2024-10-08 Updates:**
+ - Modularization: The application has been modularized to improve maintainability, scalability, and code quality. The modular structure organizes the application into logical components, each with a specific responsibility. For more details, please refer to the [Modularization Documentation](docs/MODULARIZATION.md).

## Table of Contents

1. [Examples](#examples)
2. [Overview](#overview)
3. [Features](#features)
4. [Target Audience](#target-audience)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Performance Note](#performance-note)
8. [Project Structure](#project-structure)
9. [License](#license)
10. [Contributing](#contributing)
11. [Disclaimer](#disclaimer)

## Examples

### Halftone Pattern Example
![Example 1 Composite](examples/Example_1_composite.png)

### Application Screenshot
![Mezzotint Pattern](examples/03_1c_screenshot_mezzotint_pattern_Saint_Agnes_crop_640px.png)

### [Other Examples](examples/)

## Overview

FFT Pattern Suppression is a tool designed to automatically detect and remove unwanted periodic patterns from images, such as:

- **Halftone Dots:** Repetitive grid-like patterns from printing techniques.
- **Paper Textures:** Tiny embossed or debossed circles from photographic paper.
- **Offset Printing Patterns:** Halftone dots arranged in a grid in color images.

The application restores clarity and fidelity to images by suppressing these patterns by leveraging Fast Fourier Transform (FFT) filtering techniques.
+ **New Features:**
+ - **High-Pass Filtering Preprocessing:** Separates high-frequency details from low-frequency color information to reduce artifacts.
+ - **Advanced Masking Techniques:** Offers finer control over masking areas with aspect ratio, orientation, and falloff modifiers.
+ - **Border Expansion:** Minimizes edge artifacts by adding padding during FFT processing.
+ - **Performance Optimizations:** Enhances responsiveness by limiting processing to the displayed area at appropriate resolutions.
+ - **Stepwise Workflow UI:** Guides users through a logical, phase-based filtering process for improved usability.

## Features

- **Automatic Detection and Removal:** Utilizes FFT to detect and suppress unwanted frequency components.
- **Real-Time Visualization:** Interactive filtering with immediate visual feedback.
- **Parameter Management:** Save, load, and set default filter settings.
- **Batch Processing:** Efficiently process large collections of images.
- **GPU Acceleration (Optional):** Leverages GPU via CuPy and CUDA for high-performance processing on systems with NVIDIA GPUs.
+ **Enhanced Features in Rev.2024-10-05:**
+ - **High-Pass Filtering Preprocessing:** Mitigates artifacts and supports color image processing.
+ - **Optimized Color Image Processing:** Maintains color fidelity across RGB channels.
+ - **Advanced Masking Techniques:** Includes aspect ratio, orientation, and falloff modifiers for finer control.
+ - **Border Expansion:** Adds padding to minimize edge artifacts during FFT processing.
+ - **Performance Optimizations:** Limits processing to displayed areas for improved responsiveness.
+ - **Stepwise Workflow UI:** Guides users through a logical, phase-based filtering process.

## Target Audience

- Restoration experts and archivists
- Photographers and designers
- Printing and publishing professionals
- Researchers and historians

## Installation

Please refer to the [Installation Guide](INSTALLATION.md) for installation instructions.

## Usage

1. **Load an Image:** Click on "Load Image" and select the image you want to process.
2. **Adjust Parameters:** Use the sliders and checkboxes to fine-tune the filtering process.
3. **Real-Time Feedback:** View the original image, FFT spectrum, and processed image side by side.
4. **Save the Processed Image:** Click on "Save Processed Image" to save your results.

+ **Rev.2024-10-05 Usage Enhancements:**
+ 1. **High-Pass Filtering:** Begin by applying high-pass filters to separate high-frequency details.
+ 2. **Detailed Filtering:** Utilize advanced masking techniques to target specific frequency components.
+ 3. **Border Expansion:** Ensure minimal edge artifacts by enabling border expansion during processing.
+ 4. **Performance Optimization:** Experience improved responsiveness with area-specific processing.
+ 5. **Stepwise Workflow:** Follow the guided workflow for an efficient filtering process.

## Performance Note

- **With CUDA and CuPy:** The application utilizes GPU acceleration for faster processing.
- **Without CUDA and CuPy:** The application will run using CPU only, and performance may be degraded, especially with large images.

## Project Structure

```bash
fft_image_processing_app/
├── controllers/
│ ├── init.py
│ ├── image_controller.py
│ ├── main_controller.py
│ └── processing_controller.py
├── models/
│ ├── init.py
│ ├── image_model.py
│ └── parameters_model.py
├── views/
│ ├── init.py
│ ├── main_window.py
│ ├── phase1_view.py
│ └── phase2_view.py
├── processing/
│ ├── init.py
│ ├── fft_processor.py
│ ├── mask_generator.py
│ └── utils.py
├── utils/
│ ├── init.py
│ └── file_handler.py
├── tests/
│ ├── init.py
│ └── test_fft_processor.py
├── main.py
├── requirements.txt
├── requirements_cpu.txt
├── install.bat
├── install.sh
├── LICENSE.md
├── CONTRIBUTING.md
├── INSTALLATION.md
└── README.md
```

- **controllers/**: Manages application logic and interactions.
- **models/**: Handles data structures and parameter management.
- **views/**: Manages UI components.
- **processing/**: Contains image processing logic.
- **utils/**: Provides utility functions and handles dependencies.
- **tests/**: Contains unit tests to ensure module functionality.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE.md) file for details.

### AGPL-3.0 Summary:

- You can use, modify, and distribute this software freely.
- If you distribute the software or a modified version, you must:
  - Provide the complete source code.
  - License your modifications under the same AGPL-3.0 terms.
  - Preserve the original copyright notices.
- If you run a modified version of the software on a server and allow users to interact with it remotely, you must also make the source code of your modified version available to those users.
- There's no warranty for the software.

This summary is not a substitute for the full license. Always refer to the complete [LICENSE](LICENSE.md) for legal details.

## Contributing

Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## Disclaimer

This project is a personal learning exercise. While user feedback is invaluable, please note that thorough support or continuous development beyond the current scope may not be provided.