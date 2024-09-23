# FFT Pattern Suppression for Image Restoration

**Version:** 2024-09-22

## Overview

FFT Pattern Suppression is a tool designed to automatically detect and remove unwanted periodic patterns from images, such as:

- **Halftone Dots:** Repetitive grid-like patterns from printing techniques.
- **Paper Textures:** Tiny embossed or debossed circles from photographic paper.
- **Offset Printing Patterns:** Halftone dots arranged in a grid in color images.

The application restores clarity and fidelity to images by suppressing these patterns by leveraging Fast Fourier Transform (FFT) filtering techniques.

## Features

- **Automatic Detection and Removal:** Utilizes FFT to detect and suppress unwanted frequency components.
- **Real-Time Visualization:** Interactive filtering with immediate visual feedback.
- **Parameter Management:** Save, load, and set default filter settings.
- **Batch Processing:** Efficiently process large collections of images.
- **GPU Acceleration (Optional):** Leverages GPU via CuPy and CUDA for high-performance processing on systems with NVIDIA GPUs.

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

## Performance Note

- **With CUDA and CuPy:** The application utilizes GPU acceleration for faster processing.
- **Without CUDA and CuPy:** The application will run using CPU only, and performance may be degraded, especially with large images.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## Disclaimer

This project is a personal learning exercise. While user feedback is invaluable, please note that thorough support or continuous development beyond the current scope may not be provided.