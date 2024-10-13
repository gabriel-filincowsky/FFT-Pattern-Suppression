# Modularization of FFT Pattern Suppression Application

## Table of Contents

1. [Introduction](#introduction)
2. [Modular Structure](#modular-structure)
3. [Component Descriptions](#component-descriptions)
4. [Interactions and Data Flow](#interactions-and-data-flow)
5. [Best Practices and Design Patterns](#best-practices-and-design-patterns)
6. [Conclusion](#conclusion)

## Introduction

The FFT Pattern Suppression application has undergone a significant modularization process to enhance its maintainability, scalability, and overall code quality. This document provides a comprehensive overview of the modularized structure, detailing the various components, their interactions, and the design principles employed.

The primary goals of this modularization effort were to:

1. Separate concerns and improve code organization
2. Enhance reusability of components
3. Facilitate easier testing and debugging
4. Improve code readability and maintainability
5. Allow for easier future enhancements and feature additions

By breaking down the application into distinct modules with clear responsibilities, we've created a more robust and flexible system that can adapt to future requirements more easily.

## Modular Structure

The modularized FFT Pattern Suppression application is organized into the following directory structure:

```bash
fft_image_processing_app/
├── controllers/
│   ├── __init__.py
│   ├── image_controller.py
│   ├── main_controller.py
│   └── processing_controller.py
├── models/
│   ├── __init__.py
│   ├── image_model.py
│   └── parameters_model.py
├── views/
│   ├── __init__.py
│   ├── main_window.py
│   ├── phase1_view.py
│   └── phase2_view.py
├── processing/
│   ├── __init__.py
│   ├── fft_processor.py
│   ├── mask_generator.py
│   └── utils.py
├── utils/
│   ├── __init__.py
│   ├── file_handler.py
│   └── config_manager.py
├── tests/
│   ├── __init__.py
│   └── test_fft_processor.py
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

This structure organizes the application into logical modules, each with a specific responsibility:

1. `controllers`: Manages the application's control flow and user interactions.
2. `models`: Handles data representations and business logic.
3. `views`: Manages the graphical user interface and visualization components.
4. `processing`: Contains the core FFT processing and pattern suppression logic.
5. `utils`: Provides utility functions and configuration management.
6. `main.py`: Serves as the entry point for the application.

## Component Descriptions

### 1. Controllers Module

The `controllers` module is responsible for managing the application's control flow and handling user interactions. It consists of three main components:

#### a. image_controller.py

This module manages image loading, saving, and basic image operations.

Key functions:
- `load_image(file_path: str) -> Image`: Loads an image from the specified file path.
- `save_image(image: Image, file_path: str) -> None`: Saves the processed image to the specified file path.

#### b. main_controller.py

This module orchestrates the overall workflow of the application, coordinating between different modules to perform FFT processing and pattern suppression.

Key functions:
- `initialize_application() -> None`: Sets up the application environment.
- `start_processing() -> None`: Initiates the FFT pattern suppression process.

#### c. processing_controller.py

This module handles the coordination of data processing and FFT operations.

Key functions:
- `process_data(data: pd.DataFrame) -> np.ndarray`: Preprocesses input data for FFT.
- `execute_fft(signal: np.ndarray) -> np.ndarray`: Executes FFT on the preprocessed signal.

### 2. Processing Module

The `processing` module contains the core FFT processing and pattern suppression logic. It is divided into two main components:

#### a. fft_processor.py

This module handles the Fast Fourier Transform calculations and related operations.

Key functions:
- `compute_fft(signal: np.ndarray) -> np.ndarray`: Computes the FFT of the input signal.
- `compute_inverse_fft(fft_result: np.ndarray) -> np.ndarray`: Computes the inverse FFT.
- `apply_frequency_filter(fft_result: np.ndarray, filter_func: Callable) -> np.ndarray`: Applies a frequency domain filter to the FFT result.

#### b. mask_generator.py

This module generates masks for suppressing unwanted frequency patterns.

Key functions:
- `create_mask(patterns: List[Dict]) -> np.ndarray`: Creates a mask based on identified patterns.
- `apply_mask(fft_result: np.ndarray, mask: np.ndarray) -> np.ndarray`: Applies the generated mask to the FFT result.

#### c. utils.py

This module provides utility functions specific to the processing tasks.

Key functions:
- `normalize_fft(fft_result: np.ndarray) -> np.ndarray`: Normalizes the FFT result.
- `detect_peaks(fft_result: np.ndarray) -> List[Dict]`: Detects peaks in the FFT spectrum.

### 3. Views Module

This module provides the user interface and visualization capabilities for the application.

#### main_window.py

This file contains the main window class, which handles the display of images and user interactions.

Key functions:
- `update_image_display()`: Updates the processed image displayed in the UI.
- `batch_process()`: Handles batch processing of images.

#### phase1_view.py & phase2_view.py

These files manage the specific views for different phases of the image processing workflow.

Key functions:
- `display_phase1_results()`: Displays results for Phase 1 processing.
- `display_phase2_results()`: Displays results for Phase 2 processing.

### 4. Utils Module

The `utils` module contains utility functions and configuration management for the application.

#### a. config_manager.py

This module manages the application's configuration, including user-configurable parameters and their validation.

- **Key Components:**
  - **`ConfigManager` Class:** Handles loading, saving, and accessing configuration parameters.
  - **`DEFAULT_CONFIG_PATH`:** Path to the default configuration JSON file (`config/default_parameters.json`).
  - **`VALIDATION_RULES` Dictionary:** Defines expected types, minimum, and maximum values for parameters to ensure data integrity.
  - **Validation:** The `validate_config` method ensures that all parameters meet their defined constraints before being loaded into the application.

**Configuration Parameters:**

| Parameter                      | Type    | Description                            | Units       | Constraints                      |
|--------------------------------|---------|----------------------------------------|-------------|----------------------------------|
| High-Pass Filter Radius        | float   | Radius of the high-pass filter         | pixels      | min: 0.1, max: 25.0, precision: 1 |
| Gaussian Blur (%)              | float   | Percentage of Gaussian blur applied    | percent     | min: 0.1, max: 100.0, precision: 1 |
| Peak Min Distance              | int     | Minimum distance between peaks         | pixels      | min: 1, max: N/A                  |
| Peak Threshold                 | float   | Threshold for peak detection           | unitless    | min: 0.0001, max: 1.0             |
| Radius (%)                     | float   | Radius percentage                      | percent     | min: N/A, max: N/A, precision: N/A |
| Aspect Ratio                   | float   | Aspect ratio for processing            | unitless    | min: N/A, max: N/A, precision: N/A |
| Orientation                    | float   | Orientation angle                      | degrees     | min: N/A, max: N/A, precision: N/A |
| Falloff (%)                    | float   | Falloff percentage                     | percent     | min: N/A, max: N/A, precision: N/A |
| Mask Radius (%)                | float   | Radius percentage for masking          | percent     | min: N/A, max: N/A, precision: N/A |
| Peak Mask Falloff (%)          | float   | Falloff percentage for peak masking    | percent     | min: N/A, max: N/A, precision: N/A |
| Gamma Correction               | float   | Gamma correction factor                | unitless    | min: N/A, max: N/A, precision: N/A |
| Anti-Aliasing Intensity (%)    | float   | Intensity of anti-aliasing filter      | percent     | min: N/A, max: N/A, precision: N/A |
| Enable Frequency Peak Suppression | bool | Toggle frequency peak suppression      | boolean     | True/False                        |
| Enable Attenuation             | bool    | Toggle attenuation                     | boolean     | True/False                        |
| Enable Anti-Aliasing Filter    | bool    | Toggle anti-aliasing filter            | boolean     | True/False                        |

**Usage Example:**

```python
from utils.config_manager import ConfigManager

config_manager = ConfigManager()
highpass_radius = config_manager.get_parameter("High-Pass Filter Radius")
config_manager.set_parameter("Gaussian Blur (%)", 1.2)
```

#### b. constants.py

This module defines constant values used throughout the application. These constants are not user-configurable and represent intrinsic properties or limits of the system.

**Key Constants:**
- `PADDING_SIZE`: Number of pixels to pad on each side of the image (16 pixels)
- `MAX_HIGH_PASS_RADIUS`: Maximum allowed high-pass filter radius (25.0 pixels)
- `MIN_HIGH_PASS_RADIUS`: Minimum allowed high-pass filter radius (0.1 pixels)
- `MAX_GAUSSIAN_BLUR_PCT`: Maximum allowed Gaussian blur percentage (100.0%)
- `MIN_GAUSSIAN_BLUR_PCT`: Minimum allowed Gaussian blur percentage (0.1%)
- `MAX_PEAK_THRESHOLD`: Maximum allowed peak threshold value (1.0)
- `MIN_PEAK_THRESHOLD`: Minimum allowed peak threshold value (0.0001)

**Usage Example:**

```python
from utils.constants import PADDING_SIZE, MAX_HIGH_PASS_RADIUS

# Use constants in your code
padded_image = pad_image(original_image, PADDING_SIZE)
if radius > MAX_HIGH_PASS_RADIUS:
    raise ValueError(f"High-pass filter radius cannot exceed {MAX_HIGH_PASS_RADIUS} pixels")
```

### 5. Main Application (main.py)

The `main.py` file serves as the entry point for the application. It orchestrates the overall flow of the program, utilizing the various modules to perform the complete FFT pattern suppression process.

Key responsibilities:
- Parsing command-line arguments
- Loading and validating configuration
- Coordinating the data processing, FFT computation, pattern suppression, and visualization steps
- Handling error conditions and providing user feedback

## Interactions and Data Flow

The modularized application follows a clear data flow:

1. The main application initializes and loads the configuration using the `controllers.main_controller` module.
2. User inputs are handled by the `controllers.image_controller` module.
3. Data is loaded and managed using the `controllers.processing_controller` module.
4. The loaded data is preprocessed and passed to the `processing.fft_processor` for FFT computation.
5. The FFT results are then processed by the `processing.mask_generator` to identify and suppress patterns.
6. Suppressed FFT results are converted back to the spatial domain.
7. Throughout the process, the `views` module is used to generate plots and update the user interface.
8. The `utils.config_manager` module provides configuration management utilized by various components.

This modular structure allows for clear separation of concerns and enables easy modification or replacement of individual components without affecting the entire system.

## Best Practices and Design Patterns

The modularization of the FFT Pattern Suppression application incorporates several best practices and design patterns:

1. **Single Responsibility Principle**: Each module and class has a single, well-defined responsibility, improving maintainability and reducing coupling.

2. **Dependency Injection**: The application uses dependency injection to provide required components, making it easier to swap implementations and improve testability.

3. **Factory Pattern**: Used in the data loading process to create appropriate data loader objects based on the input type.

4. **Strategy Pattern**: Employed in the pattern suppression module to allow for different suppression algorithms to be easily swapped.

5. **Observer Pattern**: Implemented for logging and progress tracking, allowing various components to be notified of important events without tight coupling.

6. **Configuration Management**: Centralized configuration management allows for easy adjustment of application parameters without code changes.

7. **Error Handling**: Comprehensive error handling and logging throughout the application improve robustness and debuggability.

8. **Type Hinting**: Extensive use of type hints improves code readability and enables better IDE support and static type checking.

9. **Docstrings and Comments**: All modules, classes, and functions are documented with clear docstrings, explaining their purpose, parameters, and return values.

10. **Unit Testing**: The modular structure facilitates comprehensive unit testing, with each module having its own set of tests in the `tests/` directory.

## Conclusion

The modularization of the FFT Pattern Suppression application has resulted in a well-structured, maintainable, and extensible codebase. By separating concerns into distinct modules and following best practices in software design, we have created a robust foundation that can easily accommodate future enhancements and modifications.

This modular architecture not only improves the current functionality but also sets the stage for potential future features, such as:

- Support for additional data input formats
- Implementation of new pattern suppression algorithms
- Enhanced visualization capabilities
- Integration with other signal processing techniques

The clear separation of responsibilities and well-defined interfaces between modules make it straightforward to extend or modify the application's functionality while minimizing the risk of introducing bugs or unintended side effects.
