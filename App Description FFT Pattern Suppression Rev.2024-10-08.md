**FFT Pattern Suppression for Image Restoration**

**Rev. 2024-10-08**

---

**Purpose**

The primary goal of this application is to **automatically detect and remove periodic patterns** from images, particularly those found in old photo prints, scanned documents, and materials produced through halftone or offset printing. These periodic patterns degrade the visual quality of images, introducing unwanted regular structures that obscure finer details.

**Enhancements in this Revision:**

- **Modularized Codebase:** The application has been restructured into a modular architecture to improve maintainability, scalability, and facilitate future enhancements. See [MODULARIZATION.md](docs/MODULARIZATION.md) for detailed information.

---

**Target Audience**

-   **Restoration Experts and Archivists**: Professionals working with old photographs, prints, and historical documents who need to digitally restore and clean up these materials for preservation or presentation.
    
-   **Photographers and Designers**: Individuals seeking to enhance and clean up scanned materials, removing unwanted noise and artifacts from images.
    
-   **Printing and Publishing Professionals**: Those dealing with scanned or printed materials who want to eliminate unwanted printing artifacts for high-quality digital outputs.
    
-   **Researchers and Historians**: Anyone involved in digitizing and preserving historical documents or images who needs to clean up scans for clearer analysis.
    
---

**Key Goals of the Application**

1.  **Automatic Detection and Removal of Unwanted Periodic Patterns**:
    
    -   Utilize the **Fast Fourier Transform (FFT)** to convert images from the spatial domain to the frequency domain, where periodic patterns manifest as distinct peaks or repetitive structures.
    
    -   **Automatically detect and isolate unwanted frequency components** through advanced algorithms, reducing the need for manual intervention.
    
    -   Enable users to suppress and filter out these unwanted frequencies while preserving essential image details.
    
    -   **High-Pass Filtering Preprocessing**: Apply high-pass filters to separate high-frequency details, reducing artifacts and supporting color image processing.
    
2.  **Flexible and Intuitive Filtering with Real-Time Visualization**:
    
    -   Provide a range of **interactive filtering options**, allowing users to target and remove specific frequencies associated with periodic patterns.
    
    -   Feature an intuitive **graphical user interface (GUI)** that offers **real-time feedback** as users adjust filter parameters. Changes to parameters are immediately reflected in the processed image, allowing for iterative refinement and instant visual feedback.
    
    -   Display three previews simultaneously: the **original image**, the **FFT spectrum**, and the **processed image**, allowing users to visually compare results and refine settings dynamically.
    
    -   **Maintain zoom and pan settings** across image updates and enable users to hover over the processed image to view the original, aiding in precise adjustments.
    
    -   **Stepwise Workflow UI**: Guide users through a logical, phase-based filtering process to enhance usability and effectiveness.
    
3.  **Parameter Management and Time Efficiency**:
    
    -   Enable users to **save and load** their preferred filter settings, facilitating consistent processing across sessions.
    
    -   Provide the ability to set **default parameters**, ensuring a personalized starting point for each user.
    
    -   The combination of automatic detection, real-time feedback, and parameter management significantly **reduces processing time**, especially when dealing with multiple images with similar patterns and resolutions.
    
    -   Support for **batch processing** allows users to process large collections of images efficiently using saved presets, further enhancing productivity.
    
4.  **Image Restoration and Enhancement**:
    
    -   Aim to **restore images to their original quality** or improve them further, which is crucial for **archival purposes**, **image restoration**, or **preparing materials for digital presentation**.
    
    -   Produce cleaner images with enhanced clarity and minimal distracting periodic structures.
    
5.  **GPU Acceleration for Large-Scale Image Processing**:
    
    -   Leverage **GPU acceleration** via CuPy to ensure high-performance processing even with large images.
    
    -   Significantly improve the speed and responsiveness of the application, making computationally intensive FFT operations more efficient.
    
---

**Core Features and Filtering Techniques**

**Automatic Frequency Domain Filtering**

The application stands out by providing **automatic selection and removal** of unwanted areas in the frequency domain:

-   **Automated Peak Detection**:
    
    -   Employs advanced algorithms to detect peaks in the frequency domain that correspond to unwanted periodic patterns.
    
    -   Users can adjust detection parameters, but the core process of identifying problematic frequencies is automated, reducing manual effort.
    
-   **Dynamic Mask Generation**:
    
    -   Automatically generates masks around detected peaks to suppress unwanted frequencies.
    
    -   **Enhanced Masking Techniques**: Includes aspect ratio, orientation, and falloff modifiers for finer control over masking areas.
    
**Interactive Filtering Parameters**

The application offers a comprehensive set of parameters that users can adjust to fine-tune the filtering process:

1.  **Gaussian Blur Sigma (%)**:
    
    -   Controls the amount of Gaussian blur applied to the magnitude spectrum in the frequency domain, relative to the image size.
    
    -   Smoothing the spectrum aids in peak detection by reducing noise and minor fluctuations.
    
2.  **Peak Min Distance**:
    
    -   Sets the minimum distance between detected peaks in the frequency domain.
    
    -   Helps avoid detecting multiple peaks in close proximity, ensuring distinct periodic patterns are targeted.
    
3.  **Peak Threshold**:
    
    -   Determines the minimum intensity required for a peak to be considered in the frequency domain.
    
    -   Offers fine-grained control with up to **four decimal places of precision**, enabling detection of subtle patterns.
    
4.  **Exclude Radius (%)**:
    
    -   **Enhanced with Aspect Ratio, Orientation, and Falloff Modifiers**:
        
        -   Defines an elliptical exclusion zone in the frequency domain to prevent the central peak (DC component) from being detected as a pattern.
        
        -   Adjusting aspect ratio and orientation allows for protecting specific frequency orientations.
        
        -   Falloff ensures smooth transitions at the edges of the exclusion zone.
        
    -   Essential for focusing the filtering process on unwanted periodic patterns while preserving the overall brightness and low-frequency content of the image.
    
5.  **Mask Radius (%)**:
    
    -   Specifies the radius of circular masks applied around detected peaks to suppress unwanted frequencies.
    
    -   Larger masks can suppress broader frequency ranges associated with periodic patterns.
    
6.  **Peak Mask Falloff (%)**:
    
    -   Introduces a gradual attenuation around the edges of the peak masks using a Gaussian falloff.
    
    -   Helps avoid harsh transitions and artifacts in the filtered image by smoothing the suppression effect.
    
7.  **Gamma Correction**:
    
    -   Applies gamma correction to the attenuation mask, allowing for nonlinear attenuation of frequencies.
    
    -   Enhances control over how different frequencies are suppressed, particularly in suppression areas, without affecting preserved regions.
    
8.  **Anti-Aliasing Intensity (%)**:
    
    -   Controls the intensity of an anti-aliasing filter applied to suppress high-frequency components.
    
    -   Reduces aliasing artifacts and enhances image smoothness by attenuating frequencies beyond the Nyquist limit.
    
9.  **Enable/Disable Features**:
    
    -   Checkboxes allow users to enable or disable specific features:
        
        -   **Enable Frequency Peak Suppression**
        
        -   **Enable Attenuation (Gamma Correction)**
        
        -   **Invert Overall Mask**
        
        -   **Enable Anti-Aliasing Filter**
        
    -   Provides flexibility to apply only the necessary filters for a particular image.
    
---

**User Interface Highlights**

-   **Real-Time Visualization and Feedback**:
    
    -   Unlike other applications, our tool provides immediate visual feedback as parameters are adjusted.
    
    -   The real-time updates allow users to see the effects of their adjustments instantly, leading to more efficient and precise filtering.
    
-   **Automatic Selection of Unwanted Frequencies**:
    
    -   The application automatically detects and targets unwanted periodic patterns in the frequency domain, setting it apart from tools that require manual frequency selection.
    
    -   This automation significantly reduces the time and effort required to clean up images, especially when dealing with complex or numerous patterns.
    
-   **Parameter Management for Time Efficiency**:
    
    -   **Save and Retrieve Preset Parameters**:
        
        -   Users can save their customized settings as presets, which can be loaded for future sessions.
        
        -   This feature is particularly beneficial when processing multiple images with similar patterns and resolutions, enabling consistent results across batches.
        
    -   **Batch Processing Capability**:
        
        -   Leveraging saved presets, users can perform batch processing on a set of images, automating the removal of periodic patterns across large datasets.
        
        -   This capability is a significant time saver for professionals dealing with extensive archives or collections.
        
-   **Intuitive Controls and Navigation**:
    
    -   **Zoom and Pan Persistence**:
        
        -   The interface maintains zoom and pan settings when updating images, ensuring users can focus on specific areas without losing their place.
        
    -   **Hover Preview of Original Image**:
        
        -   Users can hover over the processed image to temporarily view the original image, aiding in assessing the effectiveness of the filtering.
        
    -   **Convenient Reset Zoom Functionality**:
        
        -   A "Reset Zoom" button is located under the processed image preview for quick resetting of the view.
        
-   **Stepwise Workflow**:
    
    -   **Phase 1: High-Pass Filtering**
        
        -   Start by determining the optimal high-pass filter radius through interactive adjustments.
        
        -   Preview the blurred and high-pass filtered images in real-time.
        
    -   **Phase 2: Detailed Filtering**
        
        -   Dive into detailed parameters for frequency suppression, masking, and attenuation.
        
        -   Adjust filters with immediate visual feedback to fine-tune the image restoration process.
        
    -   **Navigation Controls**:
        
        -   Easy transition between phases with "Next" and "Back" buttons to streamline the workflow.
        
---

**Modular Architecture and Design**

The application now features a modular architecture that separates concerns into distinct components, enhancing code readability, maintainability, and extensibility.

- **Modular Structure:**
    - `data_processing`: Handles data loading and preprocessing.
    - `fft`: Contains the core FFT processing and pattern suppression logic.
    - `visualization`: Manages plotting and visualization of results.
    - `utils`: Provides utility functions and configuration management.

For a comprehensive overview of the modular structure, please refer to the [MODULARIZATION.md](docs/MODULARIZATION.md) file.

---

**Notes:**

-   The development of this application is an ongoing process, and while user feedback is invaluable, I am not a professional developer, and my knowledge of coding is quite limited---below that of a beginner.
    
-   This project is a personal learning exercise, and as such, I am not committed to providing thorough support or continuing development beyond my current scope.
    
-   Anyone is free to use, modify, branch, or further develop this application. However, **access to the code must remain open source and free for everyone**. This includes the original application as well as any future iterations, branches, or derivative works related to the code. **The code, in any form, must not be sold or commercialized.**
    
-   Please note that this restriction applies only to the code of the application, not to the images or outputs produced by the application, which remain the property of the user.