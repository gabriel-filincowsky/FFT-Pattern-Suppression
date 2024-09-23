# Installation Guide

This guide will walk you through installing the FFT Pattern Suppression application on your computer.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

---

## Prerequisites

- **Operating System:** Windows 10 or higher, macOS, or Linux.
- **Python 3.6 or higher**
- **Optional: NVIDIA GPU and CUDA Toolkit for GPU Acceleration**
  - If you have an NVIDIA GPU and wish to utilize GPU acceleration, CUDA and CuPy are required.
- **Note:** The application can run without CUDA and CuPy, but performance may be degraded.

---

## Installation Steps

### Windows

#### 1. Install Python

- **Download Python:**
  - Visit the [Python Downloads](https://www.python.org/downloads/windows/) page.
  - Download **Python 3.9.x** Windows installer.

- **Install Python:**
  - Run the downloaded installer.
  - **Important:** Check the box that says **"Add Python to PATH"** during installation.
  - Follow the default installation steps.

#### 2. Install Git

- **Download Git:**
  - Visit [Git for Windows](https://git-scm.com/download/win).
  - Download the installer.

- **Install Git:**
  - Run the installer and follow the default settings.

#### 3. Optional: Install NVIDIA GPU Drivers and CUDA Toolkit

- **Check for NVIDIA GPU:**
  - If you have an NVIDIA GPU and wish to use GPU acceleration.

- **Install NVIDIA GPU Drivers:**
  - Visit the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page.
  - Select your GPU model and download the latest drivers.
  - Install the drivers and restart your computer if prompted.

- **Download and Install CUDA Toolkit:**
  - Visit the [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads).
  - Select your operating system and download the appropriate CUDA Toolkit installer.
  - Run the installer and follow the default settings.

#### 4. Download the Application

- **One-Click Installation:**
  - Download the `install.bat` script from the repository.
  - Right-click on `install.bat` and select **"Run as administrator"**.
  - The script will automatically install all necessary components and run the application.

- **Manual Download:**
  - Visit the [GitHub Repository](https://github.com/your_username/FFT-Pattern-Suppression).
  - Click on the green **"Code"** button and select **"Download ZIP"**.
  - Extract the ZIP file to a folder on your computer.

#### 5. Install Dependencies (If Not Using the Script)

- **Open Command Prompt:**
  - Press `Win + R`, type `cmd`, and press Enter.

- **Navigate to the Application Directory:**
  ```bash
  cd path\to\FFT-Pattern-Suppression
  ```

- **Create a Virtual Environment (Optional but Recommended):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **Upgrade Pip:**
  ```bash
  pip install --upgrade pip
  ```

- **Install Requirements:**
  - **Without CUDA (CPU Only):**
    ```bash
    pip install -r requirements_cpu.txt
    ```
  - **With CUDA (GPU Acceleration):**
    ```bash
    pip install cupy-cuda118  # Replace '118' with your CUDA version
    pip install -r requirements.txt
    ```

### macOS

#### 1. Install Python

- **Check if Python 3 is Installed:**
  - Open Terminal (`Command + Space`, type `Terminal`).
  - Run `python3 --version`.

- **If Not Installed:**
  - Visit the [Python Downloads](https://www.python.org/downloads/macos/) page.
  - Download **Python 3.9.x** installer.
  - Run the installer and follow the instructions.

#### 2. Install Git

- **Install Xcode Command Line Tools:**
  ```bash
  xcode-select --install
  ```

#### 3. Download the Application

- **One-Click Installation:**
  - Download the `install.sh` script from the repository.
  - Open Terminal and navigate to the download location.
  - Make the script executable:
    ```bash
    chmod +x install.sh
    ```
  - Run the script:
    ```bash
    ./install.sh
    ```

- **Manual Download:**
  - Visit the [GitHub Repository](https://github.com/your_username/FFT-Pattern-Suppression).
  - Click on the green **"Code"** button and select **"Download ZIP"**.
  - Extract the ZIP file to a folder on your computer.

#### 4. Install Dependencies (If Not Using the Script)

- **Open Terminal.**

- **Navigate to the Application Directory:**
  ```bash
  cd path/to/FFT-Pattern-Suppression
  ```

- **Create a Virtual Environment (Optional but Recommended):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Upgrade Pip:**
  ```bash
  pip install --upgrade pip
  ```

- **Install Requirements:**
  - **Without CUDA (CPU Only):**
    ```bash
    pip install -r requirements_cpu.txt
    ```
  - **With CUDA (GPU Acceleration):**
    - **Note:** NVIDIA GPUs are not supported on newer macOS versions. GPU acceleration may not be available.

### Linux

#### 1. Install Python and Git

- **Open Terminal.**

- **Update Package Lists:**
  ```bash
  sudo apt-get update
  ```

- **Install Python and Git:**
  ```bash
  sudo apt-get install python3 python3-venv python3-pip git
  ```

#### 2. Optional: Install NVIDIA GPU Drivers and CUDA Toolkit

- **Check for NVIDIA GPU:**
  - Use the command:
    ```bash
    lspci | grep -i nvidia
    ```

- **Install NVIDIA GPU Drivers and CUDA Toolkit:**
  - Follow the instructions specific to your Linux distribution.

#### 3. Download the Application

- **One-Click Installation:**
  - Download the `install.sh` script from the repository.
  - Make the script executable:
    ```bash
    chmod +x install.sh
    ```
  - Run the script:
    ```bash
    ./install.sh
    ```

- **Manual Download:**
  - Clone the repository:
    ```bash
    git clone https://github.com/your_username/FFT-Pattern-Suppression.git
    ```

#### 4. Install Dependencies (If Not Using the Script)

- **Navigate to the Application Directory:**
  ```bash
  cd FFT-Pattern-Suppression
  ```

- **Create a Virtual Environment (Optional but Recommended):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Upgrade Pip:**
  ```bash
  pip install --upgrade pip
  ```

- **Install Requirements:**
  - **Without CUDA (CPU Only):**
    ```bash
    pip install -r requirements_cpu.txt
    ```
  - **With CUDA (GPU Acceleration):**
    ```bash
    pip install cupy-cuda118  # Replace '118' with your CUDA version
    pip install -r requirements.txt
    ```

---

## Running the Application

- **Using the One-Click Script:**
  - The application should start automatically after installation.

- **Manual Start:**
  - **Windows:**
    ```bash
    python fft_pattern_suppression_app.py
    ```
  - **macOS/Linux:**
    ```bash
    python3 fft_pattern_suppression_app.py
    ```

---

## Troubleshooting

- **Performance is Slow:**
  - If you are running the application without CUDA and CuPy, performance may be slower.
  - Consider installing CUDA and CuPy if you have an NVIDIA GPU.

- **Python Not Recognized:**
  - Ensure Python is added to your system PATH.
  - Restart your computer after installation.

- **CUDA Errors (When Using GPU Acceleration):**
  - Verify that the CUDA Toolkit is properly installed and matches the `cupy` version.
  - Ensure your NVIDIA GPU is compatible with the CUDA version.
  - If you encounter errors related to CuPy, ensure that you have installed the correct `cupy-cuda` package for your CUDA version.

- **Permission Issues:**
  - Run the command prompt or terminal as an administrator or use `sudo` where necessary.

---

## Support

If you encounter any issues or have questions:

- **GitHub Issues:** Submit a ticket on the [GitHub Issues](https://github.com/your_username/FFT-Pattern-Suppression/issues) page.

---

Thank you for using the FFT Pattern Suppression application!
