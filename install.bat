@echo off
title FFT Pattern Suppression Installation
echo Installing FFT Pattern Suppression Application...

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python 3.6 or higher from https://www.python.org/downloads/.
    pause
    exit /b
)

:: Check if Git is installed
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Please install Git from https://git-scm.com/downloads.
    pause
    exit /b
)

:: Prompt user for CUDA and CuPy installation
echo Do you have CUDA and wish to install CuPy for GPU acceleration? (y/n)
set /p cuda_choice=
if /i "%cuda_choice%"=="y" (
    set INSTALL_CUPY=1
) else (
    set INSTALL_CUPY=0
)

:: Clone the repository
if not exist FFT-Pattern-Suppression (
    git clone https://github.com/your_username/FFT-Pattern-Suppression.git
)
cd FFT-Pattern-Suppression

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
pip install --upgrade pip

:: Install dependencies
if %INSTALL_CUPY%==1 (
    pip install cupy-cuda118  # Replace '118' with your CUDA version
    pip install -r requirements.txt
) else (
    pip install -r requirements_cpu.txt
)

:: Run the application
python fft_pattern_suppression_app.py

pause
