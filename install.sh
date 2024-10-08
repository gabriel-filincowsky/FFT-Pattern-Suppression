#!/bin/bash

echo "Installing FFT Pattern Suppression Application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3.6 or higher."
    exit
fi

# Check if Git is installed
if ! command -v git &> /dev/null
then
    echo "Git is not installed. Please install Git."
    exit
fi

# Prompt user for CUDA and CuPy installation
read -p "Do you have CUDA and wish to install CuPy for GPU acceleration? (y/n): " cuda_choice
if [[ "$cuda_choice" == "y" || "$cuda_choice" == "Y" ]]; then
    INSTALL_CUPY=1
else
    INSTALL_CUPY=0
fi

# Clone the repository
if [ ! -d "FFT-Pattern-Suppression" ]; then
    git clone https://github.com/gabriel-filincowsky/FFT-Pattern-Suppression.git
fi
cd FFT-Pattern-Suppression

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
if [ $INSTALL_CUPY -eq 1 ]; then
    pip install cupy-cuda118  # Replace '118' with your CUDA version
    pip install -r requirements.txt
else
    pip install -r requirements_cpu.txt
fi

# Run the application
python3 main.py
