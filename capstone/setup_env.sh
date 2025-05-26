#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create new conda environment with Python 3.9
echo "Creating new conda environment 'ur_voice_control' with Python 3.9..."
conda create -y -n ur_voice_control python=3.9

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ur_voice_control

# Install required system packages (for Ubuntu/Debian)
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Additional dependencies that might be needed
pip install setuptools wheel

echo "Setup complete! To activate the environment, run:"
echo "conda activate ur_voice_control" 