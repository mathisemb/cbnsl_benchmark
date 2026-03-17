#!/bin/bash
# Automated installation script for cbnsl_benchmark

set -e  # Exit on error

echo "============================================"
echo "cbnsl_benchmark Installation"
echo "============================================"
echo ""

# Check conda installation
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda is not installed. Please install miniforge or miniconda first."
    exit 1
fi

echo "[OK] conda found: $(conda --version)"
echo ""

# Create conda environment
echo "Creating conda environment 'cbnsl'..."
conda create -n cbnsl -y
eval "$(conda shell.bash hook)"
conda activate cbnsl
echo "[OK] Environment created and activated"
echo ""

# Install otagrum (pulls pyAgrum and OpenTURNS as dependencies)
echo "============================================"
echo "Installing otagrum (+ pyAgrum, OpenTURNS)..."
echo "============================================"
conda install otagrum -y
echo "[OK] otagrum installed"
echo ""

# Install Python dependencies
echo "============================================"
echo "Installing Python dependencies..."
echo "============================================"
pip install -e .
pip install git+https://github.com/xunzheng/notears.git
echo "[OK] Python dependencies installed"
echo ""

# Final summary
echo "============================================"
echo "Installation completed successfully!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate cbnsl"
echo ""
echo "To run the benchmark:"
echo "  python views/run_all_benchmarks.py"
echo ""
