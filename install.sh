#!/bin/bash
# Automated installation script for cbnsl_benchmark

set -e  # Exit on error

echo "============================================"
echo "cbnsl_benchmark Installation"
echo "============================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed. Please install it first."
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "[OK] Virtual environment created and activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install Python dependencies
echo "============================================"
echo "Installing Python dependencies..."
echo "============================================"
pip install -e .
pip install git+https://github.com/xunzheng/notears.git
echo "[OK] Python dependencies installed"
echo ""

# Install otagrum from source
echo "============================================"
echo "otagrum Installation (from source)"
echo "============================================"
echo "This benchmark requires the latest version of otagrum."
echo "It must be built from source (requires cmake and a C++ compiler)."
echo ""

if ! command -v cmake &> /dev/null; then
    echo "[ERROR] cmake not found. Please install cmake first:"
    echo "   Ubuntu/Debian: sudo apt install cmake build-essential"
    echo "   Arch/Manjaro:  sudo pacman -S cmake base-devel"
    echo "   Fedora:        sudo dnf install cmake gcc-c++"
    echo "   macOS:         brew install cmake"
    exit 1
fi

echo "[OK] cmake found: $(cmake --version | head -n1)"
echo ""

echo "Cloning otagrum repository..."
git clone https://github.com/openturns/otagrum.git otagrum_build
cd otagrum_build
mkdir build && cd build

echo "Building otagrum (this may take a few minutes)..."
cmake .. -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV
cmake --build .
cmake --build . --target install
cd ../..

echo "[OK] otagrum installed in the virtual environment"
echo ""

# Cleanup
read -p "Remove otagrum build directory? [y/N]: " cleanup
if [[ $cleanup =~ ^[Yy]$ ]]; then
    rm -rf otagrum_build
    echo "[OK] Build directory removed"
fi
echo ""

# Optional: install dev dependencies
read -p "Install development dependencies (pytest, black)? [y/N]: " install_dev
if [[ $install_dev =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
    echo "[OK] Development dependencies installed"
fi
echo ""

# Final summary
echo "============================================"
echo "Installation completed successfully!"
echo "============================================"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the benchmark:"
echo "  python views/run_all_benchmarks.py"
echo ""
