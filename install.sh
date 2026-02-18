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

# Create virtual environment with system site-packages
# This gives access to system-installed packages (pyAgrum, openturns)
# and user-installed packages (otagrum in ~/.local/)
echo "Creating virtual environment with --system-site-packages..."
python3 -m venv venv --system-site-packages
source venv/bin/activate
echo "[OK] Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install base dependencies from pyproject.toml
echo "============================================"
echo "Installing base dependencies..."
echo "============================================"
pip install -e .
echo "[OK] Base dependencies installed"
echo ""

# Ask about otagrum installation method
echo "============================================"
echo "otagrum Installation"
echo "============================================"
echo "Choose your otagrum installation method:"
echo "  1) Standard otagrum via conda (no CPC2/CMIIC2 support)"
echo "  2) Build from source with CPC2/CMIIC2 support (requires cmake & compiler)"
echo "  3) Skip otagrum installation"
read -p "Enter choice [1/2/3]: " otagrum_choice

case $otagrum_choice in
    1)
        echo "Installing standard otagrum via conda..."
        if command -v conda &> /dev/null; then
            conda install otagrum -y
            echo "[OK] otagrum installed via conda"
        else
            echo "[ERROR] conda not found. Please install conda or choose option 2."
            exit 1
        fi
        ;;
    2)
        echo "Building otagrum from source..."

        # Check for cmake
        if ! command -v cmake &> /dev/null; then
            echo "[ERROR] cmake not found. Please install cmake first:"
            echo "   Ubuntu/Debian: sudo apt install cmake build-essential"
            echo "   Fedora: sudo dnf install cmake gcc-c++"
            echo "   macOS: brew install cmake"
            exit 1
        fi

        echo "[OK] cmake found: $(cmake --version | head -n1)"

        # Clone otagrum
        if [ -d "otagrum_build" ]; then
            echo "[WARN] otagrum_build directory exists. Removing..."
            rm -rf otagrum_build
        fi

        echo "Cloning otagrum repository..."
        git clone https://github.com/mathisemb/otagrum.git otagrum_build
        cd otagrum_build
        git checkout cpc2_cmiic2

        # Build
        echo "Building otagrum (this may take a few minutes)..."
        mkdir -p build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
        cmake --build .
        cmake --build . --target install

        cd ../..
        echo "[OK] otagrum built and installed"

        # Optional: run tests
        read -p "Run otagrum unit tests? [y/N]: " run_tests
        if [[ $run_tests =~ ^[Yy]$ ]]; then
            cd otagrum_build/build
            cmake --build . --target cppcheck
            ctest
            cd ../..
        fi

        # Cleanup
        read -p "Remove build directory? [y/N]: " cleanup
        if [[ $cleanup =~ ^[Yy]$ ]]; then
            rm -rf otagrum_build
            echo "[OK] Build directory removed"
        fi
        ;;
    3)
        echo "Skipping otagrum installation"
        ;;
    *)
        echo "[ERROR] Invalid choice"
        exit 1
        ;;
esac
echo ""

# Install NOTEARS (not on PyPI, must be installed from git)
# LiNGAM is already installed by pip install -e . (via pyproject.toml)
echo "============================================"
echo "NOTEARS Installation"
echo "============================================"
echo "Installing NOTEARS..."
pip install git+https://github.com/xunzheng/notears.git
echo "[OK] NOTEARS installed"
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
echo "To test the installation, run:"
echo "  python tests/test_comparison_sachs.py"
echo ""
echo "To activate the virtual environment later:"
echo "  source venv/bin/activate"
echo ""

echo "For more information, see the README.md file."
