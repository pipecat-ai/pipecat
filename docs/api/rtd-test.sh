#!/bin/bash
set -e

# Configuration
DOCS_DIR=$(pwd)
PROJECT_ROOT=$(cd ../../ && pwd)
TEST_DIR="/tmp/rtd-test-$(date +%Y%m%d_%H%M%S)"

echo "Creating test directory: $TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create single virtual environment
python -m venv venv
source venv/bin/activate

echo "Installing base dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r "$DOCS_DIR/requirements-base.txt"

# Try to install optional dependencies, but don't fail if they don't work
echo "Installing Riva dependencies..."
pip install -r "$DOCS_DIR/requirements-riva.txt" || echo "Failed to install Riva dependencies"

echo "Installing PlayHT dependencies..."
pip install -r "$DOCS_DIR/requirements-playht.txt" || echo "Failed to install PlayHT dependencies"

echo "Building documentation..."
cd "$DOCS_DIR"
sphinx-build -b html . "_build/html"

echo "Build complete. Check _build/html directory for output."

# Print installed packages for verification
echo "Installed packages:"
pip freeze