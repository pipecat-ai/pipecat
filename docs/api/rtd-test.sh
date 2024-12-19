#!/bin/bash
set -e

# Configuration
DOCS_DIR=$(pwd)
PROJECT_ROOT=$(cd ../../ && pwd)
TEST_DIR="/tmp/rtd-test-$(date +%Y%m%d_%H%M%S)"

echo "Creating test directory: $TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create virtual environment
python -m venv venv
source venv/bin/activate

echo "Installing build dependencies..."
pip install --upgrade pip wheel setuptools

echo "Installing documentation dependencies..."
pip install -r "$DOCS_DIR/requirements.txt"

echo "Building documentation..."
cd "$DOCS_DIR"
sphinx-build -b html . "_build/html"

echo "Build complete. Check _build/html directory for output."

# Print summary
echo -e "\n=== Build Summary ==="
echo "Documentation: $DOCS_DIR/_build/html"
echo "Test environment: $TEST_DIR"
echo -e "\nTo view the documentation:"
echo "open $DOCS_DIR/_build/html/index.html"

# Print installed packages for verification
echo -e "\n=== Installed Packages ==="
pip freeze | grep -E "sphinx|pipecat"