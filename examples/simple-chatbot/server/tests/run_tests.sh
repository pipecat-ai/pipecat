#!/bin/bash

# Change to the server directory
cd "$(dirname "$0")/.."

# Install test dependencies if needed
pip install -r tests/requirements-test.txt

# Set up Python path to find the mock modules
export PYTHONPATH="$PYTHONPATH:$(pwd)/tests"

# Run all tests
echo "Running all tests..."
python -m pytest tests/ -v

# Verify that latest recording tests are specifically run
echo ""
echo "Verifying latest recording tests specifically..."
python -m pytest tests/test_latest_recording.py -v 