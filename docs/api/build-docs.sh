#!/bin/bash

# Check if sphinx-build is installed
if ! command -v sphinx-build &> /dev/null; then
    echo "Error: sphinx-build is not installed or not in PATH" >&2
    echo "Please install Sphinx using: pip install -r requirements.txt" >&2
    exit 1
fi

# Clean previous build
rm -rf _build

# Build docs matching ReadTheDocs configuration
sphinx-build -b html -d _build/doctrees . _build/html -W --keep-going

# Open docs (MacOS)
open _build/html/index.html