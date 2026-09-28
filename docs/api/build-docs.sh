#!/bin/bash

# Usage: ./build-docs.sh [--strict]
#   --strict: Treat warnings as errors (default: warnings only)

SPHINX_OPTS=""
if [ "$1" = "--strict" ]; then
    SPHINX_OPTS="-W --keep-going"
fi

# Build docs using uv
echo "Installing dependencies with uv..."
"$(dirname "$0")/install-deps.sh"

# Check if sphinx-build is available
if ! uv run sphinx-build --version &> /dev/null; then
    echo "Error: sphinx-build is not available" >&2
    exit 1
fi

# Clean previous build
rm -rf _build

echo "Building documentation..."
uv run sphinx-build -b html -d _build/doctrees . _build/html $SPHINX_OPTS

if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    # Open docs (MacOS)
    open _build/html/index.html
else
    echo "Documentation build failed!" >&2
    exit 1
fi
