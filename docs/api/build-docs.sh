#!/bin/bash

# Clean previous build
rm -rf _build

# Build docs matching ReadTheDocs configuration
sphinx-build -b html -d _build/doctrees . _build/html -W --keep-going

# Open docs (MacOS)
open _build/html/index.html