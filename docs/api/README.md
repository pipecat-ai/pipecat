# Pipecat Documentation

This directory contains the source files for auto-generating Pipecat's server API reference documentation.

## Setup

1. Install documentation dependencies:

```bash
pip install -r requirements.txt
```

2. Make the build scripts executable:

```bash
chmod +x build-docs.sh rtd-test.py
```

## Building Documentation

From this directory, you can build the documentation in several ways:

### Local Build

```bash
# Using the build script (automatically opens docs when done)
./build-docs.sh

# Or directly with sphinx-build
sphinx-build -b html . _build/html -W --keep-going
```

### ReadTheDocs Test Build

To test the documentation build process exactly as it would run on ReadTheDocs:

```bash
./rtd-test.py
```

This script:

- Creates a fresh virtual environment
- Installs all dependencies as specified in requirements files
- Handles conflicting dependencies (like grpcio versions for Riva and PlayHT)
- Builds the documentation in an isolated environment
- Provides detailed logging of the build process

Use this script to verify your documentation will build correctly on ReadTheDocs before pushing changes.

## Viewing Documentation

The built documentation will be available at `_build/html/index.html`. To open:

```bash
# On MacOS
open _build/html/index.html

# On Linux
xdg-open _build/html/index.html

# On Windows
start _build/html/index.html
```

## Directory Structure

```
.
├── api/            # Auto-generated API documentation
├── _build/         # Built documentation
├── _static/        # Static files (images, css, etc.)
├── conf.py         # Sphinx configuration
├── index.rst       # Main documentation entry point
├── requirements-base.txt    # Base documentation dependencies
├── requirements-riva.txt    # Riva-specific dependencies
├── requirements-playht.txt  # PlayHT-specific dependencies
├── build-docs.sh   # Local build script
└── rtd-test.py     # ReadTheDocs test build script
```

## Notes

- Documentation is auto-generated from Python docstrings
- Service modules are automatically detected and included
- The build process matches our ReadTheDocs configuration
- Warnings are treated as errors (-W flag) to maintain consistency
- The --keep-going flag ensures all errors are reported
- Dependencies are split into multiple requirements files to handle version conflicts

## Troubleshooting

If you encounter missing service modules:

1. Verify the service is installed with its extras: `pip install pipecat-ai[service-name]`
2. Check the build logs for import errors
3. Ensure the service module is properly initialized in the package
4. Run `./rtd-test.py` to test in an isolated environment matching ReadTheDocs

For dependency conflicts:

1. Check the requirements files for version specifications
2. Use `rtd-test.py` to verify dependency resolution
3. Consider adding service-specific requirements files if needed

For more information:

- [ReadTheDocs Configuration](.readthedocs.yaml)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
