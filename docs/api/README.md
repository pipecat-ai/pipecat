# Pipecat Documentation

This directory contains the source files for auto-generating Pipecat's server API reference documentation.

## Setup

1. Install documentation dependencies:

```bash
pip install -r requirements.txt
```

2. Make the build script executable:

```bash
chmod +x build-docs.sh
```

## Building Documentation

From this directory, run either:

```bash
# Using the build script (automatically opens docs when done)
./build-docs.sh

# Or directly with sphinx-build
sphinx-build -b html . _build/html -W --keep-going
```

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
├── requirements.txt # Documentation dependencies
└── build-docs.sh   # Build script matching ReadTheDocs configuration
```

## Notes

- Documentation is auto-generated from Python docstrings
- Service modules are automatically detected and included
- The build process matches our ReadTheDocs configuration
- Warnings are treated as errors (-W flag) to maintain consistency
- The --keep-going flag ensures all errors are reported

## Troubleshooting

If you encounter missing service modules:

1. Verify the service is installed with its extras: `pip install pipecat-ai[service-name]`
2. Check the build logs for import errors
3. Ensure the service module is properly initialized in the package

For more information:

- [ReadTheDocs Configuration](.readthedocs.yaml)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
