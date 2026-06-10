# Pipecat API Documentation

This directory contains the source files for auto-generating Pipecat's API reference documentation.

## Building Documentation

From this directory:

```bash
# Build docs (warnings shown but don't fail the build)
cd docs/api && uv run ./build-docs.sh

# Build with strict mode (warnings treated as errors)
cd docs/api && uv run ./build-docs.sh --strict
```

The build script will:

1. Install documentation dependencies via `uv sync --group docs`
2. Clean previous build output
3. Run `sphinx-build` to generate HTML documentation
4. Open the result in your browser (macOS)

## Directory Structure

```
.
├── api/            # Auto-generated API documentation (created during build)
├── _build/         # Built documentation output
├── conf.py         # Sphinx configuration (mock imports, extensions, etc.)
├── index.rst       # Main documentation entry point
├── build-docs.sh   # Local build script
└── rtd-test.sh     # ReadTheDocs test build script (uses pip, not uv)
```

## How It Works

- `conf.py` runs `sphinx-apidoc` during Sphinx's `setup()` phase to generate `.rst` files from Python source
- Sphinx autodoc imports each module to extract docstrings
- Modules with unavailable dependencies are listed in `autodoc_mock_imports` in `conf.py`
- Napoleon extension converts Google-style docstrings to reStructuredText

## Troubleshooting

**Module not appearing in docs:**

1. Check the build output for `autodoc: failed to import` warnings
2. If the module has an unresolvable import dependency, add it to `autodoc_mock_imports` in `conf.py`
3. Verify the module is importable: `uv run python -c "import pipecat.module.name"`

**Duplicate object warnings:**

These come from re-export modules or Sphinx discovering the same class through multiple import paths. Usually cosmetic.

**Docstring formatting warnings:**

Docstrings use reStructuredText, not Markdown. Common issues:
- Use `Example::` with indented code blocks, not `` ```python ``
- Ensure blank lines between directive content and subsequent sections
- Use `Parameters:` (not `Attributes:`) for dataclass field documentation to avoid duplicate entries
