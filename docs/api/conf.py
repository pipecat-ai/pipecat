import sys
from pathlib import Path

# Add source directory to path
docs_dir = Path(__file__).parent
project_root = docs_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Project information
project = "pipecat-ai"
copyright = "2024, Daily"
author = "Daily"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# AutoDoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "no-index": True,
}

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_typehints = "description"
html_show_sphinx = False  # Remove "Built with Sphinx"


def setup(app):
    """Generate API documentation during Sphinx build."""
    from sphinx.ext.apidoc import main

    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent.parent
    output_dir = str(docs_dir / "api")
    source_dir = str(project_root / "src" / "pipecat")

    # Clean existing files
    if Path(output_dir).exists():
        import shutil

        shutil.rmtree(output_dir)

    print(f"Generating API documentation...")
    print(f"Output directory: {output_dir}")
    print(f"Source directory: {source_dir}")

    # Similar exclusions as in your generate_docs.py
    excludes = [
        str(project_root / "src/pipecat/processors/gstreamer"),
        str(project_root / "src/pipecat/transports/network"),
        str(project_root / "src/pipecat/transports/services"),
        str(project_root / "src/pipecat/transports/local"),
        str(project_root / "src/pipecat/services/to_be_updated"),
        "**/test_*.py",
        "**/tests/*.py",
    ]

    try:
        main(["-f", "-e", "-M", "--no-toc", "-o", output_dir, source_dir] + excludes)
        print("API documentation generated successfully!")
    except Exception as e:
        print(f"Error generating API documentation: {e}")
