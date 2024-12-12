import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sphinx-build")

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
    "sphinx.ext.coverage",
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
    "show-inheritance": True,
}

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_typehints = "description"
html_show_sphinx = False


def get_installed_services() -> Dict[str, Optional[str]]:
    """Scan for installed pipecat services and return their status.
    Returns a dictionary of service names and their import status/error message.
    """
    services_dir = project_root / "src" / "pipecat" / "services"
    services_status = {}

    if not services_dir.exists():
        logger.warning(f"Services directory not found: {services_dir}")
        return services_status

    for item in services_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_") and not item.name == "to_be_updated":
            service_name = item.name
            try:
                module = importlib.import_module(f"pipecat.services.{service_name}")
                services_status[service_name] = None  # None indicates success
                logger.info(f"Found service: {service_name} at {module.__file__}")
            except ImportError as e:
                services_status[service_name] = str(e)
                logger.warning(f"Failed to import {service_name}: {e}")

    return services_status


def generate_services_rst() -> str:
    """Generate RST content for services section."""
    services = get_installed_services()

    # Sort services into successful and failed imports
    successful = [name for name, status in services.items() if status is None]
    failed = [(name, status) for name, status in services.items() if status is not None]

    rst_content = [
        "Services",
        "~~~~~~~~",
        "",
        "Successfully Detected Services:",
        "",
    ]

    for service in sorted(successful):
        rst_content.append(f"* :mod:`pipecat.services.{service}`")

    if failed:
        rst_content.extend(
            [
                "",
                "Services with Import Issues:",
                "",
            ]
        )
        for service, error in sorted(failed):
            rst_content.append(f"* {service} (Import failed: {error})")

    return "\n".join(rst_content)


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
        logger.info(f"Cleaned existing documentation in {output_dir}")

    logger.info(f"Generating API documentation...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Source directory: {source_dir}")

    # Get installed services
    services = get_installed_services()
    logger.info(f"Found {len(services)} services")
    for service, status in services.items():
        if status is None:
            logger.info(f"Service available: {service}")
        else:
            logger.warning(f"Service import failed: {service} - {status}")

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
        main(
            [
                "-f",
                "-e",
                "-M",
                "--no-toc",
                "--separate",
                "--module-first",
                "-o",
                output_dir,
                source_dir,
            ]
            + excludes
        )

        logger.info("API documentation generated successfully!")

        # Generate services index file
        services_index = Path(output_dir) / "services_index.rst"
        services_index.write_text(generate_services_rst())
        logger.info(f"Generated services index at {services_index}")

    except Exception as e:
        logger.error(f"Error generating API documentation: {e}", exc_info=True)
