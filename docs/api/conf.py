import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sphinx-build")

# Add source directory to path
docs_dir = Path(__file__).parent
project_root = docs_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Project information
project = "pipecat-ai"
current_year = datetime.now().year
copyright = f"2024-{current_year}, Daily" if current_year > 2024 else "2024, Daily"
author = "Daily"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

suppress_warnings = [
    "autodoc.mocked_object",
    "toc.not_included",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# AutoDoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "exclude-members": "__weakref__,model_config",
    "show-inheritance": True,
}

# Mock imports for optional dependencies
autodoc_mock_imports = [
    # Krisp - has build issues on some platforms
    "pipecat_ai_krisp",
    "krisp",
    "krisp_audio",
    # System-specific GUI libraries
    "_tkinter",
    "tkinter",
    # Platform-specific audio libraries (if needed)
    "gi",
    "gi.require_version",
    "gi.repository",
    # OpenCV - sometimes has import issues during docs build
    "cv2",
    # Heavy ML packages excluded from ReadTheDocs
    # ultravox dependencies
    "vllm",
    "vllm.engine.arg_utils",
    # local-smart-turn dependencies
    "coremltools",
    "coremltools.models",
    "coremltools.models.MLModel",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchaudio",
    # moondream dependencies
    "transformers",
    "transformers.AutoTokenizer",
    "transformers.AutoFeatureExtractor",
    "AutoFeatureExtractor",
    "timm",
    "einops",
    "intel_extension_for_pytorch",
    "huggingface_hub",
    # riva dependencies
    "riva",
    "riva.client",
    "riva.client.Auth",
    "riva.client.ASRService",
    "riva.client.StreamingRecognitionConfig",
    "riva.client.RecognitionConfig",
    "riva.client.AudioEncoding",
    "riva.client.proto.riva_tts_pb2",
    "riva.client.SpeechSynthesisService",
    # MLX dependencies (Apple Silicon specific)
    "mlx",
    "mlx_whisper",  # Note: might need underscore format too
]

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if os.path.exists("_static") else []
autodoc_typehints = "signature"  # Show type hints in the signature only, not in the docstring
html_show_sphinx = False


def import_core_modules():
    """Import core pipecat modules for autodoc to discover."""
    core_modules = [
        "pipecat",
        "pipecat.frames",
        "pipecat.pipeline",
        "pipecat.processors",
        "pipecat.services",
        "pipecat.transports",
        "pipecat.audio",
        "pipecat.adapters",
        "pipecat.clocks",
        "pipecat.metrics",
        "pipecat.observers",
        "pipecat.runner",
        "pipecat.serializers",
        "pipecat.sync",
        "pipecat.transcriptions",
        "pipecat.utils",
    ]

    for module_name in core_modules:
        try:
            __import__(module_name)
            logger.info(f"Successfully imported {module_name}")
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")


def clean_title(title: str) -> str:
    """Automatically clean module titles."""
    # Remove everything after space (like 'module', 'processor', etc.)
    title = title.split(" ")[0]

    # Get the last part of the dot-separated path
    parts = title.split(".")
    title = parts[-1]

    return title


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

    excludes = [
        str(project_root / "src/pipecat/pipeline/to_be_updated"),
        str(project_root / "src/pipecat/examples"),
        str(project_root / "src/pipecat/tests"),
        "**/test_*.py",
        "**/tests/*.py",
    ]

    try:
        main(
            [
                "-f",  # Force overwriting
                "-e",  # Don't generate empty files
                "-M",  # Put module documentation before submodule documentation
                "--no-toc",  # Don't create a table of contents file
                "--separate",  # Put documentation for each module in its own page
                "--module-first",  # Module documentation before submodule documentation
                "--implicit-namespaces",  # Added: Handle implicit namespace packages
                "-o",
                output_dir,
                source_dir,
            ]
            + excludes
        )

        logger.info("API documentation generated successfully!")

        # Process generated RST files to update titles
        for rst_file in Path(output_dir).glob("**/*.rst"):  # Changed to recursive glob
            content = rst_file.read_text()
            lines = content.split("\n")

            # Find and clean up the title
            if lines and "=" in lines[1]:  # Title is typically the first line
                old_title = lines[0]
                new_title = clean_title(old_title)
                content = content.replace(old_title, new_title)
                rst_file.write_text(content)
                logger.info(f"Updated title: {old_title} -> {new_title}")

    except Exception as e:
        logger.error(f"Error generating API documentation: {e}", exc_info=True)


import_core_modules()
