import logging
import sys
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
    "show-inheritance": True,
}

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_typehints = "description"
html_show_sphinx = False


def clean_title(title: str) -> str:
    """Automatically clean module titles."""
    # Remove everything after space (like 'module', 'processor', etc.)
    title = title.split(" ")[0]

    # Get the last part of the dot-separated path
    parts = title.split(".")
    title = parts[-1]

    # Special cases for service names and common acronyms
    special_cases = {
        "ai": "AI",
        "aws": "AWS",
        "api": "API",
        "vad": "VAD",
        "assemblyai": "AssemblyAI",
        "deepgram": "Deepgram",
        "elevenlabs": "ElevenLabs",
        "openai": "OpenAI",
        "openpipe": "OpenPipe",
        "playht": "PlayHT",
        "xtts": "XTTS",
        "lmnt": "LMNT",
    }

    # Check if the entire title is a special case
    if title.lower() in special_cases:
        return special_cases[title.lower()]

    # Otherwise, capitalize each word
    words = title.split("_")
    cleaned_words = []
    for word in words:
        if word.lower() in special_cases:
            cleaned_words.append(special_cases[word.lower()])
        else:
            cleaned_words.append(word.capitalize())

    return " ".join(cleaned_words)


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

        # Process generated RST files to update titles
        for rst_file in Path(output_dir).glob("*.rst"):
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
