import builtins
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Fix Pydantic v2 + Sphinx autodoc incompatibility: ConfigDict(extra="allow") fails
# during Sphinx's import because __pydantic_extra__ annotation on BaseModel resolves to
# `Dict[str, Any] | None` whose get_origin() is Union, not dict. Patch the check to
# accept Union-wrapped dict types (i.e., Optional[Dict[str, Any]]).
import pydantic._internal._generate_schema as _pydantic_gs
from sphinx import addnodes

try:
    from sphinx.ext.autodoc._sentinels import INSTANCE_ATTR
except ImportError:
    # Private to Sphinx, and the public ``INSTANCEATTR`` alias is a different
    # object that never matches. Without it the redundant field stubs stay.
    INSTANCE_ATTR = None

_ORIG_DICT_TYPES = _pydantic_gs.DICT_TYPES
# Expand the accepted types to include Union (Optional[Dict[str, Any]])
import types
import typing

_pydantic_gs.DICT_TYPES = [*_ORIG_DICT_TYPES, typing.Union, types.UnionType]

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

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

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
    # MLX dependencies (Apple Silicon specific)
    "mlx",
    "mlx_whisper",  # Note: might need underscore format too
    # pocket-tts dependencies (torch is mocked above)
    "pocket_tts",
    # Pydantic v2 compatibility issues in third-party SDKs
    "hume",
    "hume.tts",
    "hume.tts.types",
    "cartesia",
    "camb",
    "sarvamai",
    "openai.types.beta.realtime",
    "langchain_core",
    "langchain_core.messages",
    # FastAPI - Pydantic v2 compatibility issues during Sphinx autodoc
    "fastapi",
    "fastapi.applications",
    "fastapi.routing",
    "fastapi.params",
    "fastapi.middleware",
    "fastapi.responses",
    "uvicorn",
    # Deepgram dependencies
    "deepgram",
    # Vonage Video Connector - wheels exist only for Linux + Python 3.13, so the
    # real package is never installed in a docs environment
    "vonage_video_connector",
    # FunASR - importing it executes the package's own module-level scanning,
    # which fails on newer Python versions
    "funasr",
]

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if os.path.exists("_static") else []
autodoc_typehints = "signature"  # Show type hints in the signature only, not in the docstring
html_show_sphinx = False


# An unqualified annotation resolves through the Python domain's fuzzy search,
# which scans every registered object for one whose name ends in ".<target>".
# A builtin like ``type`` or ``object`` matches the same-named field on a few
# hundred pipecat models, so the annotation links to whichever of them was
# registered first. Builtins are never pipecat attributes: without the flag that
# enables the fuzzy search the lookup misses, and intersphinx answers from the
# Python inventory instead.
_BUILTIN_NAMES = frozenset(dir(builtins))


def resolve_builtins_against_python(app, doctree):
    """Send builtin annotations to the Python inventory, not to pipecat fields."""
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("refdomain") == "py" and node.get("reftarget") in _BUILTIN_NAMES:
            node.attributes.pop("refspecific", None)


# Sphinx hands an annotation-only class attribute to the member filter with its
# skip flag already False, so ``undoc-members: False`` does not suppress it: a
# dataclass or model field renders once in the class signature, once in the
# Parameters block napoleon builds from the class docstring, and a third time as
# a bare ``name: type`` stub carrying no prose. Drop that stub where the
# docstring already describes the field, and keep it where it is the only place
# the field appears at all.
_FIELD_SECTION = re.compile(
    r"^([ \t]*)(?:Parameters|Attributes|Args):[ \t]*\n(.*?)(?=\n\1\S|\Z)", re.M | re.S
)
_FIELD_NAME = re.compile(r"^\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?:", re.M)
_documented_field_cache: dict[type, set] = {}


def documented_fields(cls):
    """Field names the class or one of its bases describes in its docstring."""
    if cls in _documented_field_cache:
        return _documented_field_cache[cls]
    names = set()
    for klass in getattr(cls, "__mro__", (cls,)):
        # __dict__ rather than __doc__ so an inherited docstring is not counted
        # twice; the MRO walk already covers the base that owns it.
        for _, block in _FIELD_SECTION.findall(klass.__dict__.get("__doc__") or ""):
            names.update(_FIELD_NAME.findall(block))
    _documented_field_cache[cls] = names
    return names


def skip_documented_field(app, what, name, obj, skip, options):
    """Skip a field stub whose description already lives in the docstring."""
    if INSTANCE_ATTR is None:
        return skip
    if skip or obj is not INSTANCE_ATTR:
        return skip
    module = sys.modules.get(app.env.temp_data.get("autodoc:module") or "")
    path = (app.env.temp_data.get("autodoc:class") or "").split(".")
    cls = getattr(module, path[0], None) if module and path[0] else None
    for part in path[1:]:
        cls = getattr(cls, part, None)
    if cls is not None and name in documented_fields(cls):
        return True
    return skip


def import_core_modules():
    """Import core pipecat modules for autodoc to discover."""
    core_modules = [
        "pipecat",
        "pipecat.adapters",
        "pipecat.audio",
        "pipecat.bus",
        "pipecat.cli",
        "pipecat.clocks",
        "pipecat.evals",
        "pipecat.extensions",
        "pipecat.flows",
        "pipecat.frames",
        "pipecat.metrics",
        "pipecat.observers",
        "pipecat.pipeline",
        "pipecat.processors",
        "pipecat.registry",
        "pipecat.runner",
        "pipecat.serializers",
        "pipecat.services",
        "pipecat.transcriptions",
        "pipecat.transports",
        "pipecat.turns",
        "pipecat.utils",
        "pipecat.workers",
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

    app.connect("doctree-read", resolve_builtins_against_python)
    if INSTANCE_ATTR is None:
        logger.warning(
            "sphinx.ext.autodoc._sentinels.INSTANCE_ATTR is gone; every documented "
            "dataclass and model field will render a second time as a bare stub. "
            "Find what replaced it and update skip_documented_field."
        )
    app.connect("autodoc-skip-member", skip_documented_field)

    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent.parent
    output_dir = str(docs_dir / "api")
    source_dir = str(project_root / "src" / "pipecat")

    # Clean existing files
    if Path(output_dir).exists():
        import shutil

        shutil.rmtree(output_dir)
        logger.info(f"Cleaned existing documentation in {output_dir}")

    logger.info("Generating API documentation...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Source directory: {source_dir}")

    excludes = [
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
