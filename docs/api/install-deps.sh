#!/bin/bash
#
# Installs the documentation build environment.
#
# build-docs.sh and .readthedocs.yaml both call this, so a local build and a Read
# the Docs build document the same set of modules. The excluded extras are heavy
# ML or platform-specific packages that conf.py already lists in
# autodoc_mock_imports, so installing them would not change the output.

set -e

uv sync --group docs --all-extras \
    --no-extra gstreamer \
    --no-extra local-smart-turn \
    --no-extra mlx-whisper \
    --no-extra moondream \
    --no-extra pocket-tts
