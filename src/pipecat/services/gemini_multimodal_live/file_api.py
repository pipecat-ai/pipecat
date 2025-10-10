#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini File API client for uploading and managing files.

This module provides a client for Google's Gemini File API, enabling file
uploads, metadata retrieval, listing, and deletion. Files uploaded through
this API can be referenced in Gemini generative model calls.

.. deprecated:: 0.0.90
    Importing GeminiFileAPI from this module is deprecated.
    Import it from pipecat.services.google.gemini_live.file_api instead.
"""

import warnings

from loguru import logger

try:
    from pipecat.services.google.gemini_live.file_api import GeminiFileAPI as _GeminiFileAPI
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")

# These aliases are just here for backward compatibility, since we used to
# define public-facing StartSensitivity and EndSensitivity enums in this
# module.
warnings.warn(
    "Importing GeminiFileAPI from "
    "pipecat.services.gemini_multimodal_live.file_api is deprecated. "
    "Please import it from pipecat.services.google.gemini_live.file_api instead.",
    DeprecationWarning,
    stacklevel=2,
)
GeminiFileAPI = _GeminiFileAPI
