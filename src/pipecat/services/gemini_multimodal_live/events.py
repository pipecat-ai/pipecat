#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event models and utilities for Google Gemini Multimodal Live API.

.. deprecated:: 0.0.90
    Importing StartSensitivity and EndSensitivity from this module is deprecated.
    Import them directly from google.genai.types instead.
"""

import warnings

from loguru import logger

try:
    from google.genai.types import (
        EndSensitivity as _EndSensitivity,
    )
    from google.genai.types import (
        StartSensitivity as _StartSensitivity,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")

# These aliases are just here for backward compatibility, since we used to
# define public-facing StartSensitivity and EndSensitivity enums in this
# module.
with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Importing StartSensitivity and EndSensitivity from "
        "pipecat.services.gemini_multimodal_live.events is deprecated. "
        "Please import them directly from google.genai.types instead.",
        DeprecationWarning,
        stacklevel=2,
    )

StartSensitivity = _StartSensitivity
EndSensitivity = _EndSensitivity
