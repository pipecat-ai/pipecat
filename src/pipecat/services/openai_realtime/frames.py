#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom frame types for OpenAI Realtime API integration."""

import warnings

from pipecat.services.openai.realtime.frames import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.openai_realtime.frames are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.openai.realtime.frames instead.",
        DeprecationWarning,
        stacklevel=2,
    )
