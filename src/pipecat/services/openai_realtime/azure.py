#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure OpenAI Realtime LLM service implementation."""

import warnings

from pipecat.services.azure.realtime.llm import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.openai_realtime.azure are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.azure.realtime.llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
