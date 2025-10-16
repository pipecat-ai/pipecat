#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime LLM context and aggregator implementations."""

import warnings

from pipecat.services.openai.realtime.context import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.openai_realtime.context are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.openai.realtime.context instead.",
        DeprecationWarning,
        stacklevel=2,
    )
