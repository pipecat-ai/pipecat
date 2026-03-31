#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime LLM service.

.. deprecated:: 0.0.108
    This module is deprecated. Please use GrokRealtimeLLMService from
    pipecat.services.xai.realtime.llm instead.
"""

import warnings

from pipecat.services.xai.realtime.llm import *  # noqa: F401,F403

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "pipecat.services.grok.realtime.llm is deprecated. "
        "Please use pipecat.services.xai.realtime.llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
