#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic LLM service implementation for Pipecat AI framework.

This module provides a speech-to-speech LLM service using AWS Nova Sonic, which supports
bidirectional audio streaming, text generation, and function calling capabilities.
"""

import warnings

from pipecat.services.aws.nova_sonic.llm import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.aws_nova_sonic.aws are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.aws.nova_sonic.llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
