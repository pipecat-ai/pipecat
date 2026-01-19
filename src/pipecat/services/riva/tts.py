#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Riva text-to-speech service implementation.

This module provides integration with NVIDIA Riva's TTS services through
gRPC API for high-quality speech synthesis.

.. deprecated:: 0.0.96
    This module is deprecated. Please NvidiaTTSService from
    pipecat.services.nvidia.tts instead.
"""

import warnings

from pipecat.services.nvidia.tts import NVIDIA_TTS_TIMEOUT_SECS, NvidiaTTSService

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "FastPitchTTSService and RivaTTSService "
        "from pipecat.services.nim.llm are deprecated. "
        "Please use NvidiaLLMService from pipecat.services.nvidia.tts instead.",
        DeprecationWarning,
        stacklevel=2,
    )

RivaTTSService = NvidiaTTSService
FastPitchTTSService = NvidiaTTSService
RIVA_TTS_TIMEOUT_SECS = NVIDIA_TTS_TIMEOUT_SECS
