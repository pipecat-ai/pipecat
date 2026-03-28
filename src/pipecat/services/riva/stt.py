#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Riva Speech-to-Text service implementations for real-time and batch transcription.

.. deprecated:: 0.0.96
    This module is deprecated. Please NvidiaSTTService from
    pipecat.services.nvidia.stt instead.
"""

import warnings

from pipecat.services.nvidia.stt import (
    NvidiaSegmentedSTTService,
    NvidiaSTTService,
    language_to_nvidia_riva_language,
)

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "RivaSTTService and ParakeetSTTService "
        "from pipecat.services.riva.stt is deprecated. "
        "Please use NvidiaSTTService from pipecat.services.nvidia.stt instead.",
        DeprecationWarning,
        stacklevel=2,
    )

RivaSTTService = NvidiaSTTService
language_to_riva_language = language_to_nvidia_riva_language
RivaSegmentedSTTService = NvidiaSegmentedSTTService
ParakeetSTTService = NvidiaSTTService
