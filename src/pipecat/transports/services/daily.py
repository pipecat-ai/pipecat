#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily transport implementation for Pipecat.

This module provides comprehensive Daily video conferencing integration including
audio/video streaming, transcription, recording, dial-in/out functionality, and
real-time communication features.
"""

import warnings

from pipecat.transports.daily.transport import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.transports.services.daily` is deprecated, "
        "use `pipecat.transports.daily.transport` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
