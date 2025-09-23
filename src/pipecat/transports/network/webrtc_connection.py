#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Small WebRTC connection implementation for Pipecat.

This module provides a WebRTC connection implementation using aiortc,
with support for audio/video tracks, data channels, and signaling
for real-time communication applications.
"""

import warnings

from pipecat.transports.smallwebrtc.connection import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.transports.network.webrtc_connection` is deprecated, "
        "use `pipecat.transports.smallwebrtc.connection` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
