#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Small WebRTC transport implementation for Pipecat.

This module provides a WebRTC transport implementation using aiortc for
real-time audio and video communication. It supports bidirectional media
streaming, application messaging, and client connection management.
"""

import warnings

from pipecat.transports.smallwebrtc.transport import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.transports.network.small_webrtc` is deprecated, "
        "use `pipecat.transports.smallwebrtc.transport` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
