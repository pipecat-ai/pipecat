#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport implementation for Pipecat.

This module provides WebSocket server transport functionality for real-time
audio and data streaming, including client connection management, session
handling, and frame serialization.
"""

import warnings

from pipecat.transports.websocket.server import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.transports.network.websocket_server` is deprecated, "
        "use `pipecat.transports.websocket.server` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
