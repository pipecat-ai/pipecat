#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit transport implementation for Pipecat.

This module provides comprehensive LiveKit real-time communication integration
including audio streaming, data messaging, participant management, and room
event handling for conversational AI applications.
"""

import warnings

from pipecat.transports.livekit.transport import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.transports.services.livekit` is deprecated, "
        "use `pipecat.transports.livekit.transport` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
