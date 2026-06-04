#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus message serialization for network transport.

Provides the abstract `MessageSerializer` interface and a default
`JSONMessageSerializer` implementation.
"""

from pipecat.bus.serializers.base import MessageSerializer
from pipecat.bus.serializers.json import JSONMessageSerializer

__all__ = [
    "JSONMessageSerializer",
    "MessageSerializer",
]
