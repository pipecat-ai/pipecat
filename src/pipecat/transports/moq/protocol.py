#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) protocol types for the Pipecat MOQ transport.

The actual MOQ wire protocol is handled by the upstream `moq` Python
library (see https://pypi.org/project/moq-rs/). This module just defines
a small set of enums and dataclasses that the transport surfaces in its
event callbacks.
"""

from dataclasses import dataclass
from enum import IntEnum


class MOQRole(IntEnum):
    """MOQ endpoint roles.

    The `moq` library handles pub/sub negotiation automatically; this enum
    is kept for backwards compatibility with existing example code that
    sets ``role=MOQRole.PUBSUB`` when constructing :class:`MOQParams`.
    """

    PUBLISHER = 0x01
    SUBSCRIBER = 0x02
    PUBSUB = 0x03


class MOQTrackType(IntEnum):
    """Types of media tracks in MOQ."""

    AUDIO = 0x01
    VIDEO = 0x02
    DATA = 0x03


@dataclass
class MOQTrack:
    """Identifies a MOQ track for event callbacks.

    Parameters:
        broadcast_path: The full broadcast path (e.g. ``pipecat/bot0``).
        name: The track name (e.g. ``bot-audio``).
        track_type: The track media type.
    """

    broadcast_path: str
    name: str
    track_type: MOQTrackType = MOQTrackType.DATA

    @property
    def full_name(self) -> str:
        """Get the full track identifier."""
        return f"{self.broadcast_path}/{self.name}"
