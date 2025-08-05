#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Runner session argument types for the development runner.

These types are used by the development runner to pass transport-specific
information to bot functions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket


@dataclass
class RunnerArguments:
    """Base class for runner session arguments."""

    handle_sigint: bool = field(init=False)
    handle_sigterm: bool = field(init=False)

    def __post_init__(self):
        self.handle_sigint = True
        self.handle_sigterm = False


@dataclass
class DailyRunnerArguments(RunnerArguments):
    """Daily transport session arguments for the runner.

    Parameters:
        room_url: Daily room URL to join
        token: Authentication token for the room
        body: Additional request data
    """

    room_url: str
    token: Optional[str] = None
    body: Optional[Any] = {}


@dataclass
class WebSocketRunnerArguments(RunnerArguments):
    """WebSocket transport session arguments for the runner.

    Parameters:
        websocket: WebSocket connection for audio streaming
    """

    websocket: WebSocket


@dataclass
class SmallWebRTCRunnerArguments(RunnerArguments):
    """Small WebRTC transport session arguments for the runner.

    Parameters:
        webrtc_connection: Pre-configured WebRTC peer connection
    """

    webrtc_connection: Any
