#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Runner session argument types for the development runner.

These types are used by the development runner to pass transport-specific
information to bot functions.
"""

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import WebSocket
from pydantic import BaseModel


class DialinSettings(BaseModel):
    """Dial-in settings from the Daily webhook.

    This model matches the structure sent by Pipecat Cloud and Daily.co webhooks
    for incoming PSTN/SIP calls.

    Parameters:
        call_id: Unique identifier for the call (UUID representing sessionId in SIP Network)
        call_domain: Daily domain for the call (UUID representing Daily Domain on SIP Network)
        To: The dialed phone number (optional)
        From: The caller's phone number (optional)
        sip_headers: Optional SIP headers from the call
    """

    call_id: str
    call_domain: str
    To: Optional[str] = None
    From: Optional[str] = None
    sip_headers: Optional[Dict[str, str]] = None


class DailyDialinRequest(BaseModel):
    """Request data for Daily PSTN dial-in requests.

    This is the structure passed in runner_args.body for dial-in calls.
    It matches the payload structure from Pipecat Cloud's dial-in webhook handler.

    Parameters:
        dialin_settings: Dial-in configuration including call_id, call_domain, To, From
        daily_api_key: Daily API key for pinlessCallUpdate (required for dial-in)
        daily_api_url: Daily API URL (staging or production)
    """

    dialin_settings: DialinSettings
    daily_api_key: str
    daily_api_url: str


@dataclass
class RunnerArguments:
    """Base class for runner session arguments."""

    # Use kw_only so subclasses don't need to worry about ordering.
    handle_sigint: bool = field(init=False, kw_only=True)
    handle_sigterm: bool = field(init=False, kw_only=True)
    pipeline_idle_timeout_secs: int = field(init=False, kw_only=True)
    body: Optional[Any] = field(default_factory=dict, kw_only=True)
    cli_args: Optional[argparse.Namespace] = field(default=None, init=False, kw_only=True)

    def __post_init__(self):
        self.handle_sigint = False
        self.handle_sigterm = False
        self.pipeline_idle_timeout_secs = 300


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


@dataclass
class WebSocketRunnerArguments(RunnerArguments):
    """WebSocket transport session arguments for the runner.

    Parameters:
        websocket: WebSocket connection for audio streaming
        body: Additional request data
    """

    websocket: WebSocket


@dataclass
class SmallWebRTCRunnerArguments(RunnerArguments):
    """Small WebRTC transport session arguments for the runner.

    Parameters:
        webrtc_connection: Pre-configured WebRTC peer connection
    """

    webrtc_connection: Any


@dataclass
class LiveKitRunnerArguments(RunnerArguments):
    """LiveKit transport session arguments for the runner.

    Parameters:
        room_name: LiveKit room name to join
        token: Authentication token for the room
        body: Additional request data
    """

    room_name: str
    url: str
    token: Optional[str] = None
