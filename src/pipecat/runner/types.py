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
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

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
    To: str | None = None
    From: str | None = None
    sip_headers: dict[str, str] | None = None


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
    """Base class for runner session arguments.

    Parameters:
        handle_sigint: Whether the bot should install a SIGINT handler.
        handle_sigterm: Whether the bot should install a SIGTERM handler.
        pipeline_idle_timeout_secs: Seconds the pipeline may stay idle before
            shutting down.
        body: Optional request body data passed from the runner entry point.
        session_id: Identifier for this bot session.
        cli_args: Parsed CLI arguments from the runner, when launched via the
            development runner.
    """

    # Use kw_only so subclasses don't need to worry about ordering.
    handle_sigint: bool = field(init=False, kw_only=True)
    handle_sigterm: bool = field(init=False, kw_only=True)
    pipeline_idle_timeout_secs: int = field(init=False, kw_only=True)
    body: Any | None = field(default_factory=dict, kw_only=True)
    session_id: str | None = field(default=None, kw_only=True)
    cli_args: argparse.Namespace | None = field(default=None, init=False, kw_only=True)

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
    token: str | None = None


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
    token: str


@dataclass
class MOQRunnerArguments(RunnerArguments):
    """MOQ (Media over QUIC) transport session arguments for the runner.

    The ``ready_event`` and ``cert_fingerprints`` fields are populated
    automatically by :func:`pipecat.runner.utils.create_transport`; bots
    don't need to thread them by hand.

    Parameters:
        host: MOQ relay/server hostname the browser uses to connect.
        port: MOQ relay/server port.
        path: MOQ endpoint path on the relay (client mode).
        namespace: MOQ namespace (like a room identifier).
        verify_ssl: Whether to verify SSL certificates (client mode).
        serve: When True, the bot binds its own MOQ server instead of
            dialing a relay — useful for local dev with no separate
            ``moq-relay`` process.
        serve_bind: Address to bind in serve mode (e.g. ``"[::]:4080"``).
        serve_tls_host: Hostname used for the generated self-signed cert
            when no on-disk cert/key is provided.
        serve_tls_cert: Path to a PEM-encoded TLS cert chain.
        serve_tls_key: Path to the matching PEM-encoded private key.
        ready_event: Event the bot fires once it has finished MOQ
            bring-up. The HTTP ``/start`` endpoint waits on this before
            telling the browser to open its WebTransport.
        cert_fingerprints: SHA-256 fingerprints (hex) of the bot's TLS
            cert chain — populated by the transport in serve mode so
            ``/api/config`` can hand them to the browser for pinning.
    """

    host: str
    port: int
    path: str = "/moq"
    namespace: str = "pipecat"
    participant_id: str = "bot0"
    peer_id: str = "client0"
    verify_ssl: bool = True
    serve: bool = False
    serve_bind: Optional[str] = None
    serve_tls_host: str = "localhost"
    serve_tls_cert: Optional[str] = None
    serve_tls_key: Optional[str] = None
    ready_event: Optional[asyncio.Event] = field(default=None, kw_only=True)
    cert_fingerprints: list[str] = field(default_factory=list, kw_only=True)
