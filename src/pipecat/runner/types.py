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
from pydantic import BaseModel, ConfigDict, Field


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


class CallData(BaseModel):
    """Parsed telephony handshake data from the provider's first WebSocket messages.

    Populated by :func:`pipecat.runner.utils.parse_telephony_websocket` and exposed on
    ``WebSocketRunnerArguments.call_data`` by ``create_transport``. Gives typed
    attribute access — ``call_data.to_number``, ``call_data.call_id`` — while staying
    dict-compatible (``call_data["call_id"]``, ``call_data.get("body", {})``) so bots
    written against the old dict keep working.

    Fields are populated per provider; absent ones stay ``None``. Provider-specific keys
    not modeled here remain accessible (``extra="allow"``).

    This base holds the fields common to all providers. Provider-specific fields live
    on subclasses (:class:`TelnyxCallData`, :class:`ExotelCallData`), which
    ``parse_telephony_websocket`` / ``create_transport`` construct per provider.

    Parameters:
        stream_id: Provider media-stream identifier.
        call_id: Provider call identifier, normalized across providers (Twilio
            ``callSid``, Plivo ``callId``, Exotel ``call_sid``, Telnyx
            ``call_control_id``).
        from_number: Caller's number. Wire key ``from``.
        to_number: Dialed number. Wire key ``to``.
        body: Custom parameters sent by the provider (e.g. Twilio TwiML stream
            parameters).
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    stream_id: str | None = None
    call_id: str | None = None
    from_number: str | None = Field(default=None, alias="from")
    to_number: str | None = Field(default=None, alias="to")
    body: dict = Field(default_factory=dict)

    def _wire_dict(self) -> dict:
        """The original provider dict shape (wire/alias keys, including extras)."""
        return self.model_dump(by_alias=True)

    def __getitem__(self, key: str):
        """Dict-style access, e.g. ``call_data["call_id"]``."""
        return self._wire_dict()[key]

    def __contains__(self, key: str) -> bool:
        """``"call_id" in call_data`` — True only when the provider set the value."""
        wire = self._wire_dict()
        return key in wire and wire[key] is not None

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``.get`` returning ``default`` when the key is missing or unset."""
        value = self._wire_dict().get(key, default)
        return default if value is None else value


class TelnyxCallData(CallData):
    """Telnyx-specific parsed telephony handshake data.

    Parameters:
        outbound_encoding: Telnyx outbound media encoding.
    """

    outbound_encoding: str | None = None


class ExotelCallData(CallData):
    """Exotel-specific parsed telephony handshake data.

    Parameters:
        account_sid: Exotel account sid.
        custom_parameters: Exotel custom parameters.
    """

    account_sid: str | None = None
    custom_parameters: str | dict | None = None


@dataclass
class RunnerArguments:
    """Base class for runner session arguments.

    Parameters:
        handle_sigint: Whether the bot should install a SIGINT handler.
        handle_sigterm: Whether the bot should install a SIGTERM handler.
        pipeline_idle_timeout_secs: Seconds the pipeline may stay idle before
            shutting down.
        body: Optional request body data passed from the runner entry point.
        call_data: Parsed telephony handshake as a :class:`CallData` model — typed
            attribute access (``call_data.to_number``) that's also dict-compatible
            (``call_data["call_id"]``). Populated by ``create_transport`` (or a direct
            ``parse_telephony_websocket`` call) for telephony connections; ``None``
            otherwise. Lives on the base so any bot can read ``runner_args.call_data``
            uniformly, mirroring ``body``.
        session_id: Identifier for this bot session.
        cli_args: Parsed CLI arguments from the runner, when launched via the
            development runner.
    """

    # Use kw_only so subclasses don't need to worry about ordering.
    handle_sigint: bool = field(init=False, kw_only=True)
    handle_sigterm: bool = field(init=False, kw_only=True)
    pipeline_idle_timeout_secs: int = field(init=False, kw_only=True)
    body: Any | None = field(default_factory=dict, kw_only=True)
    call_data: CallData | None = field(default=None, kw_only=True)
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
class VonageRunnerArguments(RunnerArguments):
    """Vonage transport session arguments for the runner.

    Parameters:
        application_id: Vonage application ID
        vonage_session_id: Vonage session ID
        token: Vonage Session Token
    """

    application_id: str
    vonage_session_id: str
    token: str


@dataclass
class WebSocketRunnerArguments(RunnerArguments):
    """WebSocket transport session arguments for the runner.

    The parsed telephony handshake is available on the inherited ``call_data`` field
    (a :class:`CallData` model), populated by ``create_transport``.

    Parameters:
        websocket: WebSocket connection for audio streaming
        transport_type: Transport type identifier. Set to ``"websocket"`` for plain
            WebSocket connections; ``None`` triggers auto-detection from the first
            telephony provider message. After auto-detection, ``create_transport``
            overwrites this in place with the detected provider (e.g. ``"twilio"``).
        body: Additional request data
    """

    websocket: WebSocket
    transport_type: str | None = None


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
class EvalRunnerArguments(RunnerArguments):
    """Eval transport session arguments for the runner.

    Used to launch a bot with a local ``SingleClientWebsocketServerTransport`` speaking RTVI
    (via ``RTVIEvalSerializer``). The eval harness connects as an RTVI client,
    sends scripted user input, and asserts on the RTVI events the bot emits.
    Intended for fast pipeline behavioral evaluations.

    Parameters:
        host: Host address to bind the eval transport's WebSocket server to.
        port: Port number to bind the eval transport's WebSocket server to.
    """

    host: str = "localhost"
    port: int = 7860


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
        participant_id: This bot's participant id; it broadcasts under
            ``<namespace>/<participant_id>``.
        peer_id: The peer's participant id; the bot subscribes to
            ``<namespace>/<peer_id>``.
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
    serve_bind: str | None = None
    serve_tls_host: str = "localhost"
    serve_tls_cert: str | None = None
    serve_tls_key: str | None = None
    ready_event: asyncio.Event | None = field(default=None, kw_only=True)
    cert_fingerprints: list[str] = field(default_factory=list, kw_only=True)
