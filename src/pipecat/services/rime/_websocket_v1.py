#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime WebSocket v1 framing and context state management."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, cast
from urllib.parse import unquote, urlsplit

from google.protobuf import json_format
from google.protobuf.message import DecodeError

from pipecat.services.rime._proto import websocket_v1_pb2 as proto

WebSocketProtocol = Literal["binary", "json"]
WebSocketMessage = str | bytes

BINARY_SUBPROTOCOL = "rime.v1.binary"
JSON_SUBPROTOCOL = "rime.v1.json"
PROTOCOL_VERSION = 1

_KNOWN_ERROR_KINDS = frozenset(
    {
        "invalid_input",
        "unauthenticated",
        "permission_denied",
        "not_found",
        "resource_exhausted",
        "timeout",
        "unavailable",
        "unimplemented",
        "internal",
    }
)
_JSON_RESPONSE_PAYLOADS = ("ready", "started", "audio", "done", "cancelled", "error")


class WebSocketConnection(Protocol):
    """Socket operations used by the Rime v1 protocol client."""

    @property
    def subprotocol(self) -> str | None: ...

    async def send(self, message: WebSocketMessage) -> None: ...

    async def recv(self) -> WebSocketMessage: ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class RimeV1Error(Exception):
    """Base error for sanitized Rime WebSocket v1 failures."""


class RimeV1ProtocolError(RimeV1Error):
    """Error raised when a response makes the connection unsafe."""


class RimeV1ConnectionError(RimeV1Error):
    """Error raised when a WebSocket operation fails."""


class RimeV1ProviderError(RimeV1Error):
    """Safe provider error received before normal event processing starts."""

    def __init__(self, kind: str, request_id: str | None) -> None:
        """Initialize a provider error without its free-form message.

        Args:
            kind: Stable provider error kind, or ``unknown``.
            request_id: Provider request ID when available.
        """
        super().__init__(f"Rime v1 connection failed with {kind}")
        self.kind = kind
        self.request_id = request_id


class RimeV1StateError(RimeV1Error):
    """Error raised when the caller requests an invalid context transition."""


@dataclass(frozen=True)
class SynthesisOptions:
    """Settings captured when a Rime synthesis context starts.

    Parameters:
        model: Model served by the WebSocket endpoint.
        speaker: Speaker name, if specified.
        language: Language code, if specified.
        sample_rate: Output audio sample rate in Hz.
        time_scale_factor: Audio playback speed factor, if specified.
        text_lookahead_tokens: Coda text lookahead size, if specified.
        pause_between_brackets: Whether Mist pauses between bracketed text.
        phonemize_between_brackets: Whether Mist phonemizes bracketed text.
        save_oovs: Whether Mist saves out-of-vocabulary words.
    """

    model: str
    speaker: str | None
    language: str | None
    sample_rate: int
    time_scale_factor: float | None = None
    text_lookahead_tokens: int | None = None
    pause_between_brackets: bool | None = None
    phonemize_between_brackets: bool | None = None
    save_oovs: bool | None = None


@dataclass(frozen=True)
class ReadyEvent:
    """Connection readiness data.

    Parameters:
        protocol: WebSocket protocol version reported by the server.
        languages: Language codes supported by the server.
        default_language: Default language code, if reported.
    """

    protocol: int
    languages: tuple[str, ...]
    default_language: str | None


@dataclass(frozen=True)
class StartedEvent:
    """Context acceptance data.

    Parameters:
        context_id: Client context accepted by the server.
        request_id: Server request identifier.
    """

    context_id: str
    request_id: str


@dataclass(frozen=True)
class AudioEvent:
    """Raw PCM data for one context.

    Parameters:
        context_id: Client context that owns the audio.
        audio: PCM audio bytes.
    """

    context_id: str
    audio: bytes


@dataclass(frozen=True)
class DoneEvent:
    """Normal context completion.

    Parameters:
        context_id: Client context that completed.
    """

    context_id: str


@dataclass(frozen=True)
class CancelledEvent:
    """Cancelled context completion.

    Parameters:
        context_id: Client context that was cancelled.
    """

    context_id: str


@dataclass(frozen=True)
class ContextErrorEvent:
    """A safe provider error for one context.

    Parameters:
        context_id: Client context that failed.
        kind: Stable provider error kind.
        request_id: Server request identifier, if reported.
    """

    context_id: str
    kind: str
    request_id: str | None


@dataclass(frozen=True)
class ConnectionErrorEvent:
    """A safe provider error for the connection.

    Parameters:
        kind: Stable provider error kind.
        request_id: Server request identifier, if reported.
    """

    kind: str
    request_id: str | None


TerminalEvent = DoneEvent | CancelledEvent | ContextErrorEvent
RimeV1Event = StartedEvent | AudioEvent | TerminalEvent | ConnectionErrorEvent


class InputState(Enum):
    """Client input state for an open context.

    Attributes:
        OPEN: The context accepts text.
        ENDING: The client has sent end of input.
        CANCELLING: The client has requested cancellation.
    """

    OPEN = "open"
    ENDING = "ending"
    CANCELLING = "cancelling"


@dataclass
class _ContextState:
    """Track client and server state for one synthesis context.

    Parameters:
        options: Settings fixed when the context starts.
        input_state: Current client input state.
        started_event: Notification that the server accepted the context.
        terminal_future: Terminal event returned by the server.
        activity_event: Notification used to reset the terminal watchdog.
        server_started: Whether the server sent ``started``.
        emitted_audio: Whether the server sent an audio event.
        request_id: Server request identifier, if reported.
    """

    options: SynthesisOptions
    input_state: InputState
    started_event: asyncio.Event
    terminal_future: asyncio.Future[TerminalEvent]
    activity_event: asyncio.Event
    server_started: bool = False
    emitted_audio: bool = False
    request_id: str | None = None


class _EnvelopeCodec(Protocol):
    """Encode and decode one semantic envelope per WebSocket frame."""

    subprotocol: str

    def encode_request(self, request: proto.WebSocketRequest) -> WebSocketMessage: ...

    def decode_response(self, message: WebSocketMessage) -> proto.WebSocketResponse: ...


class _BinaryEnvelopeCodec:
    """Encode and decode protobuf binary WebSocket envelopes."""

    subprotocol = BINARY_SUBPROTOCOL

    def encode_request(self, request: proto.WebSocketRequest) -> bytes:
        return request.SerializeToString()

    def decode_response(self, message: WebSocketMessage) -> proto.WebSocketResponse:
        if not isinstance(message, bytes):
            raise RimeV1ProtocolError("Rime v1 sent an unexpected WebSocket frame type")
        response = proto.WebSocketResponse()
        try:
            response.ParseFromString(message)
        except (DecodeError, TypeError):
            raise RimeV1ProtocolError("Rime v1 sent invalid protobuf") from None
        return response


class _JsonEnvelopeCodec:
    """Encode and decode proto3 JSON WebSocket envelopes."""

    subprotocol = JSON_SUBPROTOCOL

    def encode_request(self, request: proto.WebSocketRequest) -> str:
        return json_format.MessageToJson(
            request,
            preserving_proto_field_name=False,
            indent=None,
        )

    def decode_response(self, message: WebSocketMessage) -> proto.WebSocketResponse:
        if not isinstance(message, str):
            raise RimeV1ProtocolError("Rime v1 sent an unexpected WebSocket frame type")
        try:
            envelope = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            raise RimeV1ProtocolError("Rime v1 sent invalid JSON") from None
        _validate_json_envelope(envelope)

        response = proto.WebSocketResponse()
        try:
            json_format.ParseDict(envelope, response, ignore_unknown_fields=True)
        except (json_format.ParseError, TypeError):
            raise RimeV1ProtocolError("Rime v1 sent invalid JSON") from None
        return response


def _validate_json_envelope(envelope: Any) -> None:
    if not isinstance(envelope, dict):
        raise RimeV1ProtocolError("Rime v1 envelope must be an object")

    payloads = [name for name in _JSON_RESPONSE_PAYLOADS if name in envelope]
    if len(payloads) != 1:
        raise RimeV1ProtocolError("Rime v1 envelope must contain exactly one payload")

    payload = payloads[0]
    value = envelope[payload]
    if payload == "audio":
        if not isinstance(value, str):
            raise RimeV1ProtocolError("Rime v1 sent a non-string audio payload")
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise RimeV1ProtocolError("Rime v1 sent invalid Base64 audio") from None
    elif not isinstance(value, dict):
        raise RimeV1ProtocolError(f"Rime v1 sent a malformed {payload} event")


def _codec_for_protocol(protocol: WebSocketProtocol | str) -> _EnvelopeCodec:
    if protocol == "binary":
        return _BinaryEnvelopeCodec()
    if protocol == "json":
        return _JsonEnvelopeCodec()
    raise ValueError('websocket_protocol must be "binary" or "json"')


def subprotocol_for_protocol(protocol: WebSocketProtocol | str) -> str:
    """Return the WebSocket subprotocol for a public protocol name."""
    return _codec_for_protocol(protocol).subprotocol


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_trusted_rime_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized == "rime.ai" or normalized.endswith(".rime.ai")


def validate_websocket_url(websocket_url: str, *, allow_custom_endpoint: bool = False) -> None:
    """Reject a v1 endpoint that could expose credentials or select a wrong route."""
    parts = urlsplit(websocket_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc or parts.hostname is None:
        raise ValueError("Rime v1 websocket_url must be an absolute ws or wss URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Rime v1 websocket_url must not contain user information")
    if parts.fragment:
        raise ValueError("Rime v1 websocket_url must not contain a fragment")
    if not parts.path.endswith("/ws"):
        raise ValueError("Rime v1 websocket_url path must end with /ws")
    if parts.scheme == "ws" and not _is_loopback_host(parts.hostname):
        raise ValueError("Rime v1 websocket_url must use wss unless it uses a loopback host")
    if (
        not allow_custom_endpoint
        and not _is_loopback_host(parts.hostname)
        and not _is_trusted_rime_host(parts.hostname)
    ):
        raise ValueError(
            "Rime v1 websocket_url must use a trusted Rime host; "
            "set allow_custom_endpoint=True to use another host"
        )


def model_from_websocket_url(
    websocket_url: str, *, allow_custom_endpoint: bool = False
) -> str | None:
    """Return the model from ``/{model}/ws``, or None for a dedicated endpoint."""
    validate_websocket_url(websocket_url, allow_custom_endpoint=allow_custom_endpoint)
    segments = [segment for segment in urlsplit(websocket_url).path.split("/") if segment]
    if len(segments) == 1:
        return None
    model = unquote(segments[-2]).lower()
    if not model or "/" in model:
        raise ValueError("Rime v1 websocket_url contains an invalid model route")
    if model == "mistv3":
        raise ValueError("Rime v1 uses /mist/ws instead of /mistv3/ws")
    return "mistv3" if model == "mist" else model


def _start_payload(options: SynthesisOptions) -> proto.SynthesisRequest:
    audio_parameters = proto.AudioParameters(
        audio_format="audio/pcm",
        sampling_rate=options.sample_rate,
    )
    if options.time_scale_factor is not None:
        audio_parameters.time_scale_factor = options.time_scale_factor

    request = proto.SynthesisRequest(text="", audio_parameters=audio_parameters)
    if options.speaker is not None:
        request.speaker = options.speaker
    if options.language is not None:
        request.language = options.language

    if options.model == "coda" and options.text_lookahead_tokens is not None:
        request.coda_parameters.text_lookahead_tokens = options.text_lookahead_tokens
    elif options.model.startswith("mist"):
        if options.pause_between_brackets is not None:
            request.mist_parameters.pause_between_brackets = options.pause_between_brackets
        if options.phonemize_between_brackets is not None:
            request.mist_parameters.phonemize_between_brackets = options.phonemize_between_brackets
        if options.save_oovs is not None:
            request.mist_parameters.save_oovs = options.save_oovs
    return request


def _request(context_id: str, payload: str, value: object = None) -> proto.WebSocketRequest:
    request = proto.WebSocketRequest(context_id=context_id)
    if payload == "start" and isinstance(value, SynthesisOptions):
        request.start.CopyFrom(_start_payload(value))
    elif payload == "text" and isinstance(value, str):
        request.text = value
    elif payload == "end":
        request.end.SetInParent()
    elif payload == "cancel":
        request.cancel.SetInParent()
    else:
        raise ValueError("Unsupported Rime v1 request payload")
    return request


def _safe_error(error: proto.WebSocketError) -> tuple[str, str | None]:
    if not error.kind or not error.message:
        raise RimeV1ProtocolError("Rime v1 sent a malformed error event")
    kind = error.kind if error.kind in _KNOWN_ERROR_KINDS else "unknown"
    request_id = error.request_id if error.HasField("request_id") else None
    return kind, request_id


class RimeWebSocketV1Client:
    """Manage typed Rime v1 envelopes and multiplexed context state."""

    def __init__(
        self,
        websocket: WebSocketConnection,
        *,
        protocol: WebSocketProtocol,
    ) -> None:
        """Initialize a client over an open WebSocket.

        Args:
            websocket: Connected WebSocket adapter.
            protocol: Envelope encoding selected for the connection.
        """
        self._websocket = websocket
        self._codec = _codec_for_protocol(protocol)
        self._send_lock = asyncio.Lock()
        self._contexts: dict[str, _ContextState] = {}
        self._closed_contexts: set[str] = set()
        self._ready = False
        self._closed = False

    @property
    def subprotocol(self) -> str:
        """Return the required WebSocket subprotocol."""
        return self._codec.subprotocol

    @property
    def context_ids(self) -> tuple[str, ...]:
        """Return context IDs that have not completed."""
        return tuple(
            context_id
            for context_id, state in self._contexts.items()
            if not state.terminal_future.done()
        )

    def has_context(self, context_id: str) -> bool:
        """Return whether the context exists and has not completed."""
        state = self._contexts.get(context_id)
        return state is not None and not state.terminal_future.done()

    async def wait_ready(self, timeout_s: float) -> ReadyEvent:
        """Consume and validate the connection readiness event."""
        if self._ready:
            raise RimeV1StateError("Rime v1 ready was already consumed")
        if self._closed:
            raise RimeV1ConnectionError("Rime v1 client is closed")
        try:
            message = await asyncio.wait_for(self._websocket.recv(), timeout=timeout_s)
        except TimeoutError:
            raise RimeV1ConnectionError("Timed out waiting for the Rime v1 ready event") from None
        except Exception:
            raise RimeV1ConnectionError("Failed to read the Rime v1 ready event") from None

        response = self._codec.decode_response(message)
        payload = response.WhichOneof("payload")
        if payload == "error":
            kind, request_id = _safe_error(response.error)
            raise RimeV1ProviderError(kind, request_id)
        if payload != "ready" or response.context_id:
            raise RimeV1ProtocolError("Rime v1 did not send a connection-level ready event")
        if response.ready.protocol != PROTOCOL_VERSION:
            raise RimeV1ProtocolError("Rime v1 reported an unsupported protocol version")

        self._ready = True
        default_language = (
            response.ready.default_language if response.ready.HasField("default_language") else None
        )
        return ReadyEvent(
            protocol=response.ready.protocol,
            languages=tuple(response.ready.languages),
            default_language=default_language,
        )

    async def send_text(
        self,
        context_id: str,
        options: SynthesisOptions,
        text: str,
    ) -> None:
        """Open a context if needed and append one complete text fragment."""
        self._require_ready()
        if not text:
            return
        self._validate_context_id(context_id)

        async with self._send_lock:
            state = self._contexts.get(context_id)
            if state is None:
                self._closed_contexts.discard(context_id)
                loop = asyncio.get_running_loop()
                state = _ContextState(
                    options=options,
                    input_state=InputState.OPEN,
                    started_event=asyncio.Event(),
                    terminal_future=loop.create_future(),
                    activity_event=asyncio.Event(),
                )
                self._contexts[context_id] = state
                await self._send_locked(_request(context_id, "start", options))
                if state.terminal_future.done():
                    return
            elif state.input_state is not InputState.OPEN or state.terminal_future.done():
                raise RimeV1StateError("Rime v1 context no longer accepts text")

            await self._send_locked(_request(context_id, "text", text))

    async def end(self, context_id: str) -> None:
        """Declare the normal end of input for a context."""
        self._require_ready()
        async with self._send_lock:
            state = self._contexts.get(context_id)
            if state is None or state.terminal_future.done():
                return
            if state.input_state is InputState.ENDING:
                return
            if state.input_state is InputState.CANCELLING:
                return
            state.input_state = InputState.ENDING
            await self._send_locked(_request(context_id, "end"))

    async def cancel(self, context_id: str) -> None:
        """Cancel an open context."""
        self._require_ready()
        async with self._send_lock:
            state = self._contexts.get(context_id)
            if state is None or state.terminal_future.done():
                return
            if state.input_state is InputState.CANCELLING:
                return
            state.input_state = InputState.CANCELLING
            await self._send_locked(_request(context_id, "cancel"))

    async def events(self) -> AsyncIterator[RimeV1Event]:
        """Yield validated provider events until the socket closes."""
        self._require_ready()
        while not self._closed:
            try:
                message = await self._websocket.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                raise RimeV1ConnectionError("Failed to read from the Rime v1 WebSocket") from None
            response = self._codec.decode_response(message)
            yield self._event_from_response(response)

    async def wait_started(self, context_id: str, timeout_s: float) -> None:
        """Wait until the provider accepts a context."""
        state = self._require_context(context_id)
        try:
            await asyncio.wait_for(state.started_event.wait(), timeout=timeout_s)
        except TimeoutError:
            raise RimeV1ConnectionError("Timed out waiting for the Rime v1 started event") from None

    async def wait_terminal(self, context_id: str, timeout_s: float) -> TerminalEvent:
        """Wait for one terminal context event."""
        state = self._require_context(context_id)
        try:
            return await asyncio.wait_for(asyncio.shield(state.terminal_future), timeout=timeout_s)
        except TimeoutError:
            raise RimeV1ConnectionError("Timed out waiting for a Rime v1 terminal event") from None

    async def wait_activity(self, context_id: str, timeout_s: float) -> None:
        """Wait for context activity and reset its notification."""
        state = self._require_context(context_id)
        if state.terminal_future.done():
            return
        state.activity_event.clear()
        if state.terminal_future.done():
            return
        try:
            await asyncio.wait_for(state.activity_event.wait(), timeout=timeout_s)
        except TimeoutError:
            raise RimeV1ConnectionError("Timed out waiting for a Rime v1 event after end") from None

    def discard_context(self, context_id: str) -> None:
        """Release a completed context after its consumer finishes cleanup."""
        state = self._contexts.get(context_id)
        if state is not None and state.terminal_future.done():
            del self._contexts[context_id]
            self._closed_contexts.discard(context_id)

    async def close(self) -> None:
        """Close the socket and reject later work."""
        if not self._closed:
            self.invalidate()
        await self._websocket.close()

    def invalidate(self) -> None:
        """Reject new work and release protocol waiters."""
        self._closed = True
        for state in self._contexts.values():
            if not state.terminal_future.done():
                state.terminal_future.cancel()
            state.activity_event.set()
            state.started_event.set()

    async def _send_locked(self, request: proto.WebSocketRequest) -> None:
        try:
            await self._websocket.send(self._codec.encode_request(request))
        except asyncio.CancelledError:
            raise
        except Exception:
            self._closed = True
            raise RimeV1ConnectionError("Failed to write to the Rime v1 WebSocket") from None

    def _event_from_response(self, response: proto.WebSocketResponse) -> RimeV1Event:
        payload = cast(str | None, response.WhichOneof("payload"))
        if payload is None:
            raise RimeV1ProtocolError("Rime v1 envelope must contain exactly one payload")
        if payload == "ready":
            raise RimeV1ProtocolError("Rime v1 sent ready more than once")
        if payload == "error" and not response.context_id:
            kind, request_id = _safe_error(response.error)
            return ConnectionErrorEvent(kind=kind, request_id=request_id)
        if not response.context_id:
            raise RimeV1ProtocolError("Rime v1 sent a context event without a context ID")

        context_id = response.context_id
        if context_id in self._closed_contexts:
            raise RimeV1ProtocolError("Rime v1 sent an event after a terminal event")
        state = self._contexts.get(context_id)
        if state is None:
            raise RimeV1ProtocolError("Rime v1 sent an event for an unknown context")
        if state.terminal_future.done():
            raise RimeV1ProtocolError("Rime v1 sent an event after a terminal event")
        state.activity_event.set()

        if payload == "started":
            if state.server_started:
                raise RimeV1ProtocolError("Rime v1 sent an invalid started event")
            state.server_started = True
            state.request_id = response.started.request_id
            state.started_event.set()
            return StartedEvent(context_id=context_id, request_id=response.started.request_id)
        if payload == "audio":
            if not state.server_started:
                raise RimeV1ProtocolError("Rime v1 sent audio before started")
            state.emitted_audio = True
            return AudioEvent(context_id=context_id, audio=response.audio)
        if payload == "done":
            if not state.server_started:
                raise RimeV1ProtocolError("Rime v1 sent done before started")
            if state.input_state not in (InputState.ENDING, InputState.CANCELLING):
                raise RimeV1ProtocolError("Rime v1 sent done before input ended")
            event: TerminalEvent = DoneEvent(context_id=context_id)
        elif payload == "cancelled":
            if state.input_state is not InputState.CANCELLING:
                raise RimeV1ProtocolError("Rime v1 cancelled a context unexpectedly")
            event = CancelledEvent(context_id=context_id)
        elif payload == "error":
            kind, request_id = _safe_error(response.error)
            event = ContextErrorEvent(
                context_id=context_id,
                kind=kind,
                request_id=request_id or state.request_id,
            )
        else:
            raise RimeV1ProtocolError("Rime v1 sent an unsupported event")

        state.terminal_future.set_result(event)
        state.started_event.set()
        state.activity_event.set()
        self._closed_contexts.add(context_id)
        return event

    def _require_ready(self) -> None:
        if self._closed:
            raise RimeV1ConnectionError("Rime v1 client is closed")
        if not self._ready:
            raise RimeV1StateError("Rime v1 client is not ready")

    def _require_context(self, context_id: str) -> _ContextState:
        state = self._contexts.get(context_id)
        if state is None:
            raise RimeV1StateError("Rime v1 context is not open")
        return state

    @staticmethod
    def _validate_context_id(context_id: str) -> None:
        if not context_id or len(context_id.encode("utf-8")) > 128:
            raise ValueError("Rime v1 context ID must contain 1 to 128 UTF-8 bytes")
