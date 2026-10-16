#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime WebSocket v1 text-to-speech service implementation.

The first part of this module speaks the v1 wire protocol: envelope codecs,
endpoint validation, and per-context state checks. ``RimeV1TTSService`` maps
that protocol onto Pipecat audio contexts.
"""

import asyncio
import base64
import binascii
import ipaddress
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, Protocol, cast
from urllib.parse import unquote, urlsplit

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.rime.tts import language_to_rime_language
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory, classify_http_exception
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

try:
    from rime_api import text_to_speech_pb2 as proto
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use RimeV1TTSService, you need to `uv add "pipecat-ai[rime]"`.')
    raise ImportError(f"Missing module: {e}") from e

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
    def subprotocol(self) -> str | None:
        """Subprotocol the server selected."""
        ...

    async def send(self, message: WebSocketMessage) -> None:
        """Send one WebSocket frame."""
        ...

    async def recv(self) -> WebSocketMessage:
        """Receive one WebSocket frame."""
        ...

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the WebSocket."""
        ...


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
        *,
        timeout_s: float = 10.0,
    ) -> None:
        """Open a context and append text within timeout_s, including the lock wait."""
        self._require_ready()
        if not text:
            return
        self._validate_context_id(context_id)

        async with self._send_operation(timeout_s):
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

    async def end(self, context_id: str, *, timeout_s: float = 10.0) -> None:
        """End context input within timeout_s, including the lock wait."""
        self._require_ready()
        async with self._send_operation(timeout_s):
            state = self._contexts.get(context_id)
            if state is None or state.terminal_future.done():
                return
            if state.input_state is InputState.ENDING:
                return
            if state.input_state is InputState.CANCELLING:
                return
            state.input_state = InputState.ENDING
            await self._send_locked(_request(context_id, "end"))

    async def cancel(self, context_id: str, *, timeout_s: float = 1.0) -> None:
        """Cancel a context within timeout_s, including the lock wait."""
        self._require_ready()
        async with self._send_operation(timeout_s):
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

    @asynccontextmanager
    async def _send_operation(self, timeout_s: float) -> AsyncIterator[None]:
        """Bound the lock wait and writes by one deadline."""
        try:
            async with asyncio.timeout(timeout_s):
                async with self._send_lock:
                    self._require_ready()
                    yield
        except TimeoutError:
            # Delivery is uncertain. Reject queued work until the service closes the socket.
            self._closed = True
            raise RimeV1ConnectionError("Timed out writing to the Rime v1 WebSocket") from None

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


_DEFAULT_WEBSOCKET_URL = "wss://api.rime.ai/coda/ws"
_V1_READY_TIMEOUT_S = 10.0
_V1_START_TIMEOUT_S = 10.0
_V1_TERMINAL_TIMEOUT_S = 10.0
_V1_CANCEL_TIMEOUT_S = 1.0


def _resolve_websocket_v1_model(
    websocket_url: str,
    explicit_model: str | None,
    *,
    allow_custom_endpoint: bool,
) -> str:
    """Resolve the model bound to a WebSocket v1 endpoint."""
    endpoint_model = model_from_websocket_url(
        websocket_url, allow_custom_endpoint=allow_custom_endpoint
    )
    if endpoint_model is None:
        if explicit_model is None:
            raise ValueError("Rime WebSocket v1 requires a model for a dedicated endpoint")
        return explicit_model
    if explicit_model is not None and explicit_model != endpoint_model:
        raise ValueError("settings.model does not match the v1 endpoint model")
    return endpoint_model


@dataclass
class RimeV1TTSSettings(TTSSettings):
    """Settings for RimeV1TTSService.

    Each context uses the settings in effect when it starts.

    Parameters:
        time_scale_factor: Audio playback speed factor. Values above 1.0 slow down
            the audio; values below 1.0 speed it up. Not supported by mistv2.
        text_lookahead_tokens: Number of input tokens to collect before audio
            starts (Coda only).
        pause_between_brackets: Whether to add pauses between bracketed content (Mist only).
        phonemize_between_brackets: Whether to phonemize bracketed content (Mist only).
        save_oovs: Whether to save out-of-vocabulary words (Mist only).
    """

    time_scale_factor: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_lookahead_tokens: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pause_between_brackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    phonemize_between_brackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    save_oovs: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {"speaker": "voice"}


class RimeV1TTSService(WebsocketTTSService):
    """Text-to-Speech service using Rime's WebSocket v1 API.

    Sends protobuf envelopes, in binary or proto3 JSON encoding, over one
    connection that carries a synthesis context per Pipecat turn. Interruptions
    cancel the turn's context without reconnecting. The model is read from the
    endpoint path (``/coda/ws``, ``/mist/ws``); a dedicated ``/ws`` endpoint
    requires ``settings.model``. WebSocket v1 does not report word timestamps.

    Requires the ``rime`` extra: ``uv add "pipecat-ai[rime]"``.
    """

    Settings = RimeV1TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        websocket_url: str = _DEFAULT_WEBSOCKET_URL,
        websocket_protocol: Literal["binary", "json"] = "binary",
        allow_custom_endpoint: bool = False,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Rime WebSocket v1 TTS service.

        Args:
            api_key: Rime API key for authentication.
            websocket_url: Rime WebSocket v1 endpoint. Its path selects the model.
            websocket_protocol: Envelope encoding, ``"binary"`` or ``"json"``.
            allow_custom_endpoint: Allow a secure endpoint outside ``rime.ai``.
            sample_rate: Audio sample rate in Hz.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent class. Text is always
                aggregated by sentence.
        """
        if websocket_protocol not in ("binary", "json"):
            raise ValueError('websocket_protocol must be "binary" or "json"')
        text_aggregation_mode = kwargs.pop("text_aggregation_mode", None)
        aggregate_sentences = kwargs.pop("aggregate_sentences", None)
        if aggregate_sentences is False or text_aggregation_mode not in (
            None,
            TextAggregationMode.SENTENCE,
        ):
            raise ValueError("Rime WebSocket v1 requires sentence text aggregation")

        default_settings = self.Settings(
            model=None,
            voice=None,
            language=None,
            time_scale_factor=None,
            text_lookahead_tokens=None,
            pause_between_brackets=None,
            phonemize_between_brackets=None,
            save_oovs=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        explicit_model = default_settings.model if isinstance(default_settings.model, str) else None
        endpoint_model = _resolve_websocket_v1_model(
            websocket_url,
            explicit_model,
            allow_custom_endpoint=allow_custom_endpoint,
        )
        default_settings.model = endpoint_model
        self._validate_v1_settings(default_settings)

        super().__init__(
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            push_text_frames=False,
            push_start_frame=True,
            push_stop_frames=False,
            pause_frame_processing=True,
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        # Keep spelled-out text in one aggregate so a sentence split never breaks a tag.
        self._text_aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=self._text_aggregation_mode
        )

        self._api_key = api_key
        self._websocket_url = websocket_url
        self._websocket_protocol: WebSocketProtocol = websocket_protocol
        self._v1_endpoint_model = endpoint_model

        self._receive_task = None
        self._v1_client: RimeWebSocketV1Client | None = None
        self._v1_receiving_client: RimeWebSocketV1Client | None = None
        self._v1_options_by_context: dict[str, SynthesisOptions] = {}
        self._v1_keepalive_tasks: dict[str, asyncio.Task[None]] = {}
        self._v1_start_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_terminal_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_cancel_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_failed_contexts: set[str] = set()
        self._v1_closed_contexts: set[str] = set()
        self._v1_invalidating = False
        self._v1_connection_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    # A set of Rime-specific helpers for text transformations
    @staticmethod
    def SPELL(text: str) -> str:
        """Wrap text in Rime spell function."""
        return f"spell({text})"

    @staticmethod
    def PAUSE_TAG(seconds: float) -> str:
        """Convenience method to create a pause tag."""
        return f"<{seconds * 1000}>"

    @staticmethod
    def _validate_v1_settings(settings: Settings) -> None:
        """Validate one complete settings snapshot.

        Args:
            settings: Settings to validate.
        """
        if settings.extra:
            names = ", ".join(sorted(settings.extra))
            raise ValueError(f"Rime WebSocket v1 does not support these settings: {names}")

        model = settings.model
        if not isinstance(model, str):
            raise ValueError("Rime WebSocket v1 requires a model")
        mist_fields = (
            settings.pause_between_brackets,
            settings.phonemize_between_brackets,
            settings.save_oovs,
        )
        if not model.startswith("mist") and any(value is not None for value in mist_fields):
            raise ValueError("Rime WebSocket v1 Mist settings require a Mist model")
        if model != "coda" and settings.text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires the Coda model")
        if model == "mistv2" and settings.time_scale_factor is not None:
            raise ValueError("time_scale_factor is not supported by the mistv2 model")

    def _build_v1_options(self) -> SynthesisOptions:
        """Build an immutable settings snapshot for a new context."""
        self._validate_v1_settings(self._settings)
        model = self._settings.model
        if not isinstance(model, str):
            raise ValueError("Rime WebSocket v1 requires a model")
        language = self._settings.language
        return SynthesisOptions(
            model=model,
            speaker=self._settings.voice if isinstance(self._settings.voice, str) else None,
            language=language if isinstance(language, str) else None,
            sample_rate=self.sample_rate,
            time_scale_factor=cast(float | None, self._settings.time_scale_factor),
            text_lookahead_tokens=cast(int | None, self._settings.text_lookahead_tokens),
            pause_between_brackets=cast(bool | None, self._settings.pause_between_brackets),
            phonemize_between_brackets=cast(bool | None, self._settings.phonemize_between_brackets),
            save_oovs=cast(bool | None, self._settings.save_oovs),
        )

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta to contexts that start after the update."""
        candidate = self._settings.copy()
        candidate.apply_update(self.Settings.from_mapping(delta.given_fields()))
        if candidate.model != self._v1_endpoint_model:
            raise ValueError("settings.model does not match the v1 endpoint model")
        self._validate_v1_settings(candidate)
        return await super()._update_settings(delta)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        """Establish websocket connection and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close websocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        """Connect to Rime v1 and wait until its engine is ready."""
        async with self._v1_connection_lock:
            await self._connect_websocket_locked()

    async def _connect_websocket_locked(self) -> None:
        """Connect to Rime v1 while holding the connection lifecycle lock."""
        websocket = None
        try:
            if self._websocket and self._websocket.state is State.OPEN and self._v1_client:
                return

            protocol = self._websocket_protocol
            subprotocol = subprotocol_for_protocol(protocol)
            websocket = await self._websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                subprotocols=[subprotocol],
            )
            if websocket.subprotocol != subprotocol:
                raise RimeV1ProtocolError("Rime v1 selected an unexpected WebSocket subprotocol")

            client = RimeWebSocketV1Client(websocket, protocol=protocol)
            await client.wait_ready(_V1_READY_TIMEOUT_S)
            self._websocket = websocket
            self._v1_client = client
            self._v1_invalidating = False
            self._v1_closed_contexts.clear()
            await self._call_event_handler("on_connected")
        except Exception as e:
            if websocket is not None:
                await websocket.close()
            self._websocket = None
            self._v1_client = None
            category = self._v1_exception_category(e)
            message = "Failed to connect to Rime WebSocket v1"
            await self.push_error(error_msg=message, category=category)
            await self._call_event_handler("on_connection_error", message)

    async def _disconnect_websocket(self) -> None:
        """Close Rime v1 and clear all connection-scoped state."""
        async with self._v1_connection_lock:
            await self._disconnect_websocket_locked()

    async def _disconnect_websocket_locked(self) -> None:
        """Close Rime v1 while holding the connection lifecycle lock."""
        try:
            await self.stop_all_metrics()
            await self._cancel_all_v1_tasks()
            client = self._v1_client
            websocket = self._websocket
            self._v1_client = None
            self._v1_receiving_client = None
            self._websocket = None
            if client:
                await client.close()
            elif websocket:
                await websocket.close()
        except Exception:
            await self.push_error(
                error_msg="Failed to disconnect from Rime WebSocket v1",
                category=ErrorCategory.CONNECTIVITY,
            )
        finally:
            for context_id in self.get_audio_contexts():
                if context_id not in self._v1_closed_contexts and self.audio_context_available(
                    context_id
                ):
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
            self._v1_options_by_context.clear()
            self._v1_closed_contexts.clear()
            await self._call_event_handler("on_disconnected")

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect the socket without replacing a newer ready connection."""
        logger.warning(f"{self} reconnecting (attempt: {attempt_number})")
        async with self._v1_connection_lock:
            receiving_client = self._v1_receiving_client
            if (
                receiving_client is not None
                and self._v1_client is not receiving_client
                and self._v1_client is not None
                and self._websocket is not None
                and self._websocket.state is State.OPEN
            ):
                return True

            await self._disconnect_websocket_locked()
            await self._close_connection_contexts()
            await self._connect_websocket_locked()
            if not await self._verify_connection():
                raise ConnectionError(f"{self} websocket reconnection failed verification")
            return True

    def _get_v1_client(self) -> RimeWebSocketV1Client:
        """Return the ready v1 client or raise a safe connection error."""
        if self._v1_client:
            return self._v1_client
        raise RimeV1ConnectionError("Rime WebSocket v1 is not connected")

    @staticmethod
    def _v1_error_category(kind: str) -> ErrorCategory:
        categories = {
            "invalid_input": ErrorCategory.INVALID_REQUEST,
            "unauthenticated": ErrorCategory.AUTHENTICATION,
            "permission_denied": ErrorCategory.AUTHORIZATION,
            "not_found": ErrorCategory.INVALID_REQUEST,
            "resource_exhausted": ErrorCategory.RATE_LIMIT,
            "timeout": ErrorCategory.CONNECTIVITY,
            "unavailable": ErrorCategory.CONNECTIVITY,
            "unimplemented": ErrorCategory.SERVER,
            "internal": ErrorCategory.SERVER,
        }
        return categories.get(kind, ErrorCategory.UNKNOWN)

    @classmethod
    def _v1_context_error_category(cls, kind: str) -> ErrorCategory:
        """Classify a context error without changing service usability."""
        category = cls._v1_error_category(kind)
        if category.is_permanent:
            return ErrorCategory.UNKNOWN
        return category

    @classmethod
    def _v1_exception_category(cls, exception: Exception) -> ErrorCategory:
        if isinstance(exception, RimeV1ProviderError):
            return cls._v1_error_category(exception.kind)
        if isinstance(exception, RimeV1ProtocolError):
            return ErrorCategory.SERVER
        if isinstance(exception, (RimeV1ConnectionError, ConnectionError, TimeoutError)):
            return ErrorCategory.CONNECTIVITY
        if isinstance(exception, ValueError):
            return ErrorCategory.INVALID_REQUEST
        return classify_http_exception(exception)

    @staticmethod
    def _v1_error_message(kind: str, request_id: str | None) -> str:
        message = f"Rime WebSocket v1 request failed with {kind}"
        if request_id:
            message += f" (request ID: {request_id})"
        return message

    async def _cancel_v1_context_tasks(
        self,
        context_id: str,
        *,
        include_terminal: bool = True,
        include_cancel: bool = True,
    ) -> None:
        """Cancel tracked tasks for one v1 context."""
        task_maps = [self._v1_keepalive_tasks, self._v1_start_watchdogs]
        if include_terminal:
            task_maps.append(self._v1_terminal_watchdogs)
        if include_cancel:
            task_maps.append(self._v1_cancel_watchdogs)

        current = asyncio.current_task()
        for task_map in task_maps:
            task = task_map.pop(context_id, None)
            if task and task is not current:
                await self.cancel_task(task)

    async def _cancel_all_v1_tasks(self) -> None:
        """Cancel every tracked v1 context task."""
        context_ids = set()
        for task_map in (
            self._v1_keepalive_tasks,
            self._v1_start_watchdogs,
            self._v1_terminal_watchdogs,
            self._v1_cancel_watchdogs,
        ):
            context_ids.update(task_map)
        for context_id in context_ids:
            await self._cancel_v1_context_tasks(context_id)

    @staticmethod
    def _remove_current_v1_task(task_map: dict[str, asyncio.Task[None]], context_id: str) -> None:
        """Remove a context task when that same task finishes."""
        if task_map.get(context_id) is asyncio.current_task():
            task_map.pop(context_id, None)

    async def _finish_v1_context(
        self,
        context_id: str,
        *,
        error: ErrorFrame | None = None,
        emit_stop: bool = True,
    ) -> None:
        """Finish protocol and Pipecat state for one context once."""
        first_close = context_id not in self._v1_closed_contexts
        self._v1_closed_contexts.add(context_id)
        await self.stop_ttfb_metrics()
        if error:
            await self.stop_all_metrics()
        await self._cancel_v1_context_tasks(context_id)
        self._v1_options_by_context.pop(context_id, None)

        defer_audio_context_close = False
        if first_close and self.audio_context_available(context_id):
            if error:
                await self.append_to_audio_context(context_id, error)
                defer_audio_context_close = context_id == self._turn_context_id
                if defer_audio_context_close:
                    self._v1_failed_contexts.add(context_id)
            if not defer_audio_context_close:
                if emit_stop:
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                await self.remove_audio_context(context_id)

        if self._v1_client:
            self._v1_client.discard_context(context_id)
        self._v1_closed_contexts.discard(context_id)

    async def _fail_all_v1_contexts(self, message: str, category: ErrorCategory) -> None:
        """Fail each active context after the connection becomes unsafe."""
        context_ids = set(self._v1_options_by_context)
        if self._v1_client:
            context_ids.update(self._v1_client.context_ids)
        for context_id in context_ids:
            await self._finish_v1_context(
                context_id,
                error=ErrorFrame(error=message, category=category),
            )

    async def _invalidate_v1_connection(
        self,
        message: str,
        category: ErrorCategory,
        client: RimeWebSocketV1Client,
    ) -> None:
        """Close the connection owned by a failed v1 operation."""
        async with self._v1_connection_lock:
            if client is not self._v1_client or self._v1_invalidating:
                return
            self._v1_invalidating = True
            websocket = self._websocket
            client.invalidate()
            await self._fail_all_v1_contexts(message, category)
            if websocket:
                await websocket.close()

    def _start_v1_keepalive(self, context_id: str) -> None:
        client = self._v1_client
        if client and client.has_context(context_id) and context_id not in self._v1_keepalive_tasks:
            self._v1_keepalive_tasks[context_id] = self.create_task(
                self._v1_keepalive_loop(context_id),
                name=f"{self}::{context_id}::keepalive",
            )

    async def _v1_keepalive_loop(self, context_id: str) -> None:
        try:
            interval = max(0.01, min(1.0, self._stop_frame_timeout_s / 3))
            while (
                context_id not in self._v1_closed_contexts
                and self._v1_client
                and self._v1_client.has_context(context_id)
            ):
                await asyncio.sleep(interval)
                self._refresh_audio_context(context_id)
        finally:
            self._remove_current_v1_task(self._v1_keepalive_tasks, context_id)

    def _start_v1_start_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if client and client.has_context(context_id) and context_id not in self._v1_start_watchdogs:
            self._v1_start_watchdogs[context_id] = self.create_task(
                self._v1_start_watchdog(context_id),
                name=f"{self}::{context_id}::start-watchdog",
            )

    async def _v1_start_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if not client:
            return
        try:
            await client.wait_started(context_id, _V1_START_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._invalidate_v1_connection(
                "Rime v1 did not start a context", ErrorCategory.CONNECTIVITY, client
            )
        finally:
            self._remove_current_v1_task(self._v1_start_watchdogs, context_id)

    def _start_v1_terminal_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if (
            client
            and client.has_context(context_id)
            and context_id not in self._v1_terminal_watchdogs
        ):
            self._v1_terminal_watchdogs[context_id] = self.create_task(
                self._v1_terminal_watchdog(context_id),
                name=f"{self}::{context_id}::terminal-watchdog",
            )

    async def _v1_terminal_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if not client:
            return
        try:
            while client.has_context(context_id):
                await client.wait_activity(context_id, _V1_TERMINAL_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._invalidate_v1_connection(
                "Rime v1 did not finish a context", ErrorCategory.CONNECTIVITY, client
            )
        finally:
            self._remove_current_v1_task(self._v1_terminal_watchdogs, context_id)

    def _start_v1_cancel_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if (
            client
            and client.has_context(context_id)
            and context_id not in self._v1_cancel_watchdogs
        ):
            self._v1_cancel_watchdogs[context_id] = self.create_task(
                self._v1_cancel_watchdog(context_id),
                name=f"{self}::{context_id}::cancel-watchdog",
            )

    async def _v1_cancel_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if not client:
            return
        try:
            if not client.has_context(context_id):
                return
            await client.wait_terminal(context_id, _V1_CANCEL_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._invalidate_v1_connection(
                "Rime v1 did not cancel a context", ErrorCategory.CONNECTIVITY, client
            )
        finally:
            self._remove_current_v1_task(self._v1_cancel_watchdogs, context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the Rime context and stop metrics when the bot is interrupted."""
        await self.stop_all_metrics()
        self._v1_failed_contexts.discard(context_id)
        await self._cancel_v1_context_tasks(context_id, include_terminal=True, include_cancel=False)
        client = self._v1_client
        if client and client.has_context(context_id):
            self._v1_closed_contexts.add(context_id)
            try:
                await client.cancel(context_id, timeout_s=_V1_CANCEL_TIMEOUT_S)
                self._start_v1_cancel_watchdog(context_id)
            except Exception:
                await self._invalidate_v1_connection(
                    "Failed to cancel a Rime v1 context", ErrorCategory.CONNECTIVITY, client
                )
        else:
            self._v1_options_by_context.pop(context_id, None)
            self._v1_closed_contexts.discard(context_id)
            if client:
                client.discard_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_turn_context_completed(self) -> None:
        """Close a failed audio context after its Pipecat turn ends."""
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if context_id is not None and context_id in self._v1_failed_contexts:
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
            self._v1_failed_contexts.discard(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """End input for the context so Rime synthesizes the rest of its text."""
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return

        client = self._v1_client
        if not client or not client.has_context(flush_id):
            return
        try:
            await client.end(flush_id, timeout_s=_V1_TERMINAL_TIMEOUT_S)
            self._start_v1_terminal_watchdog(flush_id)
        except Exception:
            await self._invalidate_v1_connection(
                "Failed to end a Rime v1 context", ErrorCategory.CONNECTIVITY, client
            )

    async def _receive_messages(self):
        """Map validated v1 events into Pipecat audio contexts."""
        client = self._get_v1_client()
        self._v1_receiving_client = client
        try:
            async for event in client.events():
                if isinstance(event, StartedEvent):
                    continue
                if isinstance(event, AudioEvent):
                    if event.context_id in self._v1_closed_contexts:
                        continue
                    if event.audio and self.audio_context_available(event.context_id):
                        await self.append_to_audio_context(
                            event.context_id,
                            TTSAudioRawFrame(
                                audio=event.audio,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                                context_id=event.context_id,
                            ),
                        )
                    continue
                if isinstance(event, ConnectionErrorEvent):
                    await self.push_error(
                        self._v1_error_message(event.kind, event.request_id),
                        category=self._v1_error_category(event.kind),
                    )
                    continue
                if isinstance(event, ContextErrorEvent):
                    await self._finish_v1_context(
                        event.context_id,
                        error=ErrorFrame(
                            error=self._v1_error_message(event.kind, event.request_id),
                            category=self._v1_context_error_category(event.kind),
                        ),
                    )
                    continue
                if isinstance(event, DoneEvent):
                    await self._finish_v1_context(event.context_id)
                    continue
                if isinstance(event, CancelledEvent):
                    await self._finish_v1_context(event.context_id, emit_stop=False)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._invalidate_v1_connection(
                "Rime WebSocket v1 connection failed", self._v1_exception_category(e), client
            )
            raise

    async def _split_v1_sentences(self, text: str) -> list[str]:
        """Split prepared text into the sentence units required by WebSocket v1."""
        aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=TextAggregationMode.SENTENCE
        )
        sentences = [
            self._prepare_text_for_tts(aggregate.text)
            async for aggregate in aggregator.aggregate(text)
        ]
        remaining = await aggregator.flush()
        if remaining and remaining.text:
            sentences.append(self._prepare_text_for_tts(remaining.text))
        return sentences

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send sentence units to the turn's Rime context.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Error frames when the request fails; audio arrives through the
            receive task.
        """
        client = None
        try:
            if context_id in self._v1_failed_contexts:
                return
            if not self._websocket or self._websocket.state is State.CLOSED or not self._v1_client:
                await self._connect()
            client = self._get_v1_client()
            first_text = context_id not in self._v1_options_by_context
            if first_text:
                self._v1_closed_contexts.discard(context_id)
                self._v1_options_by_context[context_id] = self._build_v1_options()
            options = self._v1_options_by_context[context_id]
            for sentence in await self._split_v1_sentences(text):
                await client.send_text(context_id, options, sentence, timeout_s=_V1_START_TIMEOUT_S)
                if not client.has_context(context_id):
                    return
                if first_text:
                    self._start_v1_keepalive(context_id)
                    self._start_v1_start_watchdog(context_id)
                    first_text = False
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            category = self._v1_exception_category(e)
            self._v1_closed_contexts.add(context_id)
            self._v1_options_by_context.pop(context_id, None)
            yield ErrorFrame(error="Rime WebSocket v1 request failed", category=category)
            yield TTSStoppedFrame(context_id=context_id)
            await self.remove_audio_context(context_id)
            self._v1_closed_contexts.discard(context_id)
            if client and category not in (
                ErrorCategory.AUTHENTICATION,
                ErrorCategory.AUTHORIZATION,
                ErrorCategory.INVALID_REQUEST,
            ):
                await self._invalidate_v1_connection(
                    "Rime WebSocket v1 request failed", category, client
                )
