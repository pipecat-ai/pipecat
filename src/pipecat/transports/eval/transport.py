#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval transport for Pipecat behavioral evaluations.

This transport runs a local WebSocket server that simulates a connected client.
It accepts scripted user input as JSON messages and emits high-level semantic
events back to the connected harness (user_started_speaking,
user_stopped_speaking, llm_started, llm_response, interruption, tool_call,
error). Designed for fast, deterministic behavioral evaluations of a pipeline
without invoking real STT or full audio I/O.

Two modes, controlled by the per-eval ``tts:`` field (default True):

- TTS on (default): the bot pipeline runs end-to-end including TTS. The
  eval transport drops the audio bytes but paces ``write_audio_frame``
  at real-time (``audio_chunk_size / sample_rate`` per chunk) so the bot
  behaves as if a real audio sink were consuming output — gives
  interruption tests a realistic window in which to barge in.
- TTS off (``{"type": "settings", "tts": false}``): pushes
  ``LLMConfigureOutputFrame(skip_tts=True)`` downstream so the LLM bypasses
  TTS entirely. No audio is generated, no API calls, no pacing — the
  harness runs as fast as the bot can produce tokens.

Selected via ``-t eval`` in the development runner.
"""

import asyncio
import base64
import datetime
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesUpdateFrame,
    OutputAudioRawFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from websockets.asyncio.server import serve as websocket_serve
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the eval transport, you need to `pip install pipecat-ai[websocket]`."
    )
    raise ImportError(f"Missing module: {e}") from e


class EvalTransportParams(TransportParams):
    """Configuration parameters for the eval transport.

    Parameters:
        verbose: When True, also emits a ``frame`` event for every frame the
            transport observes (in addition to the curated semantic events).
            Useful for debugging failing scenarios; off by default.
        keep_alive: When True (default), the transport suppresses the
            ``on_client_disconnected`` event handler on the bot side so the
            bot stays up across multiple harness connections (required for
            running several evals against a single bot process). The
            ``on_client_connected`` handler always fires regardless — bots
            typically use it to kick off the conversation (greeting,
            initial context, etc.) and that needs to happen per-eval. Set
            False to also fire ``on_client_disconnected`` (typical bot
            behavior is to tear down on disconnect).
        tts: When True (default), the bot's TTS runs normally and the
            transport paces ``write_audio_frame`` at real-time to simulate
            an audio sink. When False (or toggled via a
            ``{"type": "settings", "tts": false}`` message), the transport
            pushes ``LLMConfigureOutputFrame(skip_tts=True)`` so the LLM
            bypasses TTS entirely — no audio, no TTS API calls, no pacing.
    """

    # Both default to True since the eval transport's job is to drive the
    # full pipeline. audio_in_enabled in particular matters when stt is on
    # — STT services need the input audio path active to process the
    # InputAudioRawFrames we inject from user_audio messages.
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True

    verbose: bool = False
    keep_alive: bool = True
    tts: bool = True


class EvalTransportCallbacks(BaseModel):
    """Callback functions for eval transport server events.

    Parameters:
        on_client_connected: Called when the harness connects.
        on_client_disconnected: Called when the harness disconnects.
        on_ready: Called when the harness has signalled it's ready to consume events.
    """

    on_client_connected: Callable[[Any], Awaitable[None]]
    on_client_disconnected: Callable[[Any], Awaitable[None]]
    on_ready: Callable[[], Awaitable[None]]


class EvalInputTransport(BaseInputTransport):
    """Input side of the eval transport.

    Hosts the WebSocket server, accepts a single harness connection, parses
    incoming JSON messages, and injects the corresponding frame sequence
    downstream. Two input paths:

    - ``user_input`` (text): pushes ``UserStartedSpeakingFrame`` →
      ``TranscriptionFrame`` → ``UserStoppedSpeakingFrame`` and emits the
      matching user_started/stopped_speaking events directly with the
      transcript attached. Bypasses STT.
    - ``user_audio`` (base64 PCM): chunks the audio into ~20ms
      ``InputAudioRawFrame`` slices and pushes via ``push_audio_frame`` so
      the bot's VAD/STT runs for real. Trailing silence is appended so
      VAD detects end-of-speech. No user_started/stopped_speaking events
      are emitted in this path — the assertion of interest is
      ``llm_response`` (if the bot understood the audio, the response
      reflects it).
    """

    def __init__(
        self,
        transport: "EvalTransport",
        host: str,
        port: int,
        params: EvalTransportParams,
        callbacks: EvalTransportCallbacks,
        **kwargs,
    ):
        """Initialize the test input transport.

        Args:
            transport: The parent transport instance.
            host: Host address to bind the WebSocket server to.
            port: Port number to bind the WebSocket server to.
            params: Configuration parameters.
            callbacks: Event callbacks invoked by the server.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: Any = None
        self._server_task: asyncio.Task | None = None
        self._stop_server_event = asyncio.Event()
        self._initialized = False
        self._t0 = time.monotonic()

    def _t(self) -> int:
        """Monotonic ms since the transport was constructed."""
        return int((time.monotonic() - self._t0) * 1000)

    @property
    def websocket(self) -> Any:
        """The currently connected harness WebSocket, or None."""
        return self._websocket

    async def start(self, frame: StartFrame):
        """Start the WebSocket server."""
        await super().start(frame)

        if self._initialized:
            return
        self._initialized = True

        if not self._server_task:
            self._server_task = self.create_task(self._server_task_handler())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the WebSocket server."""
        await super().stop(frame)
        self._stop_server_event.set()
        if self._server_task:
            await self._server_task
            self._server_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the WebSocket server immediately."""
        await super().cancel(frame)
        if self._server_task:
            await self.cancel_task(self._server_task)
            self._server_task = None

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def _server_task_handler(self):
        """Run the WebSocket server until stopped."""
        logger.info(f"Starting eval transport WebSocket server on {self._host}:{self._port}")
        async with websocket_serve(self._client_handler, self._host, self._port):
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket):
        """Handle a harness connection: parse messages and inject frames."""
        logger.info(f"Harness connected from {websocket.remote_address}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one harness allowed at a time; replacing connection")
        self._websocket = websocket

        await self._callbacks.on_client_connected(websocket)

        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.error(f"Eval transport: invalid JSON from harness: {e}")
                    continue
                await self._handle_message(msg)
        except Exception as e:
            logger.error(
                f"Eval transport: exception in client handler: {e.__class__.__name__} ({e})"
            )

        await self._callbacks.on_client_disconnected(websocket)
        self._websocket = None
        logger.info("Harness disconnected")

    async def _handle_message(self, msg: dict):
        """Dispatch a parsed JSON message from the harness."""
        msg_type = msg.get("type")
        if msg_type == "user_input":
            await self._inject_user_turn(msg.get("text", ""))
        elif msg_type == "user_audio":
            await self._inject_user_audio(msg)
        elif msg_type == "reset":
            await self._reset_context(msg.get("messages") or [])
        elif msg_type == "settings":
            await self._apply_settings(msg)
        elif msg_type == "ready":
            await self._callbacks.on_ready()
        else:
            logger.warning(f"Eval transport: unknown message type: {msg_type!r}")

    async def _apply_settings(self, msg: dict) -> None:
        """Apply per-eval runtime settings from a ``settings`` message.

        ``tts`` (bool, default True): pushes
        ``LLMConfigureOutputFrame(skip_tts=not tts)`` downstream. When
        tts=False, the LLM bypasses TTS entirely.
        """
        if "tts" in msg:
            await self.push_frame(LLMConfigureOutputFrame(skip_tts=not bool(msg["tts"])))

    async def _reset_context(self, messages: list):
        """Push LLMMessagesUpdateFrame downstream to reset the bot's LLM context.

        The bot's context aggregator (if any) replaces its messages with the
        supplied list. Bots without an aggregator silently let the frame flow
        through. An empty list clears the context entirely.
        """
        await self.push_frame(LLMMessagesUpdateFrame(messages=list(messages)))

    async def _inject_user_turn(self, text: str):
        """Simulate a complete user utterance by pushing the canonical frame sequence.

        Emits the user-side semantic events here (rather than from the output
        side) because SystemFrames don't go through the data queue and would
        otherwise race ahead of the TranscriptionFrame.
        """
        await self._emit({"type": "user_started_speaking", "t": self._t()})
        await self.push_frame(UserStartedSpeakingFrame())
        await self.push_frame(
            TranscriptionFrame(
                text=text,
                user_id="harness",
                timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                finalized=True,
            )
        )
        await self.push_frame(UserStoppedSpeakingFrame())
        await self._emit({"type": "user_stopped_speaking", "t": self._t(), "transcript": text})

    async def _inject_user_audio(self, msg: dict):
        """Push base64-encoded PCM audio into the pipeline so the bot's STT processes it.

        Expects ``msg`` to contain:
            ``audio``         base64-encoded raw PCM (16-bit, little-endian, mono)
            ``sample_rate``   integer (Hz)
            ``chunk_ms``      optional, defaults to 20 — audio is split into
                              chunks of this duration before being pushed.
                              Matches typical microphone capture cadence so
                              VAD/STT services behave naturally.
            ``trailing_silence_ms`` optional, defaults to 500 — silence
                              appended after the audio so the bot's VAD
                              detects end-of-speech.

        Pushes via ``push_audio_frame`` so the base input transport's
        queue/VAD/filter path runs as it would for real microphone input.
        Does NOT emit user_started_speaking or user_stopped_speaking
        events — assertions in audio mode should target ``llm_response``,
        since that's where the bot's understanding of the input shows up.
        """
        raw = msg.get("audio")
        sample_rate = msg.get("sample_rate")
        if not raw or not sample_rate:
            logger.error("Eval transport: user_audio message missing audio or sample_rate")
            return
        try:
            audio_bytes = base64.b64decode(raw)
        except Exception as e:
            logger.error(f"Eval transport: failed to decode user_audio base64: {e}")
            return

        sample_rate = int(sample_rate)
        chunk_ms = int(msg.get("chunk_ms", 20))
        trailing_ms = int(msg.get("trailing_silence_ms", 500))

        # 2 bytes per sample (16-bit), mono.
        bytes_per_chunk = (sample_rate * chunk_ms // 1000) * 2

        for offset in range(0, len(audio_bytes), bytes_per_chunk):
            chunk = audio_bytes[offset : offset + bytes_per_chunk]
            await self.push_audio_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1)
            )

        if trailing_ms > 0:
            silence = b"\x00\x00" * (sample_rate * chunk_ms // 1000)
            chunks_of_silence = trailing_ms // chunk_ms
            for _ in range(chunks_of_silence):
                await self.push_audio_frame(
                    InputAudioRawFrame(audio=silence, sample_rate=sample_rate, num_channels=1)
                )

    async def _emit(self, event: dict):
        """Send a semantic event to the connected harness, if any."""
        if not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps(event))
        except Exception as e:
            logger.error(f"Eval transport: exception sending event: {e.__class__.__name__} ({e})")


class EvalOutputTransport(BaseOutputTransport):
    """Output side of the eval transport.

    Inspects every frame arriving from upstream and translates relevant ones
    into semantic JSON events on the harness WebSocket. ``LLMFullResponseStart``
    / ``LLMFullResponseEnd`` mark the boundaries of a bot response;
    ``TextFrame`` subclasses (``LLMTextFrame`` when TTS is skipped,
    ``TTSTextFrame`` when TTS runs) flow between them and are aggregated
    into the ``llm_response.text`` payload.

    Audio pacing: when TTS is active (the bot pipeline produces audio
    frames), ``write_audio_frame`` paces each chunk at real-time
    (``audio_chunk_size / sample_rate`` seconds per chunk) so the bot
    behaves as if a real audio sink were consuming output — gives
    interruption tests a realistic window in which to barge in. When TTS
    is skipped (``LLMConfigureOutputFrame(skip_tts=True)`` observed),
    no audio frames flow and no pacing happens.
    """

    def __init__(
        self,
        transport: "EvalTransport",
        params: EvalTransportParams,
        **kwargs,
    ):
        """Initialize the eval output transport.

        Args:
            transport: The parent transport instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._params = params

        self._websocket: Any = None

        # Toggled by LLMConfigureOutputFrame flowing through process_frame.
        # When True, write_audio_frame returns immediately (no real audio
        # will flow anyway since the LLM is bypassing TTS).
        self._skip_tts = not params.tts

        # Real-time audio pacing — computed at start() once sample_rate is
        # known. Each chunk's write blocks for audio_chunk_size/sample_rate
        # seconds (its natural playback duration).
        self._send_interval = 0.0
        self._next_send_time = 0.0

        # Response aggregation state.
        self._response_text: list[str] = []
        self._in_response = False
        self._llm_started_t: float | None = None
        self._tool_calls: dict[str, dict] = {}

        self._initialized = False
        self._t0 = time.monotonic()

    def _t(self) -> int:
        """Monotonic ms since the transport was constructed."""
        return int((time.monotonic() - self._t0) * 1000)

    async def set_client_connection(self, websocket: Any):
        """Set or clear the active harness WebSocket connection."""
        self._websocket = websocket

    async def start(self, frame: StartFrame):
        """Start the output transport."""
        await super().start(frame)

        if self._initialized:
            return
        self._initialized = True

        # Real-time pacing: chunk duration in seconds. The base class
        # populates audio_chunk_size and sample_rate by the time start()
        # runs.
        self._send_interval = self.audio_chunk_size / self.sample_rate
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport."""
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport."""
        await super().cancel(frame)

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Translate observed frames into semantic events for the harness."""
        await super().process_frame(frame, direction)

        if self._params.verbose:
            await self._emit({"type": "frame", "frame": frame.__class__.__name__, "t": self._t()})

        # User-side events (user_started_speaking, user_stopped_speaking) are
        # emitted from the input side directly — SystemFrames bypass the data
        # queue and would otherwise race ahead of the TranscriptionFrame here.

        if isinstance(frame, LLMConfigureOutputFrame):
            self._skip_tts = frame.skip_tts
        elif isinstance(frame, InterruptionFrame):
            offset = (
                int((time.monotonic() - self._llm_started_t) * 1000)
                if self._llm_started_t is not None
                else None
            )
            self._next_send_time = 0  # reset audio pacing
            # Cancel the in-flight response. If the LLM's End frame for the
            # cancelled response arrives later (timing-dependent), the
            # _in_response gate will prevent emitting a spurious empty
            # llm_response. A new LLMFullResponseStart will re-arm the flag.
            self._in_response = False
            self._response_text = []
            self._llm_started_t = None
            await self._emit(
                {"type": "interruption", "t": self._t(), "into_bot_response_ms": offset}
            )
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._response_text = []
            self._in_response = True
            self._llm_started_t = time.monotonic()
            await self._emit({"type": "llm_started", "t": self._t()})
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._in_response:
                text = "".join(self._response_text)
                self._response_text = []
                self._in_response = False
                self._llm_started_t = None
                # Skip the emission if no text was produced — happens when
                # the LLM response cycle fires Start/End around a no-op
                # (cancelled or zero-token response). The harness can't do
                # anything useful with an empty llm_response and it would
                # spuriously match expectations meant for the next response.
                if text:
                    await self._emit({"type": "llm_response", "t": self._t(), "text": text})
        elif isinstance(frame, TextFrame) and self._in_response:
            # Catches LLMTextFrame when TTS is skipped and TTSTextFrame when
            # TTS runs (TTSTextFrame extends AggregatedTextFrame extends
            # TextFrame). TranscriptionFrame also extends TextFrame, but it
            # fires before LLMFullResponseStartFrame, so _in_response gates
            # it out.
            self._response_text.append(frame.text)
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._tool_calls[frame.tool_call_id] = {
                "name": frame.function_name,
                "args": frame.arguments,
                "started_t": self._t(),
            }
        elif isinstance(frame, FunctionCallResultFrame):
            call = self._tool_calls.pop(frame.tool_call_id, {})
            started_t = call.get("started_t", self._t())
            await self._emit(
                {
                    "type": "tool_call",
                    "t": self._t(),
                    "tool_call_id": frame.tool_call_id,
                    "name": frame.function_name,
                    "args": frame.arguments,
                    "result": frame.result,
                    "cancelled": False,
                    "duration_ms": self._t() - started_t,
                }
            )
        elif isinstance(frame, FunctionCallCancelFrame):
            call = self._tool_calls.pop(frame.tool_call_id, {})
            started_t = call.get("started_t", self._t())
            await self._emit(
                {
                    "type": "tool_call",
                    "t": self._t(),
                    "tool_call_id": frame.tool_call_id,
                    "name": frame.function_name,
                    "args": call.get("args"),
                    "result": None,
                    "cancelled": True,
                    "duration_ms": self._t() - started_t,
                }
            )
        elif isinstance(frame, ErrorFrame):
            await self._emit(
                {
                    "type": "error",
                    "t": self._t(),
                    "error": frame.error,
                    "fatal": frame.fatal,
                }
            )

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Drop the audio bytes but pace at real-time playback rate.

        When TTS is skipped, no audio flows through this method so the
        ``_skip_tts`` short-circuit isn't strictly necessary, but it's a
        cheap safety net and makes the intent explicit.
        """
        del frame
        if self._skip_tts:
            return True
        await self._pace_audio()
        return True

    async def _pace_audio(self) -> None:
        """Sleep so audio chunks are consumed at real-time playback rate.

        Carries a ``_next_send_time`` cursor so timing doesn't drift across
        many chunks. Reset to 0 on InterruptionFrame so the next chunk
        sends immediately.
        """
        now = time.monotonic()
        sleep_duration = max(0.0, self._next_send_time - now)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval

    async def _emit(self, event: dict):
        """Send a semantic event to the connected harness, if any."""
        if not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps(event))
        except Exception as e:
            logger.error(f"Eval transport: exception sending event: {e.__class__.__name__} ({e})")


class EvalTransport(BaseTransport):
    """Eval transport for pipeline behavioral evaluations.

    Selected via ``-t eval`` in examples. Hosts a local WebSocket server,
    accepts a single harness connection and scripted ``user_input`` messages,
    and emits high-level semantic events (``user_started_speaking``,
    ``user_stopped_speaking``, ``llm_started``, ``llm_response``,
    ``interruption``, ``tool_call``, ``error``) that a scenario runner can
    assert against.

    Designed for fast, deterministic tests of pipeline behavior — does not
    require STT/TTS/audio I/O and has no heavy network dependencies. For
    end-to-end audio-quality testing, use the evals pipeline instead.

    Event handlers available:

    - on_client_connected(transport, websocket): Harness connected
    - on_client_disconnected(transport, websocket): Harness disconnected
    - on_ready(transport): Harness signalled readiness to consume events

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, websocket):
            ...
    """

    def __init__(
        self,
        params: EvalTransportParams | None = None,
        host: str = "localhost",
        port: int = 7860,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the eval transport.

        Args:
            params: Configuration parameters. Defaults to ``EvalTransportParams()``.
            host: Host address to bind the WebSocket server to. Defaults to ``"localhost"``.
            port: Port number to bind the WebSocket server to. Defaults to ``7860``.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._host = host
        self._port = port
        self._params = params or EvalTransportParams()

        self._callbacks = EvalTransportCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_ready=self._on_ready,
        )

        self._input: EvalInputTransport | None = None
        self._output: EvalOutputTransport | None = None

        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_ready")

    def input(self) -> EvalInputTransport:
        """Get the input transport processor."""
        if not self._input:
            self._input = EvalInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> EvalOutputTransport:
        """Get the output transport processor."""
        if not self._output:
            self._output = EvalOutputTransport(self, self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket):
        """Share the WebSocket with the output side and notify handlers.

        ``on_client_connected`` always fires regardless of ``keep_alive`` —
        bots typically use it to kick off conversations (greeting on
        connect, pushing initial context messages, etc.) and that's exactly
        what evals need on each fresh connection.
        """
        if self._output:
            await self._output.set_client_connection(websocket)
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        """Clear the WebSocket on the output side and (optionally) notify handlers.

        Suppressed when ``keep_alive`` is True so bots that tear themselves
        down in ``on_client_disconnected`` (typical: ``worker.cancel()``)
        stay up across multiple harness connections.
        """
        if self._output:
            await self._output.set_client_connection(None)
        if not self._params.keep_alive:
            await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_ready(self):
        """Notify handlers that the harness is ready."""
        await self._call_event_handler("on_ready")
