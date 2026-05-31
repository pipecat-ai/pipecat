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

Two pacing modes:

- Real-time (default): pacing happens in ``process_frame`` on each
  ``TextFrame`` chunk, sleeping for ``len(text) / chars_per_second``
  seconds. Gives interruption tests a realistic window in which to barge in.
- Fast: triggered by ``{"type": "settings", "fast": true}`` from the
  harness. The transport pushes ``LLMConfigureOutputFrame(skip_tts=True)``
  downstream so the LLM produces text without invoking TTS at all — no
  audio, no API calls, no pacing.

Selected via ``-t eval`` in the development runner.
"""

import asyncio
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

# Rough speaking rate used when pacing TextFrame chunks in real-time mode —
# 12.5 chars/sec ≈ 150 WPM at ~5 chars/word average. Sleep duration for a
# chunk of N characters is N / CHARS_PER_SECOND. Override per-deployment by
# setting EvalTransportParams.chars_per_second.
DEFAULT_CHARS_PER_SECOND = 12.5

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
        keep_alive: When True (default), the transport does **not** invoke
            ``on_client_connected`` / ``on_client_disconnected`` event
            handlers on the bot side. The bot stays up across multiple harness
            connections — required for running several evals against a single
            bot process. Set False to get per-connection events (e.g. if your
            bot tears itself down on disconnect).
        fast: When True at startup (or toggled via a ``settings`` message),
            the transport pushes ``LLMConfigureOutputFrame(skip_tts=True)``
            so the LLM produces text without invoking TTS — no audio, no API
            calls, no pacing. Text-based pacing is also skipped. Use for
            text-only evals where interruption timing doesn't matter and you
            want results as fast as the bot can produce them.
        chars_per_second: Speaking rate used when pacing ``TextFrame`` chunks
            in real-time mode (default ≈ 150 WPM). Adjust per-deployment if
            your TTS voice is noticeably faster or slower than typical
            English speech.
    """

    # Eval mode doesn't actually play audio anywhere, so we override the
    # silence-related defaults from TransportParams: no continuous silence
    # injection while the audio queue is empty (saves needless write_audio_frame
    # calls in fast mode where TTS is skipped entirely), and no end-of-frame
    # silence either. write_audio_frame is a no-op regardless of mode.
    audio_out_enabled: bool = True
    audio_out_auto_silence: bool = False
    audio_out_end_silence_secs: int = 0

    verbose: bool = False
    keep_alive: bool = True
    fast: bool = False
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND


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
    downstream (e.g. a ``user_input`` message becomes
    ``UserStartedSpeakingFrame`` → ``TranscriptionFrame`` →
    ``UserStoppedSpeakingFrame``). Also emits the user-side semantic events
    (``user_started_speaking``, ``user_stopped_speaking``) directly here —
    SystemFrames bypass the data queue and arrive at the output side before
    the TranscriptionFrame, so the output side can't reliably attach the
    transcript by the time it sees ``UserStoppedSpeakingFrame``.
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

        ``fast`` (bool): when True, pushes ``LLMConfigureOutputFrame(skip_tts=True)``
        downstream so the LLM produces text without invoking TTS, and tells
        the output side to skip text pacing. When False, pushes
        ``skip_tts=False`` to re-enable TTS for the next response.
        """
        if "fast" in msg:
            fast = bool(msg["fast"])
            await self.push_frame(LLMConfigureOutputFrame(skip_tts=fast))
            if self._transport._output is not None:
                self._transport._output.set_fast(fast)

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
    ``TextFrame`` subclasses (``LLMTextFrame`` in fast mode where TTS is
    skipped, ``TTSTextFrame`` in normal mode) flow between them and are
    aggregated into the ``llm_response.text`` payload.

    In real-time mode, each ``TextFrame`` chunk is paced with a sleep
    proportional to its character count — simulating natural speaking speed
    so interruption tests have a window in which to barge in. The sleep is
    interruptible: an ``InterruptionFrame`` signals an event that breaks any
    in-flight pacing immediately.
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

        self._fast = params.fast
        self._chars_per_second = params.chars_per_second

        # Response aggregation state.
        self._response_text: list[str] = []
        self._in_response = False
        self._llm_started_t: float | None = None
        self._tool_calls: dict[str, dict] = {}

        # Interruption signal: an asyncio.Event the InterruptionFrame handler
        # sets to short-circuit any in-flight text pacing sleep.
        self._interrupt_event = asyncio.Event()

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

        if isinstance(frame, InterruptionFrame):
            offset = (
                int((time.monotonic() - self._llm_started_t) * 1000)
                if self._llm_started_t is not None
                else None
            )
            self._interrupt_event.set()
            await self._emit(
                {"type": "interruption", "t": self._t(), "into_bot_response_ms": offset}
            )
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._response_text = []
            self._in_response = True
            self._llm_started_t = time.monotonic()
            self._interrupt_event.clear()
            await self._emit({"type": "llm_started", "t": self._t()})
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._in_response:
                text = "".join(self._response_text)
                self._response_text = []
                self._in_response = False
                self._llm_started_t = None
                await self._emit({"type": "llm_response", "t": self._t(), "text": text})
        elif isinstance(frame, TextFrame) and self._in_response:
            # Catches LLMTextFrame in fast mode (TTS skipped) and TTSTextFrame
            # in normal mode (TTSTextFrame extends AggregatedTextFrame extends
            # TextFrame). TranscriptionFrame also extends TextFrame, but it
            # fires before LLMFullResponseStartFrame, so _in_response gates
            # it out.
            self._response_text.append(frame.text)
            await self._pace_text(frame.text)
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

    async def _pace_text(self, text: str) -> None:
        """Sleep proportional to text length to simulate natural speaking speed.

        Returns immediately in fast mode or if the interruption event fires.
        Computed as ``len(text) / chars_per_second`` (≈ 150 WPM at the
        default rate). Interruptible via :meth:`set_fast` toggle or any
        ``InterruptionFrame`` arrival (which sets ``self._interrupt_event``).
        """
        if self._fast or not text:
            return
        target_s = len(text) / self._chars_per_second
        try:
            await asyncio.wait_for(self._interrupt_event.wait(), timeout=target_s)
        except TimeoutError:
            pass

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Drop audio bytes; pacing happens on TextFrame in process_frame."""
        del frame
        return True

    def set_fast(self, fast: bool) -> None:
        """Toggle fast mode (skip text pacing).

        Called by the parent transport when a ``settings`` message arrives.
        """
        self._fast = fast

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
        """Share the WebSocket with the output side and (optionally) notify handlers."""
        if self._output:
            await self._output.set_client_connection(websocket)
        # When keep_alive is True, suppress the connection event so bots that
        # tear themselves down in on_client_disconnected stay up across evals.
        if not self._params.keep_alive:
            await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        """Clear the WebSocket on the output side and (optionally) notify handlers."""
        if self._output:
            await self._output.set_client_connection(None)
        if not self._params.keep_alive:
            await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_ready(self):
        """Notify handlers that the harness is ready."""
        await self._call_event_handler("on_ready")
