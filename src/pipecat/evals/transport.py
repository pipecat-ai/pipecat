#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport for the eval harness.

A subclass of :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerTransport`
that adds eval-only behavior driven by per-connection query flags the harness
sets:

- ``?skip_tts=true`` silences the bot's output for the session (text mode),
  including any on-connect greeting. This is pushed as an
  :class:`~pipecat.frames.frames.LLMConfigureOutputFrame` *before*
  ``on_client_connected`` fires: pipecat processes frames in order, and a bot
  that greets in ``on_client_connected`` queues its greeting there, so a config
  sent afterwards (as a client message) would arrive too late.
- ``?capture_bot_audio=true`` makes the serializer forward the bot's synthesized
  audio to the harness (for ``tts_response`` transcription).
- ``?record=<path>`` records the conversation audio (user + bot) to ``<path>``.
  The recorder is an :class:`~pipecat.processors.audio.audio_buffer_processor.AudioBufferProcessor`
  placed *after* the real output transport (so both input and output audio flow
  through it); ``output()`` returns that composite. Recording starts on connect
  and is written on disconnect, before the bot's ``on_client_disconnected``
  handler fires (a bot that cancels its pipeline there may exit right after, so
  the write must land first). Recording is eval-only — the generic transport is
  untouched.

The input side runs a **virtual microphone** (:class:`EvalMicrophone`), enabled
per connection by ``?user_audio=true`` (audio-mode scenarios): the harness sends
each user utterance as a few large ``raw-audio`` messages, and the input
transport plays them into the pipeline at real-time cadence (~20ms frames) with
locally generated silence in between — so VADs, turn models, and streaming STTs
see exactly what a live client's mic would produce, without a continuous frame
stream crossing the wire. Text-mode scenarios leave the mic off, so no silence
is ever fed into the bot's STT.

Client disconnects behave as on any transport: the bot's
``on_client_disconnected`` handler fires normally, and whether the pipeline
survives the disconnect is the application's choice. The server itself keeps
running either way, so a bot that opts not to cancel can serve several
sequential eval connections.
"""

import asyncio
import io
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import aiofiles
from loguru import logger
from PIL import Image

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMConfigureOutputFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerInputTransport,
    SingleClientWebsocketServerOutputTransport,
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)

SKIP_TTS_QUERY_PARAM = "skip_tts"
USER_AUDIO_QUERY_PARAM = "user_audio"
CAPTURE_AUDIO_QUERY_PARAM = "capture_bot_audio"
RECORD_QUERY_PARAM = "record"
TRIGGER_DISCONNECT_QUERY_PARAM = "trigger_disconnect"

# One virtual-mic frame per tick — the granularity a live transport delivers and
# what VAD/turn models consume.
AUDIO_CHUNK_MS = 20


def _query_string(websocket) -> str:
    """The connection URL's query string (handles both websockets API versions)."""
    # websockets exposes the request target as ``.path`` (legacy) or
    # ``.request.path`` (newer); both include the query string.
    path = getattr(websocket, "path", None)
    if path is None:
        request = getattr(websocket, "request", None)
        path = getattr(request, "path", "") if request is not None else ""
    return urlsplit(path or "").query


def _query_flag(websocket, name: str) -> bool:
    """Whether the client's connection URL set the boolean query param ``name``."""
    values = parse_qs(_query_string(websocket)).get(name, [])
    return bool(values) and values[0].strip().lower() in ("1", "true", "yes")


def _query_value(websocket, name: str) -> str | None:
    """The string value of query param ``name``, or ``None`` if absent/empty."""
    values = parse_qs(_query_string(websocket)).get(name, [])
    return values[0] if values and values[0] else None


async def _write_wav(path: str, audio: bytes, sample_rate: int, num_channels: int) -> None:
    """Write PCM ``audio`` to a 16-bit WAV at ``path`` (creating parent dirs).

    Encodes the WAV in memory, then writes it to disk without blocking the event
    loop.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(p, "wb") as f:
        await f.write(buffer.getvalue())
    logger.info(f"Eval recording saved: {p} ({len(audio)} bytes)")


class EvalMicrophone:
    """Plays harness-sent utterances into the pipeline at real-time cadence.

    The harness sends each user utterance as a few large ``raw-audio`` messages
    (cheap on the wire — no continuous frame stream to encode and ship). A real
    microphone, though, delivers small frames at real-time pace, and
    timing-sensitive consumers rely on that: VAD start windows, Krisp IP/turn
    models, and turn-detecting STTs all break if a whole utterance floods the
    pipeline at once. This class is the eval transport's virtual microphone:
    every ~20ms tick it pushes one frame — utterance audio when queued, locally
    generated silence otherwise — so the bot hears exactly what a live client
    would produce, including the silence that lets its VAD end each turn.

    Speech that falls behind after a late wake-up is sent back-to-back until
    caught up (the utterance content must stay gap-free); silence never catches
    up (the end-of-turn gap must stay honest).
    """

    def __init__(self, push: Callable[[bytes, int], Awaitable[None]]):
        """Initialize the microphone.

        Args:
            push: Async callable ``(pcm: bytes, sample_rate: int)`` invoked with
                each ~20ms mic frame.
        """
        self._push = push
        self._queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._pcm = b""
        self._rate = 0
        self._offset = 0

    def add_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Queue one utterance (or a piece of one) for real-time playout."""
        self._queue.put_nowait((pcm, sample_rate))

    def reset(self) -> None:
        """Drop queued and in-progress utterance audio (a new eval client starts fresh)."""
        while not self._queue.empty():
            self._queue.get_nowait()
        self._pcm, self._offset = b"", 0

    def _next_chunk(self) -> tuple[bytes, int]:
        """The next ~20ms of queued speech, or ``(b"", 0)`` when there is none."""
        while self._offset >= len(self._pcm):
            try:
                self._pcm, self._rate = self._queue.get_nowait()
                self._offset = 0
            except asyncio.QueueEmpty:
                return b"", 0
        bytes_per_chunk = (self._rate * AUDIO_CHUNK_MS // 1000) * 2
        chunk = self._pcm[self._offset : self._offset + bytes_per_chunk]
        self._offset += bytes_per_chunk
        return chunk, self._rate

    async def run(self, silence_sample_rate: int) -> None:
        """Emit one mic frame per ~20ms tick, forever (cancel to stop).

        Args:
            silence_sample_rate: Sample rate for the generated silence frames
                (the transport's input rate).
        """
        silence = b"\x00\x00" * (silence_sample_rate * AUDIO_CHUNK_MS // 1000)
        tick = AUDIO_CHUNK_MS / 1000
        loop = asyncio.get_running_loop()
        next_send = loop.time()
        while True:
            chunk, rate = self._next_chunk()
            speaking = bool(chunk)
            if not speaking:
                chunk, rate = silence, silence_sample_rate
            await self._push(chunk, rate)
            next_send += tick
            now = loop.time()
            if not speaking:
                # Never burst silence to catch up: re-anchor instead, so the
                # bot's VAD gets the full end-of-turn gap.
                next_send = max(next_send, now)
            if next_send > now:
                await asyncio.sleep(next_send - now)
            # Speech behind schedule loops immediately: catch-up keeps the
            # utterance gap-free after a late wake-up.


class EvalTransportParams(SingleClientWebsocketServerParams):
    """Transport parameters for the eval harness.

    A thin subclass of :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerParams`
    that gives the eval transport its own parameter type. Bots configure the
    ``"eval"`` entry of ``transport_params`` with this class so the eval setup
    reads as eval-specific rather than leaking the underlying WebSocket server
    transport.
    """

    pass


class EvalInputTransport(SingleClientWebsocketServerInputTransport):
    """Input transport that plays a virtual mic and serves the harness's images.

    Audio: the harness sends each user utterance as a few large ``raw-audio``
    messages; instead of letting them flood the VAD path at once, they are queued
    on an :class:`EvalMicrophone` that plays them into the pipeline at real-time
    cadence with locally generated silence in between — a virtual microphone.
    The mic is enabled per connection by the harness's ``?user_audio=true`` flag
    (audio-mode scenarios only — a text-mode scenario must not feed silence into
    the bot's STT, which costs real streaming-STT minutes) and starts when the
    client signals ready (the ``client-ready`` →
    ``InputTransportStartAudioStreamingFrame`` path).

    Images: a function-calling-video bot pushes a ``UserImageRequestFrame``
    upstream when it needs the user's camera image. There is no camera under
    eval, so we serve the image the harness registered for the turn (an
    ``eval-image`` message, stored on the serializer) as a ``UserImageRawFrame``
    — mirroring ``daily/transport.py`` but sourcing the image from the serializer
    instead of a live video frame.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the input transport and its virtual mic."""
        super().__init__(*args, **kwargs)
        self._mic = EvalMicrophone(self._push_mic_frame)
        self._mic_enabled = False
        self._mic_task = None

    async def configure_mic(self, enabled: bool) -> None:
        """Configure the virtual mic for a new eval client.

        Drops any utterance audio a previous client left queued, and enables or
        disables the mic for this connection (the harness sets
        ``?user_audio=true`` for audio-mode scenarios). Disabling stops a mic a
        previous audio-mode client left running, so a following text-mode
        scenario doesn't stream silence into the bot's STT.

        Args:
            enabled: Whether this connection's scenario sends user audio.
        """
        self._mic.reset()
        self._mic_enabled = enabled
        if not enabled:
            await self._stop_mic()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Route utterance audio to the virtual mic; serve image requests."""
        # Harness-sent utterance audio (decoded by the RTVIProcessor upstream):
        # queue it for real-time playout instead of letting it through in a burst.
        if isinstance(frame, InputAudioRawFrame):
            self._mic.add_audio(frame.audio, frame.sample_rate)
            return
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRequestFrame):
            await self._serve_user_image(frame)

    async def _start_audio_in_streaming(self):
        """Start the virtual mic (triggered by the client-ready handshake)."""
        if self._mic_enabled and self._mic_task is None:
            self._mic_task = self.create_task(self._mic.run(self.sample_rate))

    async def stop(self, frame: EndFrame):
        """Stop the virtual mic, then the transport."""
        await self._stop_mic()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the virtual mic, then the transport."""
        await self._stop_mic()
        await super().cancel(frame)

    async def _stop_mic(self) -> None:
        if self._mic_task is not None:
            await self.cancel_task(self._mic_task)
            self._mic_task = None

    async def _push_mic_frame(self, pcm: bytes, sample_rate: int) -> None:
        """Feed one mic frame through the normal input path (filters, VAD)."""
        await self.push_audio_frame(
            InputAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=1)
        )

    async def _serve_user_image(self, request: UserImageRequestFrame) -> None:
        serializer = getattr(self._params, "serializer", None)
        image = None
        if serializer is not None and hasattr(serializer, "get_user_image"):
            image = serializer.get_user_image()
        if image is None:
            logger.warning(f"{self}: UserImageRequestFrame but no eval image registered")
            return
        data, _fmt = image
        # The harness sends the image encoded over the wire, but a real camera
        # transport pushes raw frames -- so decode it to raw RGB here and serve a
        # genuine ``UserImageRawFrame``. The LLM context re-encodes raw frames to
        # JPEG anyway, and consumers that decode directly (e.g. a local vision
        # model doing ``Image.frombytes``) need the raw pixels and real size.
        decoded = await asyncio.to_thread(lambda: Image.open(io.BytesIO(data)).convert("RGB"))
        await self.push_frame(
            UserImageRawFrame(
                image=decoded.tobytes(),
                size=decoded.size,
                format="RGB",
                user_id=request.user_id,
                text=request.text,
                append_to_context=request.append_to_context,
                request=request,
            )
        )


class EvalOutputTransport(SingleClientWebsocketServerOutputTransport):
    """Output transport used by the eval harness.

    The eval harness sends the bot's output over the same WebSocket connection
    as any client, so this currently adds no behavior beyond
    :class:`~pipecat.transports.websocket.server.SingleClientWebsocketServerOutputTransport`.
    It exists for naming symmetry with :class:`EvalInputTransport` and as a hook
    for any future eval-specific output behavior.
    """

    pass


class EvalTransport(SingleClientWebsocketServerTransport):
    """WebSocket server transport used by the eval harness (see the module docstring)."""

    def __init__(self, *args, **kwargs):
        """Initialize the transport and the (lazily built) recording composite."""
        super().__init__(*args, **kwargs)
        self._audio_buffer = None
        self._record_output = None
        self._record_path: str | None = None

    def input(self) -> SingleClientWebsocketServerInputTransport:
        """Return an input transport that can serve harness-provided images."""
        if not self._input:
            self._input = EvalInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self):
        """Return the output as a recorder composite: ``[real_output, AudioBufferProcessor]``.

        ``self._output`` stays the real output transport, so the transport's
        ``set_client_connection`` reaches it directly (no proxy). The buffer sits
        after it, where both input and output audio flow.
        """
        if self._record_output is None:
            from pipecat.pipeline.pipeline import Pipeline
            from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

            real = EvalOutputTransport(self, self._params, name=self._output_name)
            self._output = real
            self._audio_buffer = AudioBufferProcessor()
            self._record_output = Pipeline([real, self._audio_buffer])
        return self._record_output

    async def _on_client_connected(self, websocket):
        """Apply per-connection eval flags, then proceed (config before any greeting)."""
        serializer = getattr(self._params, "serializer", None)
        if serializer is not None and hasattr(serializer, "set_capture_audio"):
            serializer.set_capture_audio(_query_flag(websocket, CAPTURE_AUDIO_QUERY_PARAM))

        # A new eval client starts a fresh conversation: drop any utterance audio
        # a previous client left queued, and enable the virtual mic only for
        # audio-mode scenarios (?user_audio=true).
        if isinstance(self._input, EvalInputTransport):
            await self._input.configure_mic(_query_flag(websocket, USER_AUDIO_QUERY_PARAM))

        # Start recording as soon as the client connects so the bot's first audio
        # (e.g. a greeting) is captured. Flushed on disconnect.
        self._record_path = _query_value(websocket, RECORD_QUERY_PARAM)
        if self._audio_buffer is not None and self._record_path:
            await self._audio_buffer.start_recording()

        if self._input is not None and _query_flag(websocket, SKIP_TTS_QUERY_PARAM):
            logger.debug(f"{self}: eval client requested skip_tts; configuring LLM output")
            await self._input.push_frame(LLMConfigureOutputFrame(skip_tts=True))

        await super()._on_client_connected(websocket)

    async def _on_client_disconnected(self, websocket):
        """Flush the recording, then handle the disconnect normally."""
        if self._audio_buffer is not None:
            if self._record_path and self._audio_buffer.has_audio():
                await _write_wav(
                    self._record_path,
                    self._audio_buffer.merge_audio_buffers(),
                    self._audio_buffer.sample_rate,
                    self._audio_buffer.num_channels,
                )
            await self._audio_buffer.stop_recording()
            self._record_path = None

        await super()._on_client_disconnected(websocket)

    async def _emit_client_disconnected(self, websocket):
        """Fire ``on_client_disconnected`` only when the harness asks for it.

        Bots often cancel their pipeline in ``on_client_disconnected``, so the
        event is suppressed by default to avoid that between eval scenarios. The
        harness sets ``?trigger_disconnect=true`` (via a scenario's
        ``trigger_disconnect`` field or ``pipecat eval run --trigger-disconnect``)
        to exercise the bot's disconnect path. Independent of ``--stop-bot``,
        which tears the bot down reliably via ``eval-cancel``.
        """
        if _query_flag(websocket, TRIGGER_DISCONNECT_QUERY_PARAM):
            await super()._emit_client_disconnected(websocket)
