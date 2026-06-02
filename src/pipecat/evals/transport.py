#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport for the eval harness.

A subclass of :class:`~pipecat.transports.websocket.server.WebsocketServerTransport`
that adds eval-only behavior driven by per-connection query flags the harness
sets:

- ``?skip_tts=true`` silences the bot's output for the session (text mode),
  including any on-connect greeting. This is pushed as an
  :class:`~pipecat.frames.frames.LLMConfigureOutputFrame` *before*
  ``on_client_connected`` fires: pipecat processes frames in order, and a bot
  that greets in ``on_client_connected`` queues its greeting there, so a config
  sent afterwards (as a client message) would arrive too late.
- ``?capture_audio=true`` makes the serializer forward the bot's synthesized
  audio to the harness (for ``tts_response`` transcription).
- ``?record=<path>`` records the conversation audio (user + bot) to ``<path>``.
  The recorder is an :class:`~pipecat.processors.audio.audio_buffer_processor.AudioBufferProcessor`
  placed *after* the real output transport (so both input and output audio flow
  through it); ``output()`` returns that composite. Recording starts on connect
  and is flushed synchronously on disconnect (reliable, unlike shutdown).
  Recording is eval-only — the generic transport is untouched.

This transport also keeps one bot alive across a whole eval suite: it overrides
``_on_client_disconnected`` to detach the connection (and flush the recording)
*without* firing the bot's ``on_client_disconnected`` handler, which would cancel
the pipeline.
"""

import wave
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from loguru import logger

from pipecat.frames.frames import LLMConfigureOutputFrame
from pipecat.transports.websocket.server import WebsocketServerTransport

SKIP_TTS_QUERY_PARAM = "skip_tts"
CAPTURE_AUDIO_QUERY_PARAM = "capture_audio"
RECORD_QUERY_PARAM = "record"


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


def _write_wav(path: str, audio: bytes, sample_rate: int, num_channels: int) -> None:
    """Write PCM ``audio`` to a 16-bit WAV at ``path`` (creating parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    logger.info(f"Eval recording saved: {p} ({len(audio)} bytes)")


class EvalWebsocketServerTransport(WebsocketServerTransport):
    """WebSocket server transport used by the eval harness (see the module docstring)."""

    def __init__(self, *args, **kwargs):
        """Initialize the transport and the (lazily built) recording composite."""
        super().__init__(*args, **kwargs)
        self._audio_buffer = None
        self._record_output = None
        self._record_path: str | None = None

    def output(self):
        """Return the output as a recorder composite: ``[real_output, AudioBufferProcessor]``.

        ``self._output`` stays the real output transport, so the transport's
        ``set_client_connection`` reaches it directly (no proxy). The buffer sits
        after it, where both input and output audio flow.
        """
        if self._record_output is None:
            from pipecat.pipeline.pipeline import Pipeline
            from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
            from pipecat.transports.websocket.server import WebsocketServerOutputTransport

            real = WebsocketServerOutputTransport(self, self._params, name=self._output_name)
            self._output = real
            self._audio_buffer = AudioBufferProcessor()
            self._record_output = Pipeline([real, self._audio_buffer])
        return self._record_output

    async def _on_client_connected(self, websocket):
        """Apply per-connection eval flags, then proceed (config before any greeting)."""
        serializer = getattr(self._params, "serializer", None)
        if serializer is not None and hasattr(serializer, "set_capture_audio"):
            serializer.set_capture_audio(_query_flag(websocket, CAPTURE_AUDIO_QUERY_PARAM))

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
        """Flush the recording and detach — but keep the bot alive for the next eval.

        We do *not* call the bot's ``on_client_disconnected`` handler (it cancels
        the pipeline); one bot serves the whole suite. Flushing here is reliable,
        unlike at shutdown where the process may exit before an async save runs.
        """
        if self._audio_buffer is not None:
            if self._record_path and self._audio_buffer.has_audio():
                _write_wav(
                    self._record_path,
                    self._audio_buffer.merge_audio_buffers(),
                    self._audio_buffer.sample_rate,
                    self._audio_buffer.num_channels,
                )
            await self._audio_buffer.stop_recording()
            self._record_path = None

        if self._output is not None:
            await self._output.set_client_connection(None)
