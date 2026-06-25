#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Client-side WebSocket transports for the eval harness.

The harness drives the bot as an RTVI client. A real transport carries a
continuous real-time audio stream in both directions — speech while a party talks,
silence otherwise — and VAD, turn detection, and streaming STT all rely on that
cadence. The eval bot's WebSocket transport instead sends and receives audio only
while a TTS is producing, with gaps during pauses, so both edges reshape it into a
continuous stream:

- :class:`EvalHarnessOutputTransport` (send side): the user TTS's audio is enqueued
  and a real-time task ships one ~40ms frame every tick (queued audio when
  available, silence when idle), so the bot receives a continuous stream and its
  stock input handles it directly.
- :class:`EvalHarnessInputTransport` (receive side): the bot's audio arrives in
  bursts with gaps; it is buffered and re-emitted at the same steady cadence, so
  the harness's VAD + smart-turn see speech-then-silence and judge the bot's real
  turn boundaries — instead of the VAD's idle timeout force-stopping a turn at a
  pause and splitting it mid-sentence.

Pacing on the send side also makes the harness-side recording faithful: the
:class:`~pipecat.processors.audio.audio_buffer_processor.AudioBufferProcessor`
places audio by wall-clock, so a burst would collapse the user's turn into a blip;
and the receive side's gap-filling keeps the bot track continuous too.

:class:`EvalHarnessTransport` is the RTVI client transport the harness builds; it
differs from :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
only in returning these two stream-shaping transports.

The harness's bot-audio VAD broadcasts interruptions when the bot speaks; those are
stopped at the harness sink before they reach the output transport (see
``_BotFrameSink``), so the paced user audio is never flushed mid-utterance.
"""

import asyncio
import time

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.transports.websocket.client import (
    WebsocketClientInputTransport,
    WebsocketClientOutputTransport,
)
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport

# One audio frame per tick, on both edges. 40ms matches the bot's output chunking
# (``audio_out_10ms_chunks`` defaults to 4), so the receive side re-emits the bot's
# chunks 1:1 rather than splitting them, and it's still fine-grained enough for the
# VAD/turn models (well under their start/stop windows).
FRAME_S = 0.05


class EvalHarnessOutputTransport(WebsocketClientOutputTransport):
    """Streams the user audio to the bot as a continuous real-time stream.

    The default output transport sends each audio frame as it arrives (paced to
    real time, but with no audio in between), so the bot would see a whole utterance
    with no trailing silence — VADs and turn detectors need that silence to end a
    turn. This transport instead enqueues the user TTS's audio and a real-time task
    emits one ~40ms frame every tick: the next queued chunk when there is one,
    silence otherwise. The bot then receives a continuous stream and its stock input
    handles it directly.

    The stream runs only when audio output is enabled (audio-mode scenarios); a
    text-mode scenario sends nothing, so no silence is ever fed to the bot's STT.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the transport and its (lazily started) send stream."""
        super().__init__(*args, **kwargs)
        self._pending = bytearray()
        self._send_task = None

    async def start(self, frame: StartFrame):
        """Start the transport and, in audio mode, the real-time send stream."""
        await super().start(frame)
        if self._params.audio_out_enabled and self._send_task is None:
            self._send_task = self.create_task(self._send_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the send stream, then the transport."""
        await self._cancel_send_task()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the send stream, then the transport."""
        await self._cancel_send_task()
        await super().cancel(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Enqueue the user audio; the send task paces it onto the wire.

        Returns False so the media sender does *not* push this (un-paced) frame
        downstream: the media sender drains its queue in a burst, which would
        collapse the user's turn in the harness recording. The send task pushes the
        paced frames downstream instead (see :meth:`_send_task_handler`).
        """
        if self._session.is_closing or not self._session.is_connected:
            return False
        self._pending.extend(frame.audio)
        return False

    async def _send_task_handler(self):
        """Send one ~40ms frame every tick: queued audio, or silence when idle."""
        chunk = int(self.sample_rate * FRAME_S) * 2  # 16-bit mono
        silence = b"\x00" * chunk
        next_send = time.monotonic()
        while True:
            if len(self._pending) >= chunk:
                pcm = bytes(self._pending[:chunk])
                del self._pending[:chunk]
            elif self._pending:
                # Pad the utterance's final partial chunk to a full frame.
                pcm = bytes(self._pending) + silence[len(self._pending) :]
                self._pending.clear()
            else:
                pcm = silence
            frame = OutputAudioRawFrame(
                audio=pcm,
                sample_rate=self.sample_rate,
                num_channels=self._params.audio_out_channels,
            )
            await self._send_frame(frame)
            # Push every frame (audio and silence) downstream at this paced cadence:
            # the harness recorder aligns tracks by wall-clock, so a continuous
            # stream keeps the user turn at the right time (pushing only audio would
            # leave the track idle and the recorder would misplace it).
            await self.push_frame(frame)
            next_send += FRAME_S
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def _send_frame(self, frame: OutputAudioRawFrame):
        """Serialize and send one frame (raw PCM, via the RTVI serializer)."""
        if self._session.is_closing or not self._session.is_connected:
            return
        await self._write_frame(frame)

    async def _cancel_send_task(self):
        if self._send_task is not None:
            await self.cancel_task(self._send_task)
            self._send_task = None


class EvalHarnessInputTransport(WebsocketClientInputTransport):
    """Feeds the bot's audio to the harness's STT/VAD as a continuous stream.

    The bot transmits audio only while its TTS is producing — there are gaps during
    natural pauses (and after a turn). The harness runs a VAD on this audio to
    detect the bot's turn end; a gap reads as *no audio*, so the VAD's idle timeout
    force-stops the turn and splits it mid-sentence (e.g. "The capital of Germany."
    then "Is Berlin." as two turns). This transport buffers the bot's audio and a
    real-time task re-emits one ~40ms frame every tick — the next queued chunk when
    there is one, silence otherwise — so the VAD + smart-turn see speech-then-silence
    and judge the bot's real turn boundaries. The receive-side counterpart to
    :class:`EvalHarnessOutputTransport`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the transport and its (lazily started) fill stream."""
        super().__init__(*args, **kwargs)
        self._bot_pcm = bytearray()
        self._fill_task = None

    async def start(self, frame: StartFrame):
        """Start the transport and, when audio is enabled, the gap-fill stream."""
        await super().start(frame)
        if self._params.audio_in_enabled and self._fill_task is None:
            self._fill_task = self.create_task(self._fill_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the fill stream, then the transport."""
        await self._cancel_fill_task()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the fill stream, then the transport."""
        await self._cancel_fill_task()
        await super().cancel(frame)

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        """Buffer the bot's audio; the fill task re-emits it at a steady cadence.

        ``on_message`` routes the bot's incoming audio here; instead of pushing it
        straight through (gaps and all) we buffer it, and :meth:`_fill_task_handler`
        paces it downstream with silence filling the gaps.
        """
        self._bot_pcm.extend(frame.audio)

    async def _fill_task_handler(self):
        """Push one ~40ms frame every tick: buffered bot audio, or silence."""
        chunk = int(self.sample_rate * FRAME_S) * 2  # 16-bit mono
        silence = b"\x00" * chunk
        next_send = time.monotonic()
        while True:
            if len(self._bot_pcm) >= chunk:
                pcm = bytes(self._bot_pcm[:chunk])
                del self._bot_pcm[:chunk]
            elif self._bot_pcm:
                pcm = bytes(self._bot_pcm) + silence[len(self._bot_pcm) :]
                self._bot_pcm.clear()
            else:
                pcm = silence
            # super() pushes through the normal input audio path (VAD/STT).
            await super().push_audio_frame(
                InputAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1)
            )
            next_send += FRAME_S
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def _cancel_fill_task(self):
        if self._fill_task is not None:
            await self.cancel_task(self._fill_task)
            self._fill_task = None


class EvalHarnessTransport(RTVIClientTransport):
    """RTVI client transport whose audio edges behave like a live transport.

    Identical to :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
    except that ``input()`` returns an :class:`EvalHarnessInputTransport` and
    ``output()`` an :class:`EvalHarnessOutputTransport` — both reshaping the audio
    into the continuous real-time stream VAD/STT expect.
    """

    def input(self) -> WebsocketClientInputTransport:
        """Return the gap-filling input transport."""
        if not self._input:
            self._input = EvalHarnessInputTransport(self, self._session, self._params)
        return self._input

    def output(self) -> WebsocketClientOutputTransport:
        """Return the stream-shaping output transport."""
        if not self._output:
            self._output = EvalHarnessOutputTransport(self, self._session, self._params)
        return self._output
