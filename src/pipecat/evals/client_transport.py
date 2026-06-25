#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Client-side WebSocket transport for the eval harness.

The harness drives the bot as an RTVI client. A real transport carries a
continuous real-time audio stream in both directions — speech while a party
talks, silence otherwise — and VAD, turn detection, and streaming STT all rely on
that cadence. The eval bot's WebSocket transport instead sends/receives audio only
while a TTS is producing, with gaps during pauses, so both edges must reconstruct
the continuous stream:

- :class:`EvalMicOutputTransport` (send side): the user TTS's audio is enqueued
  and a real-time task ships one ~20ms frame every tick (queued audio when
  available, silence when idle), so the bot receives exactly what a live mic would
  produce and needs no virtual mic of its own.
- :class:`EvalBotAudioInputTransport` (receive side): the bot's audio arrives in
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

The harness's bot-audio VAD broadcasts interruptions when the bot speaks; those
are stopped at the harness sink before they reach the output transport (see
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

# One mic frame per tick — the granularity a live transport delivers and what
# VAD/turn models consume.
MIC_FRAME_S = 0.02


class EvalMicOutputTransport(WebsocketClientOutputTransport):
    """Streams audio to the bot like a live client's microphone.

    The default output transport sends each audio frame as it arrives (paced to
    real time, but with no audio in between), so the bot would see a whole
    utterance with no trailing silence — VADs and turn detectors need that
    silence to end a turn. This transport instead enqueues the user TTS's audio
    and a real-time task emits one ~20ms frame every tick: the next queued chunk
    when there is one, silence otherwise. The result is the continuous stream a
    live mic produces, so the bot's stock input handles it with no virtual mic.

    The stream runs only when audio output is enabled (audio-mode scenarios); a
    text-mode scenario sends nothing, so no silence is ever fed to the bot's STT.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the transport and its (lazily started) mic stream."""
        super().__init__(*args, **kwargs)
        self._mic_pcm = bytearray()
        self._mic_task = None

    async def start(self, frame: StartFrame):
        """Start the transport and, in audio mode, the real-time mic stream."""
        await super().start(frame)
        if self._params.audio_out_enabled and self._mic_task is None:
            self._mic_task = self.create_task(self._mic_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the mic stream, then the transport."""
        await self._cancel_mic_task()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the mic stream, then the transport."""
        await self._cancel_mic_task()
        await super().cancel(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Enqueue the user audio; the mic task paces it onto the wire.

        Returns False so the media sender does *not* push this (un-paced) frame
        downstream: the media sender drains its queue in a burst, which would
        collapse the user's turn in the harness recording. The mic task pushes the
        paced frames downstream instead (see :meth:`_mic_task_handler`).
        """
        if self._session.is_closing or not self._session.is_connected:
            return False
        self._mic_pcm.extend(frame.audio)
        return False

    async def _mic_task_handler(self):
        """Send one ~20ms frame every tick: queued audio, or silence when idle."""
        chunk = int(self.sample_rate * MIC_FRAME_S) * 2  # 16-bit mono
        silence = b"\x00" * chunk
        next_send = time.monotonic()
        while True:
            if len(self._mic_pcm) >= chunk:
                pcm = bytes(self._mic_pcm[:chunk])
                del self._mic_pcm[:chunk]
            elif self._mic_pcm:
                # Pad the utterance's final partial chunk to a full frame.
                pcm = bytes(self._mic_pcm) + silence[len(self._mic_pcm) :]
                self._mic_pcm.clear()
            else:
                pcm = silence
            frame = OutputAudioRawFrame(
                audio=pcm,
                sample_rate=self.sample_rate,
                num_channels=self._params.audio_out_channels,
            )
            await self._send_mic_frame(frame)
            # Push every frame (audio and silence) downstream at this paced cadence:
            # the harness recorder aligns tracks by wall-clock, so a continuous
            # stream keeps the user turn at the right time (pushing only audio would
            # leave the track idle and the recorder would misplace it).
            await self.push_frame(frame)
            next_send += MIC_FRAME_S
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def _send_mic_frame(self, frame: OutputAudioRawFrame):
        """Serialize and send one mic frame (raw PCM, via the RTVI serializer)."""
        if self._session.is_closing or not self._session.is_connected:
            return
        await self._write_frame(frame)

    async def _cancel_mic_task(self):
        if self._mic_task is not None:
            await self.cancel_task(self._mic_task)
            self._mic_task = None


class EvalBotAudioInputTransport(WebsocketClientInputTransport):
    """Feeds the bot's audio to the harness's STT/VAD as a continuous stream.

    The bot transmits audio only while its TTS is producing — there are gaps during
    natural pauses (and after a turn). The harness runs a VAD on this audio to
    detect the bot's turn end; a gap reads as *no audio*, so the VAD's idle timeout
    force-stops the turn and splits it mid-sentence (e.g. "The capital of Germany."
    then "Is Berlin." as two turns). This transport buffers the bot's audio and a
    real-time task re-emits one ~20ms frame every tick — the next queued chunk when
    there is one, silence otherwise — so the VAD + smart-turn see speech-then-silence
    and judge the bot's real turn boundaries. Mirror of :class:`EvalMicOutputTransport`.
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
        """Push one ~20ms frame every tick: buffered bot audio, or silence."""
        chunk = int(self.sample_rate * MIC_FRAME_S) * 2  # 16-bit mono
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
            next_send += MIC_FRAME_S
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def _cancel_fill_task(self):
        if self._fill_task is not None:
            await self.cancel_task(self._fill_task)
            self._fill_task = None


class EvalHarnessTransport(RTVIClientTransport):
    """RTVI client transport whose audio edges behave like a live transport.

    Identical to :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
    except that ``output()`` returns an :class:`EvalMicOutputTransport` and
    ``input()`` an :class:`EvalBotAudioInputTransport` — both reshaping the audio
    into the continuous real-time stream VAD/STT expect.
    """

    def input(self) -> WebsocketClientInputTransport:
        """Return the gap-filling input transport."""
        if not self._input:
            self._input = EvalBotAudioInputTransport(self, self._session, self._params)
        return self._input

    def output(self) -> WebsocketClientOutputTransport:
        """Return the mic-like output transport."""
        if not self._output:
            self._output = EvalMicOutputTransport(self, self._session, self._params)
        return self._output
