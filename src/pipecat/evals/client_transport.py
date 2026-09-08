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

- :class:`EvalClientOutputTransport` (send side): the user TTS's audio is enqueued
  and a real-time task ships one ~40ms frame every tick (queued audio when
  available, silence when idle), so the bot receives a continuous stream and its
  stock input handles it directly.
- :class:`EvalClientInputTransport` (receive side): the bot's audio arrives in
  bursts with gaps; it is buffered and re-emitted at the same steady cadence, so
  the harness's VAD + smart-turn see speech-then-silence and judge the bot's real
  turn boundaries — instead of the VAD's idle timeout force-stopping a turn at a
  pause and splitting it mid-sentence.

The pacing and gap-filling exist for the bot and the VADs, where jitter is
harmless. The *recording* must not depend on them: Python can't hold the ~40ms
tick precisely, and a pacing underrun becomes silence wedged mid-word, so
recording the paced/filled streams stutters. Instead :class:`EvalClientRecorder` is
fed the *raw* audio on each edge -- the user's TTS as produced, the bot's chunks
as received, both gapless within a turn -- and reconstructs the recording from
those (see its docstring), padding only the real between-turn pauses.

:class:`EvalClientTransport` is the RTVI client transport the harness builds; it
differs from :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
only in returning these two stream-shaping transports and wiring the recorder.

The harness's bot-audio VAD broadcasts interruptions when the bot speaks; those are
stopped at the harness sink before they reach the output transport (see
``_BotFrameSink``), so the paced user audio is never flushed mid-utterance.
"""

import asyncio
import time
import wave
from pathlib import Path

from pipecat.audio.utils import create_stream_resampler, mix_audio
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.client import (
    WebsocketClientInputTransport,
    WebsocketClientOutputTransport,
)
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport

# One audio frame per tick, on both edges. 40ms matches the bot's output chunking
# (``audio_out_10ms_chunks`` defaults to 4), so the receive side re-emits the bot's
# chunks 1:1 rather than splitting them, and it's still fine-grained enough for the
# VAD/turn models (well under their start/stop windows).
FRAME_S = 0.04


async def _sleep_to_next_tick(next_send: float) -> float:
    """Pace to the next ~``FRAME_S`` boundary, re-anchoring when behind.

    Returns the next target time. When the loop has fallen behind (after a hiccup
    such as the worker's startup model load, a GC pause, or inline VAD), the target
    is already in the past; re-anchor to ``now + FRAME_S`` rather than firing a
    burst of catch-up frames. A catch-up burst compresses audio in time and the
    wall-clock recorder then renders it as a stutter. This mirrors the transport's
    own :meth:`_write_audio_sleep`.
    """
    sleep_s = max(0.0, next_send - time.monotonic())
    await asyncio.sleep(sleep_s)
    return time.monotonic() + FRAME_S if sleep_s == 0 else next_send + FRAME_S


class EvalClientRecorder:
    """Builds the conversation recording from raw source audio, decoupled from pacing.

    The harness paces audio to the bot and fills gaps in the bot's audio for the
    VADs, but Python can't hold a 40ms tick precisely; recording those paced/filled
    streams stutters, because a pacing underrun becomes silence wedged mid-word.
    This recorder is fed the *raw* audio instead -- the user's TTS as produced and
    the bot's chunks as received (both gapless within a turn) -- and lays each side
    out contiguously on its playout timeline, inserting silence only for a real
    between-turn pause (see :class:`_RecorderTrack`). Pacing jitter never reaches
    the recording, and audio the bot sent past its interruption is dropped
    (:meth:`drop_bot_tail`) as a client's playout would drop it.

    Each side is captured on its own timeline (monotonic clock); :meth:`write`
    resamples to a common rate, aligns the two by their first-audio offset, and
    mixes to mono.
    """

    # A chunk arriving later than its side's playout position by more than this
    # is a real pause between turns; anything shorter is jitter within a
    # contiguous turn and stays gapless.
    GAP_S = 0.2

    def __init__(self, sample_rate: int):
        """Initialize the recorder.

        Args:
            sample_rate: Output sample rate; both sides are resampled to it on write.
        """
        self._rate = sample_rate
        self._user = _RecorderTrack()
        self._bot = _RecorderTrack()

    def add_user(self, audio: bytes, in_rate: int) -> None:
        """Record a chunk of the user's TTS audio (raw, before pacing to the bot)."""
        self._user.add(audio, in_rate)

    def add_bot(self, audio: bytes, in_rate: int) -> None:
        """Record a chunk of the bot's audio (raw, before the gap-fill loop)."""
        self._bot.add(audio, in_rate)

    def drop_bot_tail(self, num_bytes: int) -> None:
        """Drop the last ``num_bytes`` of the bot's audio: sent, but never played."""
        self._bot.drop_tail(num_bytes)

    def drop_user_tail(self, num_bytes: int) -> None:
        """Drop the last ``num_bytes`` of the user's audio: synthesized, but never sent."""
        self._user.drop_tail(num_bytes)

    def has_audio(self) -> bool:
        """Whether any audio has been recorded on either side."""
        return self._user.first is not None or self._bot.first is not None

    async def write(self, path: str) -> bool:
        """Resample both sides to a common rate, align, mix to mono, and write a WAV.

        Returns:
            True if a file was written, False if nothing was recorded.
        """
        firsts = [t.first for t in (self._user, self._bot) if t.first is not None]
        if not firsts:
            return False
        start = min(firsts)
        user = await self._user.rendered(self._rate, start)
        bot = await self._bot.rendered(self._rate, start)
        mixed = mix_audio(user, bot)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            wf.writeframes(mixed)
        return True


class _RecorderTrack:
    """One side of the recording: raw audio chunks and when each arrived.

    :meth:`rendered` lays the chunks out contiguously on a playout timeline: each
    starts where the previous one ends, and silence is inserted only for a real
    pause -- a chunk arriving later than the playout position by more than
    ``EvalClientRecorder.GAP_S``. Both sources run ahead of real time (the bot's
    transport sends up to twice real time, the user TTS synthesizes faster than
    it plays), so within a turn chunks arrive before the position and a hiccup
    on the receiving side is absorbed by that lead; a small late arrival is
    jitter and shifts the timeline rather than accumulating toward a pause.
    """

    def __init__(self):
        self._chunks: list[tuple[float, bytes]] = []
        self._rate: int | None = None

    @property
    def first(self) -> float | None:
        """When the first chunk arrived, or ``None`` before any audio."""
        return self._chunks[0][0] if self._chunks else None

    def add(self, audio: bytes, in_rate: int) -> None:
        """Record a raw audio chunk as of now."""
        if not audio:
            return
        self._rate = in_rate
        self._chunks.append((time.monotonic(), audio))

    def drop_tail(self, num_bytes: int) -> None:
        """Remove the last ``num_bytes`` of audio, across as many chunks as it spans."""
        while num_bytes > 0 and self._chunks:
            arrived, chunk = self._chunks[-1]
            if len(chunk) <= num_bytes:
                self._chunks.pop()
                num_bytes -= len(chunk)
            else:
                self._chunks[-1] = (arrived, chunk[: len(chunk) - num_bytes])
                num_bytes = 0

    async def rendered(self, out_rate: int, start: float) -> bytes:
        """Lay the track out, resample to ``out_rate``, and prepend its silence since ``start``."""
        if self.first is None or self._rate is None:
            return b""
        audio = self._layout()
        if self._rate != out_rate:
            audio = await create_stream_resampler().resample(audio, self._rate, out_rate)
        lead = int((self.first - start) * out_rate * 2) & ~1
        return b"\x00" * lead + audio

    def _layout(self) -> bytes:
        """Concatenate the chunks, with silence for each real pause."""
        assert self._rate is not None
        out = bytearray()
        position = self._chunks[0][0]  # playout position, as a wall-clock time
        for arrived, chunk in self._chunks:
            late = arrived - position
            if late > EvalClientRecorder.GAP_S:
                out.extend(b"\x00" * (int(late * self._rate * 2) & ~1))
                position = arrived
            elif late > 0:
                position = arrived
            out.extend(chunk)
            position += len(chunk) / (self._rate * 2)
        return bytes(out)


class EvalClientOutputTransport(WebsocketClientOutputTransport):
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

    def __init__(self, *args, recorder: "EvalClientRecorder | None" = None, **kwargs):
        """Initialize the transport and its (lazily started) send stream."""
        super().__init__(*args, **kwargs)
        self._pending = bytearray()
        self._send_task = None
        self._recorder = recorder

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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame; an interruption drops the user audio not yet sent.

        An interruption reaching this transport means the user side stopped
        talking (a persona hearing the bot speak over it), so what the TTS
        produced but the send stream hasn't paced out yet is dropped, and the
        recorder drops the same bytes.
        """
        if isinstance(frame, InterruptionFrame) and self._pending:
            if self._recorder is not None:
                self._recorder.drop_user_tail(len(self._pending))
            self._pending.clear()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Enqueue the user audio; the send task paces it onto the wire.

        Returns False so the media sender does *not* push this (un-paced) frame
        downstream: the media sender drains its queue in a burst, which would
        collapse the user's turn in the harness recording. The send task pushes the
        paced frames downstream instead (see :meth:`_send_task_handler`).
        """
        if self._session.is_closing or not self._session.is_connected:
            return False
        # Record the raw TTS audio here (gapless), not the paced frames the send
        # task emits: pacing underruns/jitter would stutter the recording.
        if self._recorder is not None:
            self._recorder.add_user(frame.audio, frame.sample_rate)
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
            next_send = await _sleep_to_next_tick(next_send)

    async def _send_frame(self, frame: OutputAudioRawFrame):
        """Serialize and send one frame (raw PCM, via the RTVI serializer)."""
        if self._session.is_closing or not self._session.is_connected:
            return
        await self._write_frame(frame)

    async def _cancel_send_task(self):
        if self._send_task is not None:
            await self.cancel_task(self._send_task)
            self._send_task = None


class EvalClientInputTransport(WebsocketClientInputTransport):
    """Feeds the bot's audio to the harness's STT/VAD as a continuous stream.

    The bot transmits audio only while its TTS is producing — there are gaps during
    natural pauses (and after a turn). The harness runs a VAD on this audio to
    detect the bot's turn end; a gap reads as *no audio*, so the VAD's idle timeout
    force-stops the turn and splits it mid-sentence (e.g. "The capital of Germany."
    then "Is Berlin." as two turns). This transport buffers the bot's audio and a
    real-time task re-emits one ~40ms frame every tick — the next queued chunk when
    there is one, silence otherwise — so the VAD + smart-turn see speech-then-silence
    and judge the bot's real turn boundaries. The receive-side counterpart to
    :class:`EvalClientOutputTransport`.
    """

    def __init__(self, *args, recorder: "EvalClientRecorder | None" = None, **kwargs):
        """Initialize the transport and its (lazily started) fill stream."""
        super().__init__(*args, **kwargs)
        self._bot_pcm = bytearray()
        self._fill_task = None
        self._recorder = recorder

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

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame on, dropping the bot's unplayed audio when it reports an interruption.

        The bot's transport sends audio up to twice real time, so at an
        interruption the buffer here can hold seconds the bot sent but a client
        never plays; what the harness hears must stop where the bot stopped. The
        recorder drops the same bytes.
        """
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InputTransportMessageFrame)
            and isinstance(frame.message, dict)
            and frame.message.get("type") == "bot-interrupted"
        ):
            if self._recorder is not None:
                self._recorder.drop_bot_tail(len(self._bot_pcm))
            self._bot_pcm.clear()
        await super().push_frame(frame, direction)

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        """Buffer the bot's audio; the fill task re-emits it at a steady cadence.

        ``on_message`` routes the bot's incoming audio here; instead of pushing it
        straight through (gaps and all) we buffer it, and :meth:`_fill_task_handler`
        paces it downstream with silence filling the gaps.
        """
        # Record the raw bot audio here (gapless within a turn), not the gap-filled
        # frames the fill task emits: those underruns would stutter the recording.
        if self._recorder is not None:
            self._recorder.add_bot(frame.audio, frame.sample_rate)
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
            next_send = await _sleep_to_next_tick(next_send)

    async def _cancel_fill_task(self):
        if self._fill_task is not None:
            await self.cancel_task(self._fill_task)
            self._fill_task = None


class EvalClientTransport(RTVIClientTransport):
    """RTVI client transport whose audio edges behave like a live transport.

    Identical to :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
    except that ``input()`` returns an :class:`EvalClientInputTransport` and
    ``output()`` an :class:`EvalClientOutputTransport` — both reshaping the audio
    into the continuous real-time stream VAD/STT expect. When a ``recorder`` is
    given, both edges feed it the *raw* audio (before pacing/filling) so the
    recording is gapless regardless of the pacing jitter.
    """

    def __init__(self, *args, recorder: "EvalClientRecorder | None" = None, **kwargs):
        """Initialize the transport, optionally wiring a recorder to both edges.

        Args:
            recorder: Optional :class:`EvalClientRecorder` fed the raw audio on both
                edges; ``None`` disables recording.
            *args: Forwarded to :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`.
            **kwargs: Forwarded to the parent transport.
        """
        super().__init__(*args, **kwargs)
        self._recorder = recorder

    def input(self) -> WebsocketClientInputTransport:
        """Return the gap-filling input transport."""
        if not self._input:
            self._input = EvalClientInputTransport(
                self, self._session, self._params, recorder=self._recorder
            )
        return self._input

    def output(self) -> WebsocketClientOutputTransport:
        """Return the stream-shaping output transport."""
        if not self._output:
            self._output = EvalClientOutputTransport(
                self, self._session, self._params, recorder=self._recorder
            )
        return self._output
