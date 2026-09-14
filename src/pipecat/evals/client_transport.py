#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The harness's WebSocket transport and recorder.

A live transport carries continuous audio both ways, speech or silence,
and VADs, turn detection, and streaming STT rely on that cadence. The eval
bot sends and receives audio only while a TTS produces it, so both edges
here reshape it: the output paces the user's audio to the bot one 40 ms
frame per tick, silence in between, and the input re-emits the bot's audio
at the same cadence, filling its gaps, so the harness's VAD finds the
bot's real turn ends.

The recording does not use the paced streams, whose jitter would make it
stutter: :class:`EvalClientRecorder` is fed the raw audio on both edges and
lays each side out on its own timeline, the user on the left channel and
the bot on the right.
"""

import asyncio
import time
import wave
from pathlib import Path

from pipecat.audio.utils import create_stream_resampler, interleave_stereo_audio
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
    """Sleep to the next tick and return the one after.

    When the loop has fallen behind, the next tick is measured from now rather
    than firing a burst of catch-up frames, which would compress the audio.
    """
    sleep_s = max(0.0, next_send - time.monotonic())
    await asyncio.sleep(sleep_s)
    return time.monotonic() + FRAME_S if sleep_s == 0 else next_send + FRAME_S


class EvalClientRecorder:
    """Builds the conversation recording from the raw audio, not the paced streams.

    Python cannot hold the 40 ms pacing tick precisely, and a recording of the
    paced streams stutters. So each side is recorded as it was produced or
    received, laid out on its own timeline with silence only where a real
    pause was, and written as stereo at :meth:`write`: the user on the left
    channel, the bot on the right, so overlaps and onsets can be read off
    the file. Audio the bot sent past an interruption is dropped, as a real
    client would drop it.
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
        """Resample both sides to a common rate, align, and write a stereo WAV.

        The user is the left channel and the bot the right; the shorter side
        is padded with silence to the longer.

        Returns:
            True if a file was written, False if nothing was recorded.
        """
        firsts = [t.first for t in (self._user, self._bot) if t.first is not None]
        if not firsts:
            return False
        start = min(firsts)
        user = await self._user.rendered(self._rate, start)
        bot = await self._bot.rendered(self._rate, start)
        length = max(len(user), len(bot))
        stereo = interleave_stereo_audio(user.ljust(length, b"\x00"), bot.ljust(length, b"\x00"))
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            wf.writeframes(stereo)
        return True


class _RecorderTrack:
    """One side of the recording: raw audio chunks and when each arrived.

    Chunks are laid out back to back. Silence is inserted only when a chunk
    arrives later than the playout position by more than the gap threshold,
    since within a turn both sources run ahead of real time.
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
    """Streams the user's audio to the bot as a continuous real-time stream.

    A real-time task sends one 40 ms frame per tick: queued TTS audio when
    there is some, silence otherwise, so the bot's VAD and turn detection see
    the silence they need to end a turn. Runs only when audio output is on.

    A write returns once the send task has sent its audio, the way a write to
    a sound device returns once the device took it. The base transport's
    ``BotStartedSpeakingFrame`` and ``BotStoppedSpeakingFrame`` (here: the
    *user's* utterance going out) then bracket the audio as it was sent, and
    the stop frame marks the end of the user's speech for whatever times the
    turn.
    """

    def __init__(self, *args, recorder: "EvalClientRecorder | None" = None, **kwargs):
        """Initialize the transport and its (lazily started) send stream."""
        super().__init__(*args, **kwargs)
        self._pending = bytearray()
        self._send_task = None
        self._recorder = recorder
        # Bytes queued for the send task and bytes it has consumed (sent, or
        # dropped by an interruption); a write waits for its own to be consumed.
        self._queued_bytes = 0
        self._consumed_bytes = 0
        self._consumed = asyncio.Event()

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
        """Pass a frame on; an interruption drops the user audio not yet sent, and the recorder drops the same bytes."""
        if isinstance(frame, InterruptionFrame) and self._pending:
            if self._recorder is not None:
                self._recorder.drop_user_tail(len(self._pending))
            self._drop_pending()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Queue the user audio for the send task, and wait until it has gone out.

        The media sender writes one chunk at a time, so the wait is at most a
        tick or two; an interruption or the transport stopping releases it.
        Returns False so the media sender does not push this un-paced frame
        downstream; the send task pushes the paced frames instead.
        """
        if self._session.is_closing or not self._session.is_connected:
            return False
        # Record the raw TTS audio here (gapless), not the paced frames the send
        # task emits: pacing underruns/jitter would stutter the recording.
        if self._recorder is not None:
            self._recorder.add_user(frame.audio, frame.sample_rate)
        self._pending.extend(frame.audio)
        self._queued_bytes += len(frame.audio)
        sent_by = self._queued_bytes
        # With no send task (the transport stopping) nothing would send it, so
        # nothing is waited for.
        while self._send_task is not None and self._consumed_bytes < sent_by:
            self._consumed.clear()
            await self._consumed.wait()
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
            if pcm is not silence:
                # The audio went out: the write that queued it may return.
                self._consumed_bytes = self._queued_bytes - len(self._pending)
                self._consumed.set()
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

    def _drop_pending(self):
        """Forget the queued audio, releasing the write waiting on it."""
        self._pending.clear()
        self._consumed_bytes = self._queued_bytes
        self._consumed.set()

    async def _cancel_send_task(self):
        if self._send_task is not None:
            await self.cancel_task(self._send_task)
            self._send_task = None
        # Nothing will send what is queued now; a write waiting on it returns.
        self._drop_pending()


class EvalClientInputTransport(WebsocketClientInputTransport):
    """Feeds the bot's audio to the harness's VAD and STT as a continuous stream.

    The bot sends audio only while its TTS produces it, with gaps at its
    pauses. A gap reads as silence to a VAD, which would end the bot's turn
    mid-sentence, so the audio is buffered and re-emitted one 40 ms frame per
    tick, silence filling the gaps.
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
        """Push a frame on; when the bot reports an interruption, the audio it sent but a client would never play is dropped, and the recorder drops the same bytes."""
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
        """Buffer the bot's audio for the fill task."""
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
    """The harness's RTVI client transport, with audio edges that behave like a live transport.

    The input fills the gaps in the bot's audio and the output paces the
    user's; both feed the recorder the raw audio when one is given.
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
