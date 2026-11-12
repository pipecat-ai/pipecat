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
lays each side out on its own timeline.
"""

import asyncio
import json
import time
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.utils import create_file_resampler, mix_audio
from pipecat.evals.events import (
    BOT_AUDIO_RECEIVED_EVENT,
    USER_AUDIO_SENT_EVENT,
    USER_TEXT_SENT_EVENT,
)
from pipecat.evals.serializer import (
    EVAL_CAPABILITIES_MESSAGE_TYPE,
    EVAL_STT_SAMPLE_RATE,
    EvalAudioCapabilities,
    EvalAudioLimitError,
    EvalConnectionFlags,
    _bounded_json,
    _check_audio_message_size,
    _checked_audio_limit,
    eval_connect_message,
    parse_eval_capabilities,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
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

AudioFailure = Callable[[str], Awaitable[None]]
TransportTiming = Callable[[str, float], Awaitable[None]]


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
    pause was, and mixed to mono at :meth:`write`. Audio the bot sent past an
    interruption is dropped, as a real client would drop it.
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
            audio = await create_file_resampler().resample(audio, self._rate, out_rate)
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
    """

    def __init__(
        self,
        *args,
        recorder: "EvalClientRecorder | None" = None,
        timing: TransportTiming | None = None,
        **kwargs,
    ):
        """Initialize the transport and its (lazily started) send stream."""
        super().__init__(*args, **kwargs)
        self._pending = bytearray()
        self._send_task = None
        self._recorder = recorder
        self._timing = timing
        self._queued_bytes = 0
        self._consumed_bytes = 0
        self._consumed = asyncio.Event()
        self._audio_generation = 0
        self._utterance_last_sent_at: float | None = None

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
        if isinstance(frame, InterruptionFrame):
            if self._recorder is not None and self._pending:
                self._recorder.drop_user_tail(len(self._pending))
            self._drop_pending()
            self._utterance_last_sent_at = None
        await super().process_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        """Time completed caller utterances after the paced audio drained."""
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._utterance_last_sent_at = None
            elif isinstance(frame, BotStoppedSpeakingFrame):
                generation = self._audio_generation
                sent_by = self._queued_bytes
                while (
                    self._send_task is not None
                    and self._consumed_bytes < sent_by
                    and generation == self._audio_generation
                ):
                    self._consumed.clear()
                    await self._consumed.wait()
                if generation == self._audio_generation:
                    observed_at = self._utterance_last_sent_at
                    self._utterance_last_sent_at = None
                    if self._timing is not None and observed_at is not None:
                        await self._timing(USER_AUDIO_SENT_EVENT, observed_at)
        await super().push_frame(frame, direction)

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Time a text input only after the transport accepts its send."""
        sent = await self._write_frame(frame)
        if isinstance(frame.message, dict) and frame.message.get("type") == "send-text":
            if not sent:
                await self.push_error("The eval transport could not send text")
            elif self._timing is not None:
                await self._timing(USER_TEXT_SENT_EVENT, time.monotonic())

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Queue user audio immediately so smaller media frames can be combined.

        Returns False so the media sender does not push this un-paced frame
        downstream; the send task pushes the paced frames instead. The utterance
        end marker waits for delivery of the assembled audio.
        """
        if self._session.is_closing or not self._session.is_connected:
            return False
        # Record the raw TTS audio here (gapless), not the paced frames the send
        # task emits: pacing underruns/jitter would stutter the recording.
        if self._recorder is not None:
            self._recorder.add_user(frame.audio, frame.sample_rate)
        self._pending.extend(frame.audio)
        self._queued_bytes += len(frame.audio)
        return False

    async def _send_task_handler(self):
        """Run the paced sender, releasing utterance acknowledgments when it stops."""
        try:
            await self._send_audio()
        except Exception as exc:
            await self.push_error(f"The eval audio send stream failed ({type(exc).__name__})")
        finally:
            self._send_task = None
            self._drop_pending()
            self._utterance_last_sent_at = None

    async def _send_audio(self):
        """Send one ~40ms frame every tick: queued audio, or silence when idle."""
        chunk = int(self.sample_rate * FRAME_S) * 2  # 16-bit mono
        silence = b"\x00" * chunk
        next_send = time.monotonic()
        while True:
            generation = self._audio_generation
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
            sent = await self._send_frame(frame)
            if sent is False:
                self._drop_pending()
                self._utterance_last_sent_at = None
                await self.push_error("The eval transport could not send audio")
                return
            if pcm is not silence and generation == self._audio_generation:
                self._utterance_last_sent_at = time.monotonic()
                self._consumed_bytes = self._queued_bytes - len(self._pending)
                self._consumed.set()
            # Push every frame (audio and silence) downstream at this paced cadence:
            # the harness recorder aligns tracks by wall-clock, so a continuous
            # stream keeps the user turn at the right time (pushing only audio would
            # leave the track idle and the recorder would misplace it).
            await self.push_frame(frame)
            next_send = await _sleep_to_next_tick(next_send)

    async def _send_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Serialize and send one frame (raw PCM, via the RTVI serializer)."""
        if self._session.is_closing or not self._session.is_connected:
            return False
        return await self._write_frame(frame)

    def _drop_pending(self) -> None:
        """Release utterance acknowledgments whose queued audio was discarded."""
        self._audio_generation += 1
        self._pending.clear()
        self._consumed_bytes = self._queued_bytes
        self._consumed.set()

    async def _cancel_send_task(self):
        if self._send_task is not None:
            await self.cancel_task(self._send_task)
            self._send_task = None
        self._drop_pending()
        self._utterance_last_sent_at = None


class EvalClientInputTransport(WebsocketClientInputTransport):
    """Feeds the bot's audio to the harness's VAD and STT as a continuous stream.

    The bot sends audio only while its TTS produces it, with gaps at its
    pauses. A gap reads as silence to a VAD, which would end the bot's turn
    mid-sentence, so the audio is buffered and re-emitted one 40 ms frame per
    tick, silence filling the gaps.
    """

    def __init__(
        self,
        *args,
        recorder: "EvalClientRecorder | None" = None,
        max_bot_audio_bytes: int | None = None,
        audio_failure: AudioFailure | None = None,
        timing: TransportTiming | None = None,
        **kwargs,
    ):
        """Initialize the transport and its (lazily started) fill stream."""
        super().__init__(*args, **kwargs)
        self._bot_pcm = bytearray()
        self._fill_task = None
        self._recorder = recorder
        self._max_bot_audio_bytes = _checked_audio_limit(max_bot_audio_bytes)
        self._raw_audio_bytes = 0
        self._paced_audio_bytes = 0
        self._audio_failure = audio_failure
        self._audio_failed = False
        self._audio_closed = False
        self._timing = timing
        self._bot_audio_received = False

    async def start(self, frame: StartFrame):
        """Start the transport and, when audio is enabled, the gap-fill stream."""
        await super().start(frame)
        if self._params.audio_in_enabled and self._fill_task is None:
            self._fill_task = self.create_task(self._fill_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the fill stream, then the transport."""
        await self.close_audio()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the fill stream, then the transport."""
        await self.close_audio()
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
            self._bot_audio_received = False
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_audio_received = False
        await super().push_frame(frame, direction)

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        """Buffer the bot's audio for the fill task."""
        if self._audio_failed or self._audio_closed:
            return
        if self._max_bot_audio_bytes is not None:
            if (
                type(frame.sample_rate) is not int
                or frame.sample_rate != EVAL_STT_SAMPLE_RATE
                or type(frame.num_channels) is not int
                or frame.num_channels != 1
                or len(frame.audio) % 2
            ):
                await self.fail_audio("The eval target sent invalid PCM audio")
                return
            if len(frame.audio) > self._max_bot_audio_bytes - self._raw_audio_bytes:
                await self.fail_audio("The eval target exceeded its PCM admission limit")
                return
            self._raw_audio_bytes += len(frame.audio)
        observed_at: float | None = None
        if frame.audio and not self._bot_audio_received:
            self._bot_audio_received = True
            observed_at = time.monotonic()
        # Record the raw bot audio here (gapless within a turn), not the gap-filled
        # frames the fill task emits: those underruns would stutter the recording.
        if self._recorder is not None:
            self._recorder.add_bot(frame.audio, frame.sample_rate)
        self._bot_pcm.extend(frame.audio)
        if self._timing is not None and observed_at is not None:
            await self._timing(BOT_AUDIO_RECEIVED_EVENT, observed_at)

    async def fail_audio(self, reason: str) -> None:
        """Invalidate the run without waiting behind downstream audio queues."""
        if self._audio_failed or self._audio_closed:
            return
        self._audio_failed = True
        self._bot_pcm.clear()
        task = self._fill_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        if self._audio_failure is not None:
            await self._audio_failure(reason)
        else:
            await self.push_error(reason)

    async def close_audio(self) -> None:
        """Stop audio admission and gap fill, leaving control messages available."""
        self._audio_closed = True
        await self._cancel_fill_task()
        self._bot_pcm.clear()

    async def _fill_task_handler(self):
        """Push one ~40ms frame every tick: buffered bot audio, or silence."""
        if self._max_bot_audio_bytes is not None and (
            type(self.sample_rate) is not int
            or self.sample_rate != EVAL_STT_SAMPLE_RATE
            or type(self._params.audio_in_channels) is not int
            or self._params.audio_in_channels != 1
        ):
            await self.fail_audio("The eval harness has an invalid PCM format")
            return
        chunk = int(self.sample_rate * FRAME_S) * 2  # 16-bit mono
        silence = b"\x00" * chunk
        next_send = time.monotonic()
        while not self._audio_failed and not self._audio_closed:
            if self._max_bot_audio_bytes is not None:
                if chunk > self._max_bot_audio_bytes - self._paced_audio_bytes:
                    await self.fail_audio("The eval harness exceeded its PCM pipeline limit")
                    return
                self._paced_audio_bytes += chunk
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
        self._bot_pcm.clear()


class EvalClientTransport(RTVIClientTransport):
    """The harness's RTVI client transport, with audio edges that behave like a live transport.

    The input fills the gaps in the bot's audio and the output paces the
    user's; both feed the recorder the raw audio when one is given. On connect
    it sends the connection's flags in an ``eval-connect`` message, ahead of
    the RTVI handshake.
    """

    def __init__(
        self,
        *args,
        flags: EvalConnectionFlags | None = None,
        recorder: "EvalClientRecorder | None" = None,
        max_bot_audio_bytes: int | None = None,
        audio_failure: AudioFailure | None = None,
        timing: TransportTiming | None = None,
        **kwargs,
    ):
        """Initialize the transport, optionally wiring a recorder to both edges.

        Args:
            flags: The connection's flags, sent first on connect; ``None``
                sends the defaults.
            recorder: Optional :class:`EvalClientRecorder` fed the raw audio on both
                edges; ``None`` disables recording.
            max_bot_audio_bytes: Optional independent raw and paced PCM quotas.
            audio_failure: Receives static failure reasons independently of audio queues.
            timing: Observes completed input delivery and raw bot audio receipt.
            *args: Forwarded to :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`.
            **kwargs: Forwarded to the parent transport.
        """
        super().__init__(*args, **kwargs)
        self._flags = flags or EvalConnectionFlags()
        self._recorder = recorder
        self._max_bot_audio_bytes = _checked_audio_limit(max_bot_audio_bytes)
        self._audio_failure = audio_failure
        self._timing = timing
        self._audio_capabilities: EvalAudioCapabilities | None = None

    @property
    def audio_capabilities(self) -> EvalAudioCapabilities | None:
        """The target's advertised audio configuration, or ``None`` when unknown."""
        return self._audio_capabilities

    async def _on_connected(self, websocket):
        """Send ``eval-connect`` first, then fire the user's handler and start the handshake."""
        self._audio_capabilities = None
        await self._session.send(eval_connect_message(self._flags))
        await super()._on_connected(websocket)

    async def _on_message(self, websocket, message):
        """Refuse oversized or invalid bounded input, and read target capabilities, before the RTVI ready parser."""
        if self._max_bot_audio_bytes is not None:
            try:
                _check_audio_message_size(message)
                if not self.bot_ready:
                    _bounded_json(message)
            except EvalAudioLimitError as exc:
                await self.input().fail_audio(str(exc))
                return
        if not self.bot_ready and self._read_capabilities(message):
            return
        try:
            await super()._on_message(websocket, message)
        except EvalAudioLimitError as exc:
            await self.input().fail_audio(str(exc))

    def _read_capabilities(self, message) -> bool:
        """Record an ``eval-capabilities`` advertisement; whether ``message`` was one."""
        try:
            decoded = json.loads(message)
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
            return False
        if (
            isinstance(decoded, dict)
            and decoded.get("label") == RTVI.MESSAGE_LABEL
            and decoded.get("type") == EVAL_CAPABILITIES_MESSAGE_TYPE
        ):
            self._audio_capabilities = parse_eval_capabilities(decoded)
            return True
        return False

    def input(self) -> EvalClientInputTransport:
        """Return the gap-filling input transport."""
        if not self._input:
            self._input = EvalClientInputTransport(
                self,
                self._session,
                self._params,
                recorder=self._recorder,
                max_bot_audio_bytes=self._max_bot_audio_bytes,
                audio_failure=self._audio_failure,
                timing=self._timing,
            )
        return cast(EvalClientInputTransport, self._input)

    def output(self) -> WebsocketClientOutputTransport:
        """Return the stream-shaping output transport."""
        if not self._output:
            self._output = EvalClientOutputTransport(
                self, self._session, self._params, recorder=self._recorder, timing=self._timing
            )
        return self._output
