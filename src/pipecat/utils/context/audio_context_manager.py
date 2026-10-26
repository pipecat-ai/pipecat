#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio contexts a TTS service plays one after another."""

import asyncio
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.utils.frame_queue import FrameQueue


@dataclass
class AudioContextFrame(Frame):
    """Marks where an audio context plays among the queued frames.

    Parameters:
        context_id: The audio context to play.
    """

    context_id: str


@dataclass
class _AudioRemainder:
    """The bytes of a sample split across audio chunks, with their format."""

    audio: bytes
    sample_rate: int
    num_channels: int


class AudioContextManager:
    """Audio contexts and frames, played in the order they are queued.

    Each audio context has its own queue of items (audio, other frames, word
    timestamps), so a context's audio can arrive while an earlier one is still
    playing. :meth:`next_for_playback` returns the next context or frame to
    play, and :meth:`get` returns a context's items until the context ends.

    ``playing_context_id`` is the context :meth:`next_for_playback` returned
    last. The caller sets it back to ``None`` when that context stops playing.
    """

    _KEEPALIVE = object()

    def __init__(self):
        """Initialize the audio context manager."""
        self._contexts: dict[str, asyncio.Queue] = {}
        # Trailing bytes of a sample split across audio chunks, per context.
        self._remainders: dict[str, _AudioRemainder] = {}
        # Contexts and frames in play order. None ends playback.
        self._queue = FrameQueue(
            frame_getter=lambda item: item if isinstance(item, Frame) else None
        )
        self.playing_context_id: str | None = None

    @property
    def context_ids(self) -> list[str]:
        """The audio contexts that haven't finished playing."""
        return list(self._contexts.keys())

    def available(self, context_id: str) -> bool:
        """Whether an audio context exists and hasn't finished playing.

        Args:
            context_id: The audio context to check.
        """
        return context_id in self._contexts

    async def create(self, context_id: str, *, interruptible: bool = True):
        """Create an audio context and queue it to play after what's already queued.

        Args:
            context_id: The audio context to create.
            interruptible: Whether :meth:`reset` drops the context while it waits to play.
        """
        self._contexts[context_id] = asyncio.Queue()
        frame = AudioContextFrame(context_id)
        frame.interruptible = interruptible
        await self._queue.put(frame)

    async def append(self, context_id: str, item: Any):
        """Append an item to an audio context. ``None`` ends the context.

        Audio is trimmed to whole samples, and a split sample is carried to the
        context's next audio frame. Providers may cut their PCM stream at any
        byte, but consumers read audio frame by frame.

        Args:
            context_id: The audio context to append to. It must exist.
            item: The item to append, or ``None`` to end the context.
        """
        if isinstance(item, TTSAudioRawFrame) and not self._align_audio_frame(context_id, item):
            return
        # The context's audio ends here, so a held-back partial sample is
        # queued before the stop frame (or the end marker) that follows it.
        if item is None or isinstance(item, TTSStoppedFrame):
            await self._flush_audio_remainder(context_id)
        await self._contexts[context_id].put(item)

    def keepalive(self, context_id: str):
        """Restart the wait of :meth:`get` for an audio context, if it exists.

        Args:
            context_id: The audio context still in use.
        """
        if self.available(context_id):
            self._contexts[context_id].put_nowait(self._KEEPALIVE)

    async def put_frame(self, frame: Frame):
        """Queue a frame to play after what's already queued.

        Args:
            frame: The frame to queue.
        """
        await self._queue.put(frame)

    async def end_playback(self):
        """End playback once everything already queued has played."""
        await self._queue.put(None)

    async def next_for_playback(self) -> Frame | None:
        """Wait for the next audio context or frame to play.

        Contexts removed before they play are skipped.

        Returns:
            An :class:`AudioContextFrame`, whose context becomes the playing
            one, another queued frame, or ``None`` when playback ends.
        """
        item = await self._next_queued()
        while isinstance(item, AudioContextFrame) and not self.available(item.context_id):
            item = await self._next_queued()
        if isinstance(item, AudioContextFrame):
            self.playing_context_id = item.context_id
        return item

    async def get(self, context_id: str, timeout: float) -> Any:
        """Return an audio context's next item, waiting up to ``timeout`` for it.

        A keepalive restarts the wait and is not returned.

        Args:
            context_id: The audio context to read.
            timeout: Seconds to wait for an item.

        Returns:
            The next item, or ``None`` when the context ends.

        Raises:
            TimeoutError: If no item arrives in time.
        """
        queue = self._contexts[context_id]
        item = self._KEEPALIVE
        while item is self._KEEPALIVE:
            item = await asyncio.wait_for(queue.get(), timeout=timeout)
        return item

    def remove(self, context_id: str):
        """Remove an audio context and its queued items.

        Args:
            context_id: The audio context to remove.
        """
        self._contexts.pop(context_id, None)
        self._remainders.pop(context_id, None)

    def reset(self):
        """Drop the frames and contexts waiting to play, except uninterruptible ones.

        The dropped contexts stay available until they are removed.
        """
        self._queue.reset()

    async def _next_queued(self) -> Frame | None:
        """Take the next item in play order."""
        item = await self._queue.get()
        self._queue.task_done()
        return item

    def _align_audio_frame(self, context_id: str, frame: TTSAudioRawFrame) -> bool:
        """Trim an audio frame to whole samples, carrying a split sample forward.

        Returns:
            False if the frame holds less than one whole sample and should be
            dropped, True otherwise.
        """
        block_size = 2 * max(frame.num_channels, 1)
        remainder = self._remainders.pop(context_id, None)
        if not remainder and len(frame.audio) % block_size == 0:
            return True
        audio = (remainder.audio if remainder else b"") + frame.audio
        aligned_length = len(audio) - len(audio) % block_size
        if aligned_length < len(audio):
            self._remainders[context_id] = _AudioRemainder(
                audio[aligned_length:], frame.sample_rate, frame.num_channels
            )
        if aligned_length:
            frame.audio = audio[:aligned_length]
            frame.num_frames = aligned_length // block_size
        return aligned_length > 0

    async def _flush_audio_remainder(self, context_id: str):
        """Queue a context's held-back partial sample, zero-padded to a whole one."""
        remainder = self._remainders.pop(context_id, None)
        if not remainder:
            return
        block_size = 2 * max(remainder.num_channels, 1)
        audio = remainder.audio.ljust(block_size, b"\x00")
        frame = TTSAudioRawFrame(
            audio, remainder.sample_rate, remainder.num_channels, context_id=context_id
        )
        await self._contexts[context_id].put(frame)
