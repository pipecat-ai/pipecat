#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bot-side silence padder for recording fidelity.

When the bot is silent, downstream recording processors such as
:class:`pipecat.processors.audio.audio_buffer_processor.AudioBufferProcessor`
only advance the bot recording buffer with silence when an
``InputAudioRawFrame`` arrives. If no user input audio is flowing for some
stretch of wall-clock time (push-to-talk transports, server-side input gating
during function/tool calls, manual muting between turns), the bot recording
buffer freezes during that gap. The next bot utterance is then appended
directly after the previous one and the saved recording loses the real-time
gap — two utterances spoken seconds apart sound concatenated.

This processor sits between ``transport.output()`` and the recording
processor and, while the bot is not speaking, emits silent
``OutputAudioRawFrame`` chunks downstream. The recording processor receives
them and the bot recording buffer advances in sync with wall-clock time.

The emitted frames are pushed downstream only, so the upstream output
transport never sees them — nothing is played out the line and no
``BotStarted/Stopped`` events are emitted.
"""

import asyncio
from typing import Optional

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotRecordingSilencePadder(FrameProcessor):
    """Emit silent ``OutputAudioRawFrame`` downstream while the bot is quiet.

    Place this processor between ``transport.output()`` and the recording
    processor (e.g. :class:`AudioBufferProcessor`) in the pipeline. While the
    bot is not speaking, the processor pushes silent
    ``OutputAudioRawFrame`` chunks downstream at ``chunk_ms`` intervals so the
    bot recording buffer keeps pace with wall-clock time even when user input
    audio is not flowing.

    Args:
        chunk_ms: Padding chunk length in milliseconds. Clamped to a minimum
            of 20 ms. Defaults to 100.
        **kwargs: Additional arguments passed to the parent class.
    """

    def __init__(self, *, chunk_ms: int = 100, **kwargs):
        """Initialise the padder.

        Args:
            chunk_ms: Silence chunk length in milliseconds (minimum 20).
        """
        super().__init__(**kwargs)
        self._chunk_ms = max(20, int(chunk_ms))

        self._sample_rate: Optional[int] = None
        self._bot_speaking = False

        # Pinged on any state change so the silence loop wakes promptly.
        self._state_changed = asyncio.Event()
        self._silence_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage the silence-padding loop.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._sample_rate = frame.audio_out_sample_rate
            self._ensure_silence_task_running()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._state_changed.set()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._state_changed.set()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()

        await self.push_frame(frame, direction)

    def _ensure_silence_task_running(self):
        """Start the silence loop if not already running."""
        if self._sample_rate is None:
            return
        if self._silence_task is None or self._silence_task.done():
            self._silence_task = self.create_task(self._silence_loop())

    async def _silence_loop(self):
        """Push silent ``OutputAudioRawFrame`` while the bot is not speaking."""
        chunk_secs = self._chunk_ms / 1000.0
        sample_rate = self._sample_rate
        if sample_rate is None:
            return
        num_samples = int(sample_rate * chunk_secs)
        # 16-bit mono PCM silence.
        silence_bytes = b"\x00" * (num_samples * 2)

        try:
            while True:
                self._state_changed.clear()
                if self._bot_speaking:
                    # Bot is speaking; wait quietly. Re-check on state change.
                    try:
                        await asyncio.wait_for(self._state_changed.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                # Push one chunk of silence so the recorder advances the bot
                # buffer in sync with wall-clock time.
                await self.push_frame(
                    OutputAudioRawFrame(
                        audio=silence_bytes,
                        sample_rate=sample_rate,
                        num_channels=1,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                # Sleep until the next tick or wake early on state change.
                try:
                    await asyncio.wait_for(self._state_changed.wait(), timeout=chunk_secs)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _stop(self):
        """Stop the silence loop."""
        self._state_changed.set()
        if self._silence_task is not None and not self._silence_task.done():
            await self.cancel_task(self._silence_task)
        self._silence_task = None

    async def cleanup(self):
        """Clean up processor resources."""
        await super().cleanup()
        await self._stop()
