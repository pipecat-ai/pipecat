#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcription logging observer for Pipecat.

This module provides an observer that logs transcription frames to the console,
allowing developers to monitor speech-to-text activity in real-time.
"""

from loguru import logger

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.services.stt_service import STTService


class TranscriptionLogObserver(BaseObserver):
    """Observer to log transcription activity to the console.

    Monitors and logs all transcription frames from STT services, including
    both final transcriptions and interim results. This allows developers
    to track speech recognition activity and debug transcription issues.

    Only processes frames from STTService instances to avoid logging
    unrelated transcription frames from other sources.
    """

    async def on_push_frame(self, data: FramePushed):
        """Handle frame push events and log transcription frames.

        Logs TranscriptionFrame and InterimTranscriptionFrame instances
        with timestamps and user information for debugging purposes.

        Args:
            data: Frame push event data containing source, frame, and timestamp.
        """
        src = data.source
        frame = data.frame
        timestamp = data.timestamp

        if not isinstance(src, STTService):
            return

        time_sec = timestamp / 1_000_000_000

        arrow = "→"

        if isinstance(frame, TranscriptionFrame):
            logger.debug(
                f"💬 {src} {arrow} TRANSCRIPTION: {frame.text!r} from {frame.user_id!r} at {time_sec:.2f}s"
            )
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug(
                f"💬 {src} {arrow} INTERIM TRANSCRIPTION: {frame.text!r} from {frame.user_id!r} at {time_sec:.2f}s"
            )
